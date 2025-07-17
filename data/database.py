"""
Database layer for PCB inspection system.

This module handles data persistence, storage optimization, and database
operations for inspection results and analytics.
"""

import sqlite3
import json
import os
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import cv2
import numpy as np

from core.interfaces import BaseDatabase, InspectionResult
from core.config import DB_CONFIG, STORAGE_CONFIG, DEFECT_CLASSES


class PCBDatabase(BaseDatabase):
    """
    PCB inspection database with optimized storage and analytics support.
    
    Features:
    - SQLite database with thread-safe operations
    - Optimized storage (metadata only, selective image saving)
    - Built-in analytics queries
    - Automatic cleanup and archival
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DB_CONFIG["path"]
        self.storage_config = STORAGE_CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self.lock = threading.Lock()
        self.local = threading.local()
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Initialize database
        self._initialize_database()
        
        self.logger.info(f"Database initialized: {self.db_path}")
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            os.path.dirname(self.db_path),
            self.storage_config["images_dir"],
            self.storage_config["defects_dir"],
            "logs"
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self.local, 'connection'):
            self.local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self.local.connection.row_factory = sqlite3.Row
            
            # Enable WAL mode for better concurrency
            self.local.connection.execute("PRAGMA journal_mode=WAL")
            self.local.connection.execute("PRAGMA synchronous=NORMAL")
            self.local.connection.execute("PRAGMA temp_store=MEMORY")
            self.local.connection.execute("PRAGMA mmap_size=268435456")  # 256MB
            
        return self.local.connection
    
    def _initialize_database(self):
        """Initialize database schema."""
        with self.lock:
            conn = self._get_connection()
            
            # Create inspections table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    unix_timestamp REAL NOT NULL,
                    has_defects BOOLEAN NOT NULL,
                    defect_count INTEGER NOT NULL,
                    defects TEXT,
                    defect_locations TEXT,
                    confidence_scores TEXT,
                    focus_score REAL,
                    processing_time REAL,
                    image_path TEXT,
                    image_size_bytes INTEGER,
                    pcb_area INTEGER,
                    trigger_type TEXT DEFAULT 'auto',
                    session_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create defect statistics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS defect_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    defect_type TEXT NOT NULL,
                    total_count INTEGER DEFAULT 0,
                    total_confidence REAL DEFAULT 0.0,
                    avg_confidence REAL DEFAULT 0.0,
                    first_seen TEXT,
                    last_seen TEXT,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(defect_type)
                )
            ''')
            
            # Create system statistics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_inspections INTEGER DEFAULT 0,
                    total_defects INTEGER DEFAULT 0,
                    defect_rate REAL DEFAULT 0.0,
                    avg_processing_time REAL DEFAULT 0.0,
                    avg_focus_score REAL DEFAULT 0.0,
                    uptime_hours REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            ''')
            
            # Create performance metrics table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_inspections_unix_timestamp ON inspections(unix_timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_inspections_has_defects ON inspections(has_defects)",
                "CREATE INDEX IF NOT EXISTS idx_inspections_defect_count ON inspections(defect_count)",
                "CREATE INDEX IF NOT EXISTS idx_inspections_session ON inspections(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_defect_stats_type ON defect_statistics(defect_type)",
                "CREATE INDEX IF NOT EXISTS idx_system_stats_date ON system_statistics(date)",
                "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_performance_type ON performance_metrics(metric_type)"
            ]
            
            for index in indexes:
                conn.execute(index)
            
            # Initialize defect statistics for all defect types
            self._initialize_defect_statistics()
            
            conn.commit()
    
    def _initialize_defect_statistics(self):
        """Initialize defect statistics for all defect types."""
        conn = self._get_connection()
        
        for defect_type in DEFECT_CLASSES:
            conn.execute('''
                INSERT OR IGNORE INTO defect_statistics 
                (defect_type, total_count, total_confidence, avg_confidence, first_seen, last_seen)
                VALUES (?, 0, 0.0, 0.0, NULL, NULL)
            ''', (defect_type,))
        
        conn.commit()
    
    def save_inspection(self, timestamp: str, defects: List[str], 
                       locations: List[Dict], **kwargs) -> int:
        """
        Save inspection results to database.
        
        Args:
            timestamp: Inspection timestamp
            defects: List of detected defects
            locations: List of defect locations
            **kwargs: Additional metadata
            
        Returns:
            Inspection ID
        """
        with self.lock:
            conn = self._get_connection()
            
            # Extract metadata
            confidence_scores = kwargs.get('confidence_scores', [])
            focus_score = kwargs.get('focus_score', 0.0)
            processing_time = kwargs.get('processing_time', 0.0)
            raw_image = kwargs.get('raw_image')
            processed_image = kwargs.get('processed_image')
            pcb_area = kwargs.get('pcb_area', 0)
            trigger_type = kwargs.get('trigger_type', 'auto')
            session_id = kwargs.get('session_id', 'default')
            
            # Convert timestamp to datetime
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            
            unix_timestamp = dt.timestamp()
            
            # Determine if we should save image
            has_defects = len(defects) > 0
            defect_count = len(defects)
            
            # Save image if defects found and configured to save
            image_path = None
            image_size_bytes = 0
            
            if has_defects and DB_CONFIG.get("save_processed_images", True):
                if processed_image is not None:
                    image_path = self._save_defect_image(processed_image, dt, defects)
                    if image_path and os.path.exists(image_path):
                        image_size_bytes = os.path.getsize(image_path)
            
            # Insert inspection record
            cursor = conn.execute('''
                INSERT INTO inspections 
                (timestamp, unix_timestamp, has_defects, defect_count, defects, 
                 defect_locations, confidence_scores, focus_score, processing_time, 
                 image_path, image_size_bytes, pcb_area, trigger_type, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                dt.isoformat(),
                unix_timestamp,
                has_defects,
                defect_count,
                json.dumps(defects),
                json.dumps(locations),
                json.dumps(confidence_scores),
                focus_score,
                processing_time,
                image_path,
                image_size_bytes,
                pcb_area,
                trigger_type,
                session_id
            ))
            
            inspection_id = cursor.lastrowid
            
            # Update defect statistics
            if has_defects:
                self._update_defect_statistics(defects, confidence_scores, dt)
            
            # Update system statistics
            self._update_system_statistics(dt, has_defects, defect_count, 
                                         processing_time, focus_score)
            
            conn.commit()
            
            self.logger.debug(f"Saved inspection {inspection_id} with {defect_count} defects")
            
            return inspection_id
    
    def _save_defect_image(self, image: np.ndarray, timestamp: datetime, 
                          defects: List[str]) -> Optional[str]:
        """
        Save defect image to disk.
        
        Args:
            image: Image to save
            timestamp: Timestamp for filename
            defects: List of defects for filename
            
        Returns:
            Path to saved image or None if failed
        """
        try:
            # Generate filename
            defect_names = "_".join(defects[:3])  # Max 3 defects in filename
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{defect_names}.jpg"
            
            # Ensure filename is valid
            filename = "".join(c for c in filename if c.isalnum() or c in '._-')
            
            # Create full path
            image_path = os.path.join(self.storage_config["defects_dir"], filename)
            
            # Resize image if too large
            max_size = self.storage_config.get("max_image_size", 1920)
            if image.shape[0] > max_size or image.shape[1] > max_size:
                scale = max_size / max(image.shape[:2])
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Save image with compression
            quality = self.storage_config.get("image_quality", 85)
            success = cv2.imwrite(image_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            if success:
                return image_path
            else:
                self.logger.warning(f"Failed to save defect image: {image_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving defect image: {str(e)}")
            return None
    
    def _update_defect_statistics(self, defects: List[str], 
                                 confidence_scores: List[float], 
                                 timestamp: datetime):
        """Update defect statistics."""
        conn = self._get_connection()
        
        # Group defects by type
        defect_stats = {}
        for i, defect in enumerate(defects):
            if defect not in defect_stats:
                defect_stats[defect] = {'count': 0, 'total_confidence': 0.0}
            
            defect_stats[defect]['count'] += 1
            if i < len(confidence_scores):
                defect_stats[defect]['total_confidence'] += confidence_scores[i]
        
        # Update statistics for each defect type
        for defect_type, stats in defect_stats.items():
            count = stats['count']
            total_confidence = stats['total_confidence']
            
            # Get current statistics
            current = conn.execute('''
                SELECT total_count, total_confidence, first_seen 
                FROM defect_statistics 
                WHERE defect_type = ?
            ''', (defect_type,)).fetchone()
            
            if current:
                new_total_count = current['total_count'] + count
                new_total_confidence = current['total_confidence'] + total_confidence
                new_avg_confidence = new_total_confidence / new_total_count if new_total_count > 0 else 0.0
                
                first_seen = current['first_seen'] or timestamp.isoformat()
                
                conn.execute('''
                    UPDATE defect_statistics 
                    SET total_count = ?, total_confidence = ?, avg_confidence = ?,
                        first_seen = ?, last_seen = ?, last_updated = ?
                    WHERE defect_type = ?
                ''', (
                    new_total_count,
                    new_total_confidence,
                    new_avg_confidence,
                    first_seen,
                    timestamp.isoformat(),
                    datetime.now().isoformat(),
                    defect_type
                ))
    
    def _update_system_statistics(self, timestamp: datetime, has_defects: bool,
                                 defect_count: int, processing_time: float,
                                 focus_score: float):
        """Update system statistics."""
        conn = self._get_connection()
        
        date_str = timestamp.strftime('%Y-%m-%d')
        
        # Get current daily statistics
        current = conn.execute('''
            SELECT * FROM system_statistics WHERE date = ?
        ''', (date_str,)).fetchone()
        
        if current:
            # Update existing record
            new_total_inspections = current['total_inspections'] + 1
            new_total_defects = current['total_defects'] + defect_count
            new_defect_rate = new_total_defects / new_total_inspections if new_total_inspections > 0 else 0.0
            
            # Calculate new averages
            old_avg_processing = current['avg_processing_time']
            old_avg_focus = current['avg_focus_score']
            
            new_avg_processing = ((old_avg_processing * current['total_inspections'] + processing_time) / 
                                new_total_inspections)
            new_avg_focus = ((old_avg_focus * current['total_inspections'] + focus_score) / 
                           new_total_inspections)
            
            conn.execute('''
                UPDATE system_statistics 
                SET total_inspections = ?, total_defects = ?, defect_rate = ?,
                    avg_processing_time = ?, avg_focus_score = ?
                WHERE date = ?
            ''', (
                new_total_inspections,
                new_total_defects,
                new_defect_rate,
                new_avg_processing,
                new_avg_focus,
                date_str
            ))
        else:
            # Create new record
            defect_rate = defect_count / 1 if defect_count > 0 else 0.0
            
            conn.execute('''
                INSERT INTO system_statistics 
                (date, total_inspections, total_defects, defect_rate,
                 avg_processing_time, avg_focus_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                date_str,
                1,
                defect_count,
                defect_rate,
                processing_time,
                focus_score
            ))
    
    def get_recent_inspections(self, limit: int = 50) -> List[Dict]:
        """
        Get recent inspection records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of inspection records
        """
        conn = self._get_connection()
        
        cursor = conn.execute('''
            SELECT id, timestamp, has_defects, defect_count, defects, 
                   defect_locations, confidence_scores, focus_score, 
                   processing_time, image_path, trigger_type, session_id
            FROM inspections
            ORDER BY unix_timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        inspections = []
        for row in cursor.fetchall():
            inspection = dict(row)
            
            # Parse JSON fields
            if inspection['defects']:
                inspection['defects'] = json.loads(inspection['defects'])
            else:
                inspection['defects'] = []
            
            if inspection['defect_locations']:
                inspection['defect_locations'] = json.loads(inspection['defect_locations'])
            else:
                inspection['defect_locations'] = []
            
            if inspection['confidence_scores']:
                inspection['confidence_scores'] = json.loads(inspection['confidence_scores'])
            else:
                inspection['confidence_scores'] = []
            
            inspections.append(inspection)
        
        return inspections
    
    def get_inspections_by_date_range(self, start_date: datetime, 
                                     end_date: datetime) -> List[Dict]:
        """
        Get inspections within date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of inspection records
        """
        conn = self._get_connection()
        
        cursor = conn.execute('''
            SELECT * FROM inspections
            WHERE unix_timestamp BETWEEN ? AND ?
            ORDER BY unix_timestamp DESC
        ''', (start_date.timestamp(), end_date.timestamp()))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_defect_statistics(self) -> List[Dict]:
        """
        Get defect statistics.
        
        Returns:
            List of defect statistics
        """
        conn = self._get_connection()
        
        cursor = conn.execute('''
            SELECT * FROM defect_statistics
            ORDER BY total_count DESC
        ''')
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_system_statistics(self, days: int = 30) -> List[Dict]:
        """
        Get system statistics for last N days.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            List of daily statistics
        """
        conn = self._get_connection()
        
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor = conn.execute('''
            SELECT * FROM system_statistics
            WHERE date >= ?
            ORDER BY date DESC
        ''', (start_date,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_metrics(self, metric_type: str = None, 
                               hours: int = 24) -> List[Dict]:
        """
        Get performance metrics.
        
        Args:
            metric_type: Type of metric to retrieve
            hours: Number of hours to retrieve
            
        Returns:
            List of performance metrics
        """
        conn = self._get_connection()
        
        start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        if metric_type:
            cursor = conn.execute('''
                SELECT * FROM performance_metrics
                WHERE metric_type = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            ''', (metric_type, start_time))
        else:
            cursor = conn.execute('''
                SELECT * FROM performance_metrics
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (start_time,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def record_performance_metric(self, metric_type: str, value: float,
                                 metadata: Dict = None):
        """
        Record a performance metric.
        
        Args:
            metric_type: Type of metric
            value: Metric value
            metadata: Additional metadata
        """
        conn = self._get_connection()
        
        conn.execute('''
            INSERT INTO performance_metrics 
            (timestamp, metric_type, metric_value, metadata)
            VALUES (?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            metric_type,
            value,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
    
    def cleanup_old_data(self, days: int = 30):
        """
        Clean up old data beyond specified days.
        
        Args:
            days: Number of days to keep
        """
        with self.lock:
            conn = self._get_connection()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            # Get images to delete
            cursor = conn.execute('''
                SELECT image_path FROM inspections
                WHERE unix_timestamp < ? AND image_path IS NOT NULL
            ''', (cutoff_timestamp,))
            
            images_to_delete = [row['image_path'] for row in cursor.fetchall()]
            
            # Delete old inspection records
            conn.execute('''
                DELETE FROM inspections
                WHERE unix_timestamp < ?
            ''', (cutoff_timestamp,))
            
            # Delete old performance metrics
            conn.execute('''
                DELETE FROM performance_metrics
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            # Delete old system statistics (keep monthly summaries)
            monthly_cutoff = datetime.now() - timedelta(days=365)
            conn.execute('''
                DELETE FROM system_statistics
                WHERE date < ?
            ''', (monthly_cutoff.strftime('%Y-%m-%d'),))
            
            conn.commit()
            
            # Delete image files
            deleted_images = 0
            for image_path in images_to_delete:
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        deleted_images += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete image {image_path}: {e}")
            
            self.logger.info(f"Cleaned up {len(images_to_delete)} records, "
                           f"deleted {deleted_images} images")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        conn = self._get_connection()
        
        # Database size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        # Image storage size
        defects_dir = self.storage_config["defects_dir"]
        image_size = 0
        image_count = 0
        
        if os.path.exists(defects_dir):
            for file in os.listdir(defects_dir):
                file_path = os.path.join(defects_dir, file)
                if os.path.isfile(file_path):
                    image_size += os.path.getsize(file_path)
                    image_count += 1
        
        # Record counts
        cursor = conn.execute("SELECT COUNT(*) as count FROM inspections")
        inspection_count = cursor.fetchone()['count']
        
        cursor = conn.execute("SELECT COUNT(*) as count FROM inspections WHERE has_defects = 1")
        defect_inspection_count = cursor.fetchone()['count']
        
        return {
            'database_size_bytes': db_size,
            'image_storage_size_bytes': image_size,
            'image_count': image_count,
            'total_inspections': inspection_count,
            'defect_inspections': defect_inspection_count,
            'storage_directories': {
                'database': self.db_path,
                'defects': defects_dir,
                'images': self.storage_config["images_dir"]
            }
        }
    
    def close(self):
        """Close database connection."""
        if hasattr(self.local, 'connection'):
            self.local.connection.close()
            del self.local.connection
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Utility functions
def create_test_database(db_path: str = "test_pcb.db") -> PCBDatabase:
    """
    Create a test database for testing.
    
    Args:
        db_path: Path to test database
        
    Returns:
        PCBDatabase instance
    """
    # Remove existing test database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Create new database
    db = PCBDatabase(db_path)
    
    return db


def populate_test_data(db: PCBDatabase, num_inspections: int = 100):
    """
    Populate database with test data.
    
    Args:
        db: Database instance
        num_inspections: Number of test inspections to create
    """
    import random
    
    defect_types = DEFECT_CLASSES
    
    for i in range(num_inspections):
        # Generate random timestamp
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 30),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        # Generate random defects
        num_defects = random.choices([0, 1, 2, 3], weights=[0.7, 0.2, 0.08, 0.02])[0]
        
        defects = []
        locations = []
        confidence_scores = []
        
        for j in range(num_defects):
            defect = random.choice(defect_types)
            confidence = random.uniform(0.5, 0.95)
            
            defects.append(defect)
            confidence_scores.append(confidence)
            
            # Random location
            x1, y1 = random.randint(0, 500), random.randint(0, 500)
            x2, y2 = x1 + random.randint(50, 150), y1 + random.randint(50, 150)
            
            locations.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_name': defect
            })
        
        # Save inspection
        db.save_inspection(
            timestamp.isoformat(),
            defects,
            locations,
            confidence_scores=confidence_scores,
            focus_score=random.uniform(80, 200),
            processing_time=random.uniform(0.02, 0.15),
            trigger_type=random.choice(['auto', 'manual']),
            session_id=f"session_{i // 20}"
        )
    
    print(f"Created {num_inspections} test inspections")


if __name__ == "__main__":
    # Test database functionality
    db = create_test_database()
    
    # Test basic operations
    print("Testing basic operations...")
    
    # Test saving inspection
    inspection_id = db.save_inspection(
        datetime.now().isoformat(),
        ["Missing Hole", "Open Circuit"],
        [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.85, 'class_name': 'Missing Hole'},
            {'bbox': [300, 300, 400, 400], 'confidence': 0.75, 'class_name': 'Open Circuit'}
        ],
        confidence_scores=[0.85, 0.75],
        focus_score=150.0,
        processing_time=0.05
    )
    
    print(f"Saved inspection with ID: {inspection_id}")
    
    # Test retrieval
    recent = db.get_recent_inspections(10)
    print(f"Retrieved {len(recent)} recent inspections")
    
    # Test statistics
    stats = db.get_defect_statistics()
    print(f"Defect statistics: {len(stats)} defect types")
    
    # Test storage stats
    storage = db.get_storage_stats()
    print(f"Storage stats: {storage}")
    
    print("Database test completed successfully!")