"""
Unit tests for Data layer components.

Tests database operations, analytics, and data management functionality.
"""

import unittest
import tempfile
import os
import sys
import sqlite3
import json
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import DB_CONFIG, DEFECT_CLASSES


class TestPCBDatabase(unittest.TestCase):
    """Test PCBDatabase functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        try:
            from data.database import PCBDatabase
            self.database = PCBDatabase(self.db_path)
        except ImportError:
            self.skipTest("Data module not available")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'database'):
            self.database.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_database_initialization(self):
        """Test database initialization and table creation."""
        self.assertIsNotNone(self.database)
        
        # Check that database file was created
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check that tables were created
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check inspections table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='inspections'")
        self.assertIsNotNone(cursor.fetchone())
        
        # Check defect_statistics table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='defect_statistics'")
        self.assertIsNotNone(cursor.fetchone())
        
        conn.close()
    
    def test_save_inspection_metadata_no_defects(self):
        """Test saving inspection with no defects."""
        timestamp = datetime.now()
        defects = []
        locations = []
        confidence_scores = []
        
        inspection_id = self.database.save_inspection_metadata(
            timestamp=timestamp,
            defects=defects,
            locations=locations,
            confidence_scores=confidence_scores,
            raw_image_shape=(480, 640),
            focus_score=150.0,
            processing_time=0.1
        )
        
        self.assertIsInstance(inspection_id, int)
        self.assertGreater(inspection_id, 0)
        
        # Verify data was saved correctly
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[3], False)  # has_defects
        self.assertEqual(row[4], 0)      # defect_count
    
    def test_save_inspection_metadata_with_defects(self):
        """Test saving inspection with defects."""
        timestamp = datetime.now()
        defects = ["Missing Hole", "Spur"]
        locations = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9},
            {"bbox": [300, 150, 400, 250], "confidence": 0.8}
        ]
        confidence_scores = [0.9, 0.8]
        
        inspection_id = self.database.save_inspection_metadata(
            timestamp=timestamp,
            defects=defects,
            locations=locations,
            confidence_scores=confidence_scores,
            raw_image_shape=(480, 640),
            focus_score=120.0,
            processing_time=0.15
        )
        
        self.assertIsInstance(inspection_id, int)
        self.assertGreater(inspection_id, 0)
        
        # Verify data was saved correctly
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,))
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[3], True)   # has_defects
        self.assertEqual(row[4], 2)      # defect_count
        
        # Check defects were stored as JSON
        stored_defects = json.loads(row[5])
        self.assertEqual(stored_defects, defects)
    
    def test_defect_statistics_update(self):
        """Test defect statistics tracking."""
        # Save inspection with defects
        timestamp = datetime.now()
        defects = ["Missing Hole", "Missing Hole", "Spur"]  # Duplicate defect
        
        self.database.save_inspection_metadata(
            timestamp=timestamp,
            defects=defects,
            locations=[{}, {}, {}],
            confidence_scores=[0.9, 0.8, 0.7],
            raw_image_shape=(480, 640),
            focus_score=130.0
        )
        
        # Check defect statistics
        stats = self.database.get_defect_statistics()
        
        # Should have entries for both defect types
        defect_types = [stat['defect_type'] for stat in stats]
        self.assertIn("Missing Hole", defect_types)
        self.assertIn("Spur", defect_types)
        
        # Check counts
        missing_hole_stat = next(s for s in stats if s['defect_type'] == "Missing Hole")
        self.assertEqual(missing_hole_stat['total_count'], 2)
        
        spur_stat = next(s for s in stats if s['defect_type'] == "Spur")
        self.assertEqual(spur_stat['total_count'], 1)
    
    def test_get_recent_inspections(self):
        """Test retrieving recent inspections."""
        # Save multiple inspections
        base_time = datetime.now()
        
        for i in range(5):
            timestamp = base_time + timedelta(minutes=i)
            defects = ["Test Defect"] if i % 2 == 0 else []
            
            self.database.save_inspection_metadata(
                timestamp=timestamp,
                defects=defects,
                locations=[{}] if defects else [],
                confidence_scores=[0.8] if defects else [],
                raw_image_shape=(480, 640),
                focus_score=100.0 + i
            )
        
        # Get recent inspections
        recent = self.database.get_recent_inspections(limit=3)
        
        self.assertEqual(len(recent), 3)
        
        # Should be ordered by timestamp (most recent first)
        timestamps = [r['timestamp'] for r in recent]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))
    
    def test_thread_safety(self):
        """Test thread-safe database operations."""
        import threading
        
        results = []
        errors = []
        
        def save_inspection(thread_id):
            try:
                for i in range(10):
                    timestamp = datetime.now()
                    defects = [f"Thread{thread_id}_Defect{i}"]
                    
                    inspection_id = self.database.save_inspection_metadata(
                        timestamp=timestamp,
                        defects=defects,
                        locations=[{}],
                        confidence_scores=[0.8],
                        raw_image_shape=(480, 640),
                        focus_score=100.0
                    )
                    results.append(inspection_id)
                    time.sleep(0.001)  # Small delay to test concurrency
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=save_inspection, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have no errors and all inspections saved
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 30)  # 3 threads * 10 inspections
        
        # All inspection IDs should be unique
        self.assertEqual(len(set(results)), len(results))
    
    def test_performance_bulk_insert(self):
        """Test database performance with bulk inserts."""
        import time
        
        start_time = time.time()
        
        # Insert many inspections
        for i in range(100):
            timestamp = datetime.now()
            defects = ["Defect"] if i % 5 == 0 else []
            
            self.database.save_inspection_metadata(
                timestamp=timestamp,
                defects=defects,
                locations=[{}] if defects else [],
                confidence_scores=[0.8] if defects else [],
                raw_image_shape=(480, 640),
                focus_score=100.0
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should be reasonably fast (< 1 second for 100 inserts)
        self.assertLess(total_time, 1.0)
        
        # Verify all were inserted
        recent = self.database.get_recent_inspections(limit=100)
        self.assertEqual(len(recent), 100)


class TestDefectAnalyzer(unittest.TestCase):
    """Test DefectAnalyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        try:
            from data.database import PCBDatabase
            from analytics.analyzer import DefectAnalyzer
            
            self.database = PCBDatabase(self.db_path)
            self.analyzer = DefectAnalyzer(self.database)
            
            # Populate with test data
            self._populate_test_data()
            
        except ImportError:
            self.skipTest("Data or analytics modules not available")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'database'):
            self.database.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def _populate_test_data(self):
        """Populate database with test data."""
        base_time = datetime.now() - timedelta(days=7)
        
        # Add various inspections over past week
        test_cases = [
            (0, ["Missing Hole"]),
            (1, []),
            (2, ["Spur", "Mouse Bite"]),
            (3, []),
            (4, ["Missing Hole"]),
            (5, ["Short Circuit"]),
            (6, []),
        ]
        
        for day_offset, defects in test_cases:
            timestamp = base_time + timedelta(days=day_offset)
            
            self.database.save_inspection_metadata(
                timestamp=timestamp,
                defects=defects,
                locations=[{} for _ in defects],
                confidence_scores=[0.8 for _ in defects],
                raw_image_shape=(480, 640),
                focus_score=120.0
            )
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.database, self.database)
    
    def test_get_realtime_stats(self):
        """Test real-time statistics calculation."""
        stats = self.analyzer.get_realtime_stats()
        
        self.assertIsInstance(stats, dict)
        
        # Check required keys
        required_keys = ['total_inspections', 'total_defects', 'pass_rate', 'defect_rate']
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Check values make sense
        self.assertGreaterEqual(stats['total_inspections'], 0)
        self.assertGreaterEqual(stats['total_defects'], 0)
        self.assertGreaterEqual(stats['pass_rate'], 0)
        self.assertLessEqual(stats['pass_rate'], 100)
    
    def test_get_defect_frequency(self):
        """Test defect frequency analysis."""
        frequency = self.analyzer.get_defect_frequency()
        
        self.assertIsInstance(frequency, dict)
        
        # Should have entries for defects in test data
        self.assertIn("Missing Hole", frequency)
        self.assertIn("Spur", frequency)
        self.assertIn("Mouse Bite", frequency)
        self.assertIn("Short Circuit", frequency)
        
        # Check that Missing Hole appears twice
        self.assertEqual(frequency["Missing Hole"], 2)
    
    def test_get_time_based_analysis(self):
        """Test time-based analysis."""
        try:
            analysis = self.analyzer.get_time_based_analysis(days=7)
            
            self.assertIsInstance(analysis, dict)
            
            # Check for time-based keys
            if 'daily_counts' in analysis:
                self.assertIsInstance(analysis['daily_counts'], list)
            
        except AttributeError:
            # Method might not be implemented yet
            self.skipTest("Time-based analysis not implemented")
    
    def test_get_comprehensive_report(self):
        """Test comprehensive report generation."""
        report = self.analyzer.get_comprehensive_report()
        
        self.assertIsInstance(report, dict)
        
        # Should include basic statistics
        self.assertIn('total_inspections', report)
        self.assertIn('total_defects', report)
        
        # Check data consistency
        total_inspections = report['total_inspections']
        self.assertGreater(total_inspections, 0)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        try:
            metrics = self.analyzer.get_performance_metrics()
            
            self.assertIsInstance(metrics, dict)
            
            # Should include various performance indicators
            possible_keys = ['avg_processing_time', 'inspection_rate', 'system_uptime']
            
            # At least some performance metrics should be present
            has_metrics = any(key in metrics for key in possible_keys)
            self.assertTrue(has_metrics or len(metrics) > 0)
            
        except AttributeError:
            # Method might not be implemented yet
            self.skipTest("Performance metrics not implemented")
    
    def test_caching_behavior(self):
        """Test analytics caching behavior."""
        import time
        
        # Get stats twice in quick succession
        start_time = time.time()
        stats1 = self.analyzer.get_realtime_stats()
        mid_time = time.time()
        stats2 = self.analyzer.get_realtime_stats()
        end_time = time.time()
        
        # Second call should be faster if cached
        first_call_time = mid_time - start_time
        second_call_time = end_time - mid_time
        
        # Results should be identical if cached
        if hasattr(self.analyzer, '_cache_timeout'):
            self.assertEqual(stats1, stats2)


class TestDataIntegration(unittest.TestCase):
    """Test integration between data components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        try:
            from data.database import PCBDatabase
            from analytics.analyzer import DefectAnalyzer
            
            self.database = PCBDatabase(self.db_path)
            self.analyzer = DefectAnalyzer(self.database)
            
        except ImportError:
            self.skipTest("Data modules not available")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'database'):
            self.database.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_database_to_analytics_pipeline(self):
        """Test pipeline from database operations to analytics."""
        # Save some inspection data
        timestamp = datetime.now()
        defects = ["Missing Hole", "Spur"]
        
        inspection_id = self.database.save_inspection_metadata(
            timestamp=timestamp,
            defects=defects,
            locations=[{}, {}],
            confidence_scores=[0.9, 0.8],
            raw_image_shape=(480, 640),
            focus_score=140.0
        )
        
        # Analytics should reflect the new data
        stats = self.analyzer.get_realtime_stats()
        
        self.assertEqual(stats['total_inspections'], 1)
        self.assertEqual(stats['total_defects'], 2)
        self.assertFalse(stats['pass_rate'] == 100)  # Should not be 100% with defects
    
    def test_concurrent_database_analytics(self):
        """Test concurrent database operations and analytics."""
        import threading
        
        results = []
        errors = []
        
        def add_inspections():
            try:
                for i in range(10):
                    timestamp = datetime.now()
                    defects = ["Test Defect"] if i % 3 == 0 else []
                    
                    self.database.save_inspection_metadata(
                        timestamp=timestamp,
                        defects=defects,
                        locations=[{}] if defects else [],
                        confidence_scores=[0.8] if defects else [],
                        raw_image_shape=(480, 640),
                        focus_score=100.0
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def read_analytics():
            try:
                for i in range(10):
                    stats = self.analyzer.get_realtime_stats()
                    results.append(stats)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Start concurrent operations
        write_thread = threading.Thread(target=add_inspections)
        read_thread = threading.Thread(target=read_analytics)
        
        write_thread.start()
        read_thread.start()
        
        write_thread.join()
        read_thread.join()
        
        # Should complete without errors
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)


class TestDataErrorHandling(unittest.TestCase):
    """Test error handling in data operations."""
    
    def test_invalid_database_path(self):
        """Test handling of invalid database paths."""
        try:
            from data.database import PCBDatabase
            
            # Try to create database in non-existent directory
            invalid_path = "/non/existent/path/test.db"
            
            with self.assertRaises((OSError, sqlite3.OperationalError)):
                PCBDatabase(invalid_path)
                
        except ImportError:
            self.skipTest("Data module not available")
    
    def test_corrupted_database_recovery(self):
        """Test recovery from corrupted database."""
        try:
            from data.database import PCBDatabase
            
            # Create a corrupted database file
            temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            temp_db.write(b"This is not a valid SQLite database")
            temp_db.close()
            
            try:
                # Should handle corrupted database gracefully
                with self.assertRaises(sqlite3.DatabaseError):
                    PCBDatabase(temp_db.name)
            finally:
                os.unlink(temp_db.name)
                
        except ImportError:
            self.skipTest("Data module not available")
    
    def test_invalid_inspection_data(self):
        """Test handling of invalid inspection data."""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        try:
            from data.database import PCBDatabase
            
            database = PCBDatabase(temp_db.name)
            
            # Test with invalid data types
            with self.assertRaises((TypeError, ValueError)):
                database.save_inspection_metadata(
                    timestamp="invalid_timestamp",  # Should be datetime
                    defects=None,  # Should be list
                    locations=None,  # Should be list
                    confidence_scores=None,  # Should be list
                    raw_image_shape=(480, 640),
                    focus_score=120.0
                )
            
            database.close()
            
        except ImportError:
            self.skipTest("Data module not available")
        finally:
            if os.path.exists(temp_db.name):
                os.unlink(temp_db.name)


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    unittest.main(verbosity=2)