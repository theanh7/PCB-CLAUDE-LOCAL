"""
Comprehensive test suite for data layer components.

This module provides testing for database operations, analytics,
concurrent access, and performance of the data management system.
"""

import unittest
import os
import tempfile
import threading
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import PCBDatabase, populate_test_data
from analytics.analyzer import DefectAnalyzer
from core.config import DEFECT_CLASSES
from core.interfaces import InspectionResult


class TestPCBDatabase(unittest.TestCase):
    """Test cases for PCBDatabase class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db = PCBDatabase(self.temp_db.name)
        
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.db.close()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization."""
        # Check if database file exists
        self.assertTrue(os.path.exists(self.temp_db.name))
        
        # Check if tables exist
        conn = self.db._get_connection()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [table[0] for table in tables]
        
        expected_tables = ['inspections', 'defect_statistics', 'system_statistics', 'performance_metrics']
        for table in expected_tables:
            self.assertIn(table, table_names)
    
    def test_save_inspection(self):
        """Test saving inspection results."""
        # Test data
        timestamp = datetime.now().isoformat()
        defects = ["Missing Hole", "Open Circuit"]
        locations = [
            {'bbox': [100, 100, 200, 200], 'confidence': 0.85, 'class_name': 'Missing Hole'},
            {'bbox': [300, 300, 400, 400], 'confidence': 0.75, 'class_name': 'Open Circuit'}
        ]
        confidence_scores = [0.85, 0.75]
        
        # Save inspection
        inspection_id = self.db.save_inspection(
            timestamp=timestamp,
            defects=defects,
            locations=locations,
            confidence_scores=confidence_scores,
            focus_score=150.0,
            processing_time=0.05
        )
        
        # Verify inspection was saved
        self.assertIsInstance(inspection_id, int)
        self.assertGreater(inspection_id, 0)
        
        # Verify data in database
        inspections = self.db.get_recent_inspections(1)
        self.assertEqual(len(inspections), 1)
        
        inspection = inspections[0]
        self.assertEqual(inspection['defects'], defects)
        self.assertEqual(inspection['defect_count'], 2)
        self.assertTrue(inspection['has_defects'])
        self.assertEqual(inspection['focus_score'], 150.0)
        self.assertEqual(inspection['processing_time'], 0.05)
    
    def test_save_inspection_no_defects(self):
        """Test saving inspection with no defects."""
        timestamp = datetime.now().isoformat()
        
        inspection_id = self.db.save_inspection(
            timestamp=timestamp,
            defects=[],
            locations=[],
            confidence_scores=[],
            focus_score=180.0,
            processing_time=0.03
        )
        
        # Verify inspection was saved
        self.assertIsInstance(inspection_id, int)
        
        # Verify data
        inspections = self.db.get_recent_inspections(1)
        inspection = inspections[0]
        
        self.assertEqual(inspection['defects'], [])
        self.assertEqual(inspection['defect_count'], 0)
        self.assertFalse(inspection['has_defects'])
        self.assertEqual(inspection['focus_score'], 180.0)
    
    def test_get_recent_inspections(self):
        """Test retrieving recent inspections."""
        # Add test data
        for i in range(10):
            self.db.save_inspection(
                timestamp=datetime.now().isoformat(),
                defects=["Missing Hole"] if i % 2 == 0 else [],
                locations=[{'bbox': [i*10, i*10, i*10+50, i*10+50], 'confidence': 0.8}] if i % 2 == 0 else [],
                confidence_scores=[0.8] if i % 2 == 0 else [],
                focus_score=150.0,
                processing_time=0.05
            )
        
        # Test retrieval
        inspections = self.db.get_recent_inspections(5)
        self.assertEqual(len(inspections), 5)
        
        # Check that results are ordered by timestamp (newest first)
        timestamps = [insp['timestamp'] for insp in inspections]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))
    
    def test_get_inspections_by_date_range(self):
        """Test retrieving inspections by date range."""
        # Add inspections with different timestamps
        base_time = datetime.now()
        
        for i in range(5):
            timestamp = base_time - timedelta(days=i)
            self.db.save_inspection(
                timestamp=timestamp.isoformat(),
                defects=["Missing Hole"],
                locations=[{'bbox': [100, 100, 200, 200], 'confidence': 0.8}],
                confidence_scores=[0.8],
                focus_score=150.0,
                processing_time=0.05
            )
        
        # Test date range query
        start_date = base_time - timedelta(days=2)
        end_date = base_time + timedelta(days=1)
        
        inspections = self.db.get_inspections_by_date_range(start_date, end_date)
        self.assertEqual(len(inspections), 3)  # Days 0, 1, 2
    
    def test_defect_statistics(self):
        """Test defect statistics functionality."""
        # Add inspections with different defects
        defects_data = [
            ["Missing Hole", "Open Circuit"],
            ["Missing Hole"],
            ["Spur", "Mouse Bite"],
            ["Missing Hole"]
        ]
        
        for defects in defects_data:
            confidence_scores = [0.8] * len(defects)
            locations = [{'bbox': [100, 100, 200, 200], 'confidence': 0.8}] * len(defects)
            
            self.db.save_inspection(
                timestamp=datetime.now().isoformat(),
                defects=defects,
                locations=locations,
                confidence_scores=confidence_scores,
                focus_score=150.0,
                processing_time=0.05
            )
        
        # Test defect statistics
        stats = self.db.get_defect_statistics()
        
        # Check that all defect types are present
        defect_types = [stat['defect_type'] for stat in stats]
        for defect_type in DEFECT_CLASSES:
            self.assertIn(defect_type, defect_types)
        
        # Check specific counts
        missing_hole_stat = next(stat for stat in stats if stat['defect_type'] == 'Missing Hole')
        self.assertEqual(missing_hole_stat['total_count'], 3)
        
        open_circuit_stat = next(stat for stat in stats if stat['defect_type'] == 'Open Circuit')
        self.assertEqual(open_circuit_stat['total_count'], 1)
    
    def test_system_statistics(self):
        """Test system statistics functionality."""
        # Add test data
        for i in range(5):
            self.db.save_inspection(
                timestamp=datetime.now().isoformat(),
                defects=["Missing Hole"] if i % 2 == 0 else [],
                locations=[{'bbox': [100, 100, 200, 200], 'confidence': 0.8}] if i % 2 == 0 else [],
                confidence_scores=[0.8] if i % 2 == 0 else [],
                focus_score=150.0,
                processing_time=0.05
            )
        
        # Test system statistics
        stats = self.db.get_system_statistics(7)
        
        # Should have at least one day of statistics
        self.assertGreater(len(stats), 0)
        
        # Check structure
        stat = stats[0]
        required_fields = ['date', 'total_inspections', 'total_defects', 'defect_rate']
        for field in required_fields:
            self.assertIn(field, stat)
    
    def test_performance_metrics(self):
        """Test performance metrics functionality."""
        # Add some performance metrics
        self.db.record_performance_metric('inference_time', 0.05, {'gpu': 'Tesla P4'})
        self.db.record_performance_metric('preprocessing_time', 0.02, {'method': 'bayer'})
        
        # Test retrieval
        metrics = self.db.get_performance_metrics('inference_time', 1)
        self.assertEqual(len(metrics), 1)
        
        metric = metrics[0]
        self.assertEqual(metric['metric_type'], 'inference_time')
        self.assertEqual(metric['metric_value'], 0.05)
        
        # Test all metrics
        all_metrics = self.db.get_performance_metrics(hours=1)
        self.assertEqual(len(all_metrics), 2)
    
    def test_storage_stats(self):
        """Test storage statistics."""
        # Add some data
        for i in range(3):
            self.db.save_inspection(
                timestamp=datetime.now().isoformat(),
                defects=["Missing Hole"],
                locations=[{'bbox': [100, 100, 200, 200], 'confidence': 0.8}],
                confidence_scores=[0.8],
                focus_score=150.0,
                processing_time=0.05
            )
        
        # Test storage stats
        stats = self.db.get_storage_stats()
        
        required_fields = ['database_size_bytes', 'total_inspections', 'defect_inspections']
        for field in required_fields:
            self.assertIn(field, stats)
        
        self.assertEqual(stats['total_inspections'], 3)
        self.assertEqual(stats['defect_inspections'], 3)
        self.assertGreater(stats['database_size_bytes'], 0)
    
    def test_cleanup_old_data(self):
        """Test data cleanup functionality."""
        # Add old data
        old_time = datetime.now() - timedelta(days=35)
        
        for i in range(3):
            self.db.save_inspection(
                timestamp=old_time.isoformat(),
                defects=["Missing Hole"],
                locations=[{'bbox': [100, 100, 200, 200], 'confidence': 0.8}],
                confidence_scores=[0.8],
                focus_score=150.0,
                processing_time=0.05
            )
        
        # Add recent data
        for i in range(2):
            self.db.save_inspection(
                timestamp=datetime.now().isoformat(),
                defects=["Missing Hole"],
                locations=[{'bbox': [100, 100, 200, 200], 'confidence': 0.8}],
                confidence_scores=[0.8],
                focus_score=150.0,
                processing_time=0.05
            )
        
        # Verify initial count
        inspections = self.db.get_recent_inspections(10)
        self.assertEqual(len(inspections), 5)
        
        # Clean up old data
        self.db.cleanup_old_data(30)
        
        # Verify cleanup
        inspections = self.db.get_recent_inspections(10)
        self.assertEqual(len(inspections), 2)


class TestDefectAnalyzer(unittest.TestCase):
    """Test cases for DefectAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db = PCBDatabase(self.temp_db.name)
        self.analyzer = DefectAnalyzer(self.db)
        
        # Add test data
        self._populate_test_data()
        
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.db.close()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def _populate_test_data(self):
        """Populate database with test data."""
        import random
        
        # Add varied test data
        for i in range(50):
            timestamp = datetime.now() - timedelta(days=random.randint(0, 7))
            
            # Random defects
            num_defects = random.choices([0, 1, 2], weights=[0.7, 0.25, 0.05])[0]
            
            defects = []
            locations = []
            confidence_scores = []
            
            for j in range(num_defects):
                defect = random.choice(DEFECT_CLASSES)
                confidence = random.uniform(0.6, 0.95)
                
                defects.append(defect)
                confidence_scores.append(confidence)
                locations.append({
                    'bbox': [100+j*50, 100+j*50, 150+j*50, 150+j*50],
                    'confidence': confidence,
                    'class_name': defect
                })
            
            self.db.save_inspection(
                timestamp=timestamp.isoformat(),
                defects=defects,
                locations=locations,
                confidence_scores=confidence_scores,
                focus_score=random.uniform(100, 200),
                processing_time=random.uniform(0.02, 0.1)
            )
    
    def test_realtime_analysis(self):
        """Test real-time analysis."""
        analysis = self.analyzer.get_realtime_analysis()
        
        # Check structure
        required_keys = ['timestamp', 'inspection_metrics', 'defect_metrics', 
                        'quality_metrics', 'performance_metrics', 'system_health']
        for key in required_keys:
            self.assertIn(key, analysis)
        
        # Check inspection metrics
        inspection_metrics = analysis['inspection_metrics']
        self.assertIn('total_inspections', inspection_metrics)
        self.assertIn('defect_rate', inspection_metrics)
        self.assertIn('avg_processing_time', inspection_metrics)
        
        # Check defect metrics
        defect_metrics = analysis['defect_metrics']
        self.assertIsInstance(defect_metrics, list)
        
        # Check system health
        system_health = analysis['system_health']
        self.assertIn('status', system_health)
        self.assertIn('score', system_health)
        self.assertIn('issues', system_health)
    
    def test_time_period_analysis(self):
        """Test time period analysis."""
        analysis = self.analyzer.get_time_period_analysis('7d')
        
        # Check structure
        required_keys = ['period', 'start_date', 'end_date', 'daily_trends', 
                        'defect_trends', 'performance_trends']
        for key in required_keys:
            self.assertIn(key, analysis)
        
        # Check daily trends
        daily_trends = analysis['daily_trends']
        self.assertIsInstance(daily_trends, list)
        
        if daily_trends:
            trend = daily_trends[0]
            self.assertIn('date', trend)
            self.assertIn('total_inspections', trend)
            self.assertIn('defect_rate', trend)
        
        # Check defect trends
        defect_trends = analysis['defect_trends']
        self.assertIsInstance(defect_trends, dict)
        
        # Should have trends for all defect types
        for defect_type in DEFECT_CLASSES:
            self.assertIn(defect_type, defect_trends)
    
    def test_report_generation(self):
        """Test report generation."""
        # Test JSON report
        json_report = self.analyzer.generate_report('7d', 'json')
        self.assertIsInstance(json_report, dict)
        
        # Test HTML report
        html_report = self.analyzer.generate_report('7d', 'html')
        self.assertIsInstance(html_report, str)
        self.assertIn('<html>', html_report)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            self.analyzer.generate_report('7d', 'invalid')
    
    def test_cache_functionality(self):
        """Test caching functionality."""
        # First call should not be cached
        start_time = time.time()
        analysis1 = self.analyzer.get_realtime_analysis()
        first_call_time = time.time() - start_time
        
        # Second call should be cached (faster)
        start_time = time.time()
        analysis2 = self.analyzer.get_realtime_analysis()
        second_call_time = time.time() - start_time
        
        # Should be the same result
        self.assertEqual(analysis1['timestamp'], analysis2['timestamp'])
        
        # Second call should be faster (cached)
        self.assertLess(second_call_time, first_call_time)
        
        # Clear cache
        self.analyzer.clear_cache()
        
        # Next call should be slower again
        start_time = time.time()
        analysis3 = self.analyzer.get_realtime_analysis()
        third_call_time = time.time() - start_time
        
        self.assertGreater(third_call_time, second_call_time)


class TestConcurrentAccess(unittest.TestCase):
    """Test concurrent access to database."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db = PCBDatabase(self.temp_db.name)
        
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.db.close()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_concurrent_writes(self):
        """Test concurrent write operations."""
        results = []
        errors = []
        
        def write_worker(worker_id):
            try:
                for i in range(10):
                    inspection_id = self.db.save_inspection(
                        timestamp=datetime.now().isoformat(),
                        defects=["Missing Hole"],
                        locations=[{'bbox': [100, 100, 200, 200], 'confidence': 0.8}],
                        confidence_scores=[0.8],
                        focus_score=150.0,
                        processing_time=0.05
                    )
                    results.append(f"Worker {worker_id}: {inspection_id}")
                    time.sleep(0.001)  # Small delay to simulate processing
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")
        
        # Start multiple worker threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 50)  # 5 workers × 10 inspections each
        
        # Verify data integrity
        inspections = self.db.get_recent_inspections(100)
        self.assertEqual(len(inspections), 50)
    
    def test_concurrent_read_write(self):
        """Test concurrent read and write operations."""
        write_results = []
        read_results = []
        errors = []
        
        def write_worker():
            try:
                for i in range(20):
                    inspection_id = self.db.save_inspection(
                        timestamp=datetime.now().isoformat(),
                        defects=["Missing Hole"],
                        locations=[{'bbox': [100, 100, 200, 200], 'confidence': 0.8}],
                        confidence_scores=[0.8],
                        focus_score=150.0,
                        processing_time=0.05
                    )
                    write_results.append(inspection_id)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Write worker: {str(e)}")
        
        def read_worker():
            try:
                for i in range(20):
                    inspections = self.db.get_recent_inspections(10)
                    read_results.append(len(inspections))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(f"Read worker: {str(e)}")
        
        # Start writer and readers
        write_thread = threading.Thread(target=write_worker)
        read_threads = [threading.Thread(target=read_worker) for _ in range(3)]
        
        write_thread.start()
        for thread in read_threads:
            thread.start()
        
        # Wait for completion
        write_thread.join()
        for thread in read_threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(write_results), 20)
        self.assertEqual(len(read_results), 60)  # 3 readers × 20 operations each


class TestPerformance(unittest.TestCase):
    """Performance tests for data layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db = PCBDatabase(self.temp_db.name)
        
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.db.close()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_write_performance(self):
        """Test write performance."""
        num_inspections = 1000
        
        start_time = time.time()
        
        for i in range(num_inspections):
            self.db.save_inspection(
                timestamp=datetime.now().isoformat(),
                defects=["Missing Hole"] if i % 2 == 0 else [],
                locations=[{'bbox': [100, 100, 200, 200], 'confidence': 0.8}] if i % 2 == 0 else [],
                confidence_scores=[0.8] if i % 2 == 0 else [],
                focus_score=150.0,
                processing_time=0.05
            )
        
        elapsed_time = time.time() - start_time
        
        # Performance expectations
        inspections_per_second = num_inspections / elapsed_time
        
        print(f"Write performance: {inspections_per_second:.2f} inspections/sec")
        print(f"Average time per inspection: {elapsed_time/num_inspections*1000:.2f}ms")
        
        # Should be able to handle at least 100 inspections per second
        self.assertGreater(inspections_per_second, 100)
    
    def test_read_performance(self):
        """Test read performance."""
        # Add test data
        for i in range(1000):
            self.db.save_inspection(
                timestamp=datetime.now().isoformat(),
                defects=["Missing Hole"] if i % 2 == 0 else [],
                locations=[{'bbox': [100, 100, 200, 200], 'confidence': 0.8}] if i % 2 == 0 else [],
                confidence_scores=[0.8] if i % 2 == 0 else [],
                focus_score=150.0,
                processing_time=0.05
            )
        
        # Test read performance
        start_time = time.time()
        
        for i in range(100):
            inspections = self.db.get_recent_inspections(50)
            self.assertEqual(len(inspections), 50)
        
        elapsed_time = time.time() - start_time
        
        reads_per_second = 100 / elapsed_time
        
        print(f"Read performance: {reads_per_second:.2f} reads/sec")
        print(f"Average time per read: {elapsed_time/100*1000:.2f}ms")
        
        # Should be able to handle at least 50 reads per second
        self.assertGreater(reads_per_second, 50)
    
    def test_analytics_performance(self):
        """Test analytics performance."""
        # Add test data
        populate_test_data(self.db, 500)
        
        analyzer = DefectAnalyzer(self.db)
        
        # Test realtime analysis performance
        start_time = time.time()
        
        for i in range(10):
            analysis = analyzer.get_realtime_analysis()
            self.assertIn('inspection_metrics', analysis)
        
        elapsed_time = time.time() - start_time
        
        analyses_per_second = 10 / elapsed_time
        
        print(f"Analytics performance: {analyses_per_second:.2f} analyses/sec")
        print(f"Average time per analysis: {elapsed_time/10*1000:.2f}ms")
        
        # Should be able to handle at least 2 analyses per second
        self.assertGreater(analyses_per_second, 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for data layer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.db = PCBDatabase(self.temp_db.name)
        self.analyzer = DefectAnalyzer(self.db)
        
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.db.close()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_ai_integration(self):
        """Test integration with AI results."""
        # Simulate AI result
        from core.interfaces import InspectionResult
        
        ai_result = InspectionResult(
            defects=["Missing Hole", "Open Circuit"],
            locations=[
                {'bbox': [100, 100, 200, 200], 'confidence': 0.85, 'class_name': 'Missing Hole'},
                {'bbox': [300, 300, 400, 400], 'confidence': 0.75, 'class_name': 'Open Circuit'}
            ],
            confidence_scores=[0.85, 0.75],
            processing_time=0.05
        )
        
        # Save AI result to database
        inspection_id = self.db.save_inspection(
            timestamp=datetime.now().isoformat(),
            defects=ai_result.defects,
            locations=ai_result.locations,
            confidence_scores=ai_result.confidence_scores,
            processing_time=ai_result.processing_time,
            focus_score=150.0
        )
        
        # Verify integration
        self.assertIsInstance(inspection_id, int)
        
        # Test analytics on AI data
        analysis = self.analyzer.get_realtime_analysis()
        self.assertIn('inspection_metrics', analysis)
        
        inspection_metrics = analysis['inspection_metrics']
        self.assertEqual(inspection_metrics['total_inspections'], 1)
        self.assertEqual(inspection_metrics['defect_rate'], 1.0)
    
    def test_processing_integration(self):
        """Test integration with processing results."""
        # Simulate processing pipeline results
        from processing.pcb_detector import PCBDetector
        
        # Mock PCB detection result
        pcb_detected = True
        focus_score = 150.0
        processing_time = 0.05
        
        # Save processing result
        inspection_id = self.db.save_inspection(
            timestamp=datetime.now().isoformat(),
            defects=["Missing Hole"],
            locations=[{'bbox': [100, 100, 200, 200], 'confidence': 0.8}],
            confidence_scores=[0.8],
            focus_score=focus_score,
            processing_time=processing_time,
            pcb_area=50000,
            trigger_type='auto'
        )
        
        # Verify integration
        self.assertIsInstance(inspection_id, int)
        
        # Test analytics
        analysis = self.analyzer.get_realtime_analysis()
        quality_metrics = analysis['quality_metrics']
        
        self.assertIn('focus_score_stats', quality_metrics)
        self.assertEqual(quality_metrics['focus_score_stats']['mean'], focus_score)


def run_all_tests():
    """Run all data layer tests."""
    print("Running data layer tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestPCBDatabase,
        TestDefectAnalyzer,
        TestConcurrentAccess,
        TestPerformance,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_performance_tests():
    """Run performance tests only."""
    print("Running performance tests...")
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPerformance)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_integration_tests():
    """Run integration tests only."""
    print("Running integration tests...")
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestIntegration)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test data layer components")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.performance:
        success = run_performance_tests()
    elif args.integration:
        success = run_integration_tests()
    else:
        success = run_all_tests()
    
    print(f"\nTest result: {'PASS' if success else 'FAIL'}")
    sys.exit(0 if success else 1)