"""
Analytics module for PCB inspection system.

This module provides comprehensive analytics, reporting, and trend analysis
for PCB inspection data including real-time metrics and historical analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
from dataclasses import dataclass
from collections import defaultdict
import statistics

from core.interfaces import BaseAnalyzer
from core.config import DEFECT_CLASSES, DEFECT_COLORS
from data.database import PCBDatabase


@dataclass
class InspectionMetrics:
    """Data class for inspection metrics."""
    total_inspections: int
    defect_inspections: int
    defect_rate: float
    avg_processing_time: float
    avg_focus_score: float
    avg_defects_per_inspection: float
    throughput_per_hour: float


@dataclass
class DefectMetrics:
    """Data class for defect metrics."""
    defect_type: str
    total_count: int
    percentage: float
    avg_confidence: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    avg_inference_time: float
    avg_preprocessing_time: float
    avg_total_time: float
    fps: float
    memory_usage_mb: float
    gpu_utilization: float
    uptime_hours: float


class DefectAnalyzer(BaseAnalyzer):
    """
    Comprehensive defect analysis and reporting system.
    
    Provides real-time analytics, trend analysis, and reporting
    capabilities for PCB inspection data.
    """
    
    def __init__(self, database: PCBDatabase):
        """
        Initialize defect analyzer.
        
        Args:
            database: Database instance
        """
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Cache for performance
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
        self._last_cache_update = {}
        
        self.logger.info("DefectAnalyzer initialized")
    
    def analyze(self, data: Any) -> Dict[str, Any]:
        """
        Analyze inspection data.
        
        Args:
            data: Data to analyze (can be inspection results or time period)
            
        Returns:
            Analysis results
        """
        if isinstance(data, dict) and 'time_period' in data:
            return self.get_time_period_analysis(data['time_period'])
        else:
            return self.get_realtime_analysis()
    
    def get_realtime_analysis(self) -> Dict[str, Any]:
        """
        Get real-time analysis of current system state.
        
        Returns:
            Real-time analysis results
        """
        # Check cache
        if self._is_cache_valid('realtime'):
            return self._cache['realtime']
        
        try:
            # Get recent data
            recent_inspections = self.database.get_recent_inspections(100)
            defect_stats = self.database.get_defect_statistics()
            system_stats = self.database.get_system_statistics(7)  # Last 7 days
            
            # Calculate metrics
            inspection_metrics = self._calculate_inspection_metrics(recent_inspections)
            defect_metrics = self._calculate_defect_metrics(defect_stats, recent_inspections)
            quality_metrics = self._calculate_quality_metrics(recent_inspections)
            performance_metrics = self._calculate_performance_metrics(recent_inspections)
            
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'inspection_metrics': inspection_metrics.__dict__,
                'defect_metrics': [dm.__dict__ for dm in defect_metrics],
                'quality_metrics': quality_metrics,
                'performance_metrics': performance_metrics.__dict__,
                'system_health': self._assess_system_health(inspection_metrics, defect_metrics),
                'alerts': self._generate_alerts(inspection_metrics, defect_metrics),
                'summary': self._generate_summary(inspection_metrics, defect_metrics)
            }
            
            # Cache result
            self._cache['realtime'] = analysis
            self._last_cache_update['realtime'] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in realtime analysis: {str(e)}")
            return {'error': str(e)}
    
    def get_time_period_analysis(self, period: str) -> Dict[str, Any]:
        """
        Get analysis for specific time period.
        
        Args:
            period: Time period ('1d', '7d', '30d', '90d')
            
        Returns:
            Time period analysis
        """
        # Check cache
        cache_key = f'period_{period}'
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Calculate date range
            days = {'1d': 1, '7d': 7, '30d': 30, '90d': 90}.get(period, 7)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get data
            inspections = self.database.get_inspections_by_date_range(start_date, end_date)
            system_stats = self.database.get_system_statistics(days)
            
            # Calculate analysis
            analysis = {
                'period': period,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'daily_trends': self._calculate_daily_trends(inspections),
                'defect_trends': self._calculate_defect_trends(inspections),
                'performance_trends': self._calculate_performance_trends(inspections),
                'quality_trends': self._calculate_quality_trends(inspections),
                'hourly_patterns': self._calculate_hourly_patterns(inspections),
                'weekly_patterns': self._calculate_weekly_patterns(inspections),
                'comparative_analysis': self._calculate_comparative_analysis(inspections, period)
            }
            
            # Cache result
            self._cache[cache_key] = analysis
            self._last_cache_update[cache_key] = datetime.now()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in time period analysis: {str(e)}")
            return {'error': str(e)}
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Get comprehensive system report with all metrics.
        
        Returns:
            Comprehensive report dictionary
        """
        try:
            # Get basic realtime stats
            realtime_stats = self.get_realtime_analysis()
            
            # Get recent inspections for additional metrics
            recent_inspections = self.database.get_recent_inspections(100)
            
            # Calculate summary metrics
            total_inspections = len(recent_inspections)
            total_defects = sum(len(json.loads(insp.get('defects', '[]'))) 
                              for insp in recent_inspections if insp.get('defects'))
            
            pass_rate = 100.0
            if total_inspections > 0:
                failed_inspections = sum(1 for insp in recent_inspections 
                                       if insp.get('has_defects', False))
                pass_rate = ((total_inspections - failed_inspections) / total_inspections) * 100
            
            return {
                'total_inspections': total_inspections,
                'total_defects': total_defects,
                'pass_rate': pass_rate,
                'defect_rate': 100 - pass_rate,
                'realtime_stats': realtime_stats,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            return {
                'total_inspections': 0,
                'total_defects': 0,
                'pass_rate': 100.0,
                'defect_rate': 0.0,
                'error': str(e)
            }
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """Alias for get_realtime_analysis for compatibility."""
        return self.get_realtime_analysis()
    
    def _calculate_inspection_metrics(self, inspections: List[Dict]) -> InspectionMetrics:
        """Calculate inspection metrics."""
        if not inspections:
            return InspectionMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        total_inspections = len(inspections)
        defect_inspections = sum(1 for i in inspections if i['has_defects'])
        defect_rate = defect_inspections / total_inspections if total_inspections > 0 else 0.0
        
        # Calculate averages
        processing_times = [i['processing_time'] for i in inspections if i['processing_time']]
        focus_scores = [i['focus_score'] for i in inspections if i['focus_score']]
        defect_counts = [i['defect_count'] for i in inspections]
        
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0
        avg_focus_score = statistics.mean(focus_scores) if focus_scores else 0.0
        avg_defects_per_inspection = statistics.mean(defect_counts) if defect_counts else 0.0
        
        # Calculate throughput (inspections per hour)
        if len(inspections) > 1:
            time_span = self._parse_timestamp(inspections[0]['timestamp']) - self._parse_timestamp(inspections[-1]['timestamp'])
            hours = time_span.total_seconds() / 3600
            throughput_per_hour = total_inspections / hours if hours > 0 else 0.0
        else:
            throughput_per_hour = 0.0
        
        return InspectionMetrics(
            total_inspections=total_inspections,
            defect_inspections=defect_inspections,
            defect_rate=defect_rate,
            avg_processing_time=avg_processing_time,
            avg_focus_score=avg_focus_score,
            avg_defects_per_inspection=avg_defects_per_inspection,
            throughput_per_hour=throughput_per_hour
        )
    
    def _calculate_defect_metrics(self, defect_stats: List[Dict], 
                                 recent_inspections: List[Dict]) -> List[DefectMetrics]:
        """Calculate defect metrics."""
        metrics = []
        
        # Total defects across all types
        total_defects = sum(stat['total_count'] for stat in defect_stats)
        
        for stat in defect_stats:
            defect_type = stat['defect_type']
            total_count = stat['total_count']
            percentage = (total_count / total_defects * 100) if total_defects > 0 else 0.0
            avg_confidence = stat['avg_confidence']
            
            # Calculate trend
            trend = self._calculate_defect_trend(defect_type, recent_inspections)
            
            # Parse dates
            first_seen = self._parse_timestamp(stat['first_seen']) if stat['first_seen'] else None
            last_seen = self._parse_timestamp(stat['last_seen']) if stat['last_seen'] else None
            
            metrics.append(DefectMetrics(
                defect_type=defect_type,
                total_count=total_count,
                percentage=percentage,
                avg_confidence=avg_confidence,
                trend=trend,
                first_seen=first_seen,
                last_seen=last_seen
            ))
        
        # Sort by total count
        metrics.sort(key=lambda x: x.total_count, reverse=True)
        
        return metrics
    
    def _calculate_quality_metrics(self, inspections: List[Dict]) -> Dict[str, Any]:
        """Calculate quality metrics."""
        if not inspections:
            return {}
        
        # Focus score analysis
        focus_scores = [i['focus_score'] for i in inspections if i['focus_score']]
        
        quality_metrics = {
            'focus_score_stats': {
                'min': min(focus_scores) if focus_scores else 0,
                'max': max(focus_scores) if focus_scores else 0,
                'mean': statistics.mean(focus_scores) if focus_scores else 0,
                'median': statistics.median(focus_scores) if focus_scores else 0,
                'std': statistics.stdev(focus_scores) if len(focus_scores) > 1 else 0
            },
            'focus_score_distribution': self._calculate_focus_distribution(focus_scores),
            'quality_grade': self._calculate_quality_grade(focus_scores),
            'pass_rate': self._calculate_pass_rate(inspections),
            'defect_severity_distribution': self._calculate_defect_severity(inspections)
        }
        
        return quality_metrics
    
    def _calculate_performance_metrics(self, inspections: List[Dict]) -> PerformanceMetrics:
        """Calculate performance metrics."""
        if not inspections:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)
        
        processing_times = [i['processing_time'] for i in inspections if i['processing_time']]
        
        avg_total_time = statistics.mean(processing_times) if processing_times else 0
        fps = 1.0 / avg_total_time if avg_total_time > 0 else 0
        
        # Get performance metrics from database
        perf_metrics = self.database.get_performance_metrics('inference_time', 24)
        
        inference_times = [m['metric_value'] for m in perf_metrics]
        avg_inference_time = statistics.mean(inference_times) if inference_times else 0
        
        return PerformanceMetrics(
            avg_inference_time=avg_inference_time,
            avg_preprocessing_time=avg_total_time - avg_inference_time,
            avg_total_time=avg_total_time,
            fps=fps,
            memory_usage_mb=0,  # Would need to be tracked separately
            gpu_utilization=0,   # Would need to be tracked separately
            uptime_hours=0       # Would need to be tracked separately
        )
    
    def _calculate_defect_trend(self, defect_type: str, inspections: List[Dict]) -> str:
        """Calculate trend for specific defect type."""
        # Get recent occurrences
        recent_occurrences = []
        for inspection in inspections:
            if inspection['defects']:
                defects = json.loads(inspection['defects']) if isinstance(inspection['defects'], str) else inspection['defects']
                count = defects.count(defect_type)
                if count > 0:
                    recent_occurrences.append({
                        'timestamp': inspection['timestamp'],
                        'count': count
                    })
        
        if len(recent_occurrences) < 2:
            return 'stable'
        
        # Split into two halves and compare
        mid = len(recent_occurrences) // 2
        first_half = recent_occurrences[:mid]
        second_half = recent_occurrences[mid:]
        
        first_avg = statistics.mean([r['count'] for r in first_half])
        second_avg = statistics.mean([r['count'] for r in second_half])
        
        if second_avg > first_avg * 1.2:
            return 'increasing'
        elif second_avg < first_avg * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_daily_trends(self, inspections: List[Dict]) -> List[Dict]:
        """Calculate daily trends."""
        daily_data = defaultdict(lambda: {
            'total_inspections': 0,
            'defect_inspections': 0,
            'total_defects': 0,
            'avg_focus_score': 0,
            'avg_processing_time': 0,
            'focus_scores': [],
            'processing_times': []
        })
        
        for inspection in inspections:
            timestamp = self._parse_timestamp(inspection['timestamp'])
            date_key = timestamp.strftime('%Y-%m-%d')
            
            daily_data[date_key]['total_inspections'] += 1
            if inspection['has_defects']:
                daily_data[date_key]['defect_inspections'] += 1
                daily_data[date_key]['total_defects'] += inspection['defect_count']
            
            if inspection['focus_score']:
                daily_data[date_key]['focus_scores'].append(inspection['focus_score'])
            
            if inspection['processing_time']:
                daily_data[date_key]['processing_times'].append(inspection['processing_time'])
        
        # Calculate averages
        trends = []
        for date, data in sorted(daily_data.items()):
            avg_focus = statistics.mean(data['focus_scores']) if data['focus_scores'] else 0
            avg_processing = statistics.mean(data['processing_times']) if data['processing_times'] else 0
            defect_rate = data['defect_inspections'] / data['total_inspections'] if data['total_inspections'] > 0 else 0
            
            trends.append({
                'date': date,
                'total_inspections': data['total_inspections'],
                'defect_inspections': data['defect_inspections'],
                'defect_rate': defect_rate,
                'total_defects': data['total_defects'],
                'avg_focus_score': avg_focus,
                'avg_processing_time': avg_processing
            })
        
        return trends
    
    def _calculate_defect_trends(self, inspections: List[Dict]) -> Dict[str, List[Dict]]:
        """Calculate defect trends by type."""
        defect_trends = {defect_type: [] for defect_type in DEFECT_CLASSES}
        
        # Group by date
        daily_defects = defaultdict(lambda: {defect_type: 0 for defect_type in DEFECT_CLASSES})
        
        for inspection in inspections:
            timestamp = self._parse_timestamp(inspection['timestamp'])
            date_key = timestamp.strftime('%Y-%m-%d')
            
            if inspection['defects']:
                defects = json.loads(inspection['defects']) if isinstance(inspection['defects'], str) else inspection['defects']
                for defect in defects:
                    if defect in daily_defects[date_key]:
                        daily_defects[date_key][defect] += 1
        
        # Convert to trends format
        for defect_type in DEFECT_CLASSES:
            trend_data = []
            for date in sorted(daily_defects.keys()):
                trend_data.append({
                    'date': date,
                    'count': daily_defects[date][defect_type]
                })
            defect_trends[defect_type] = trend_data
        
        return defect_trends
    
    def _calculate_performance_trends(self, inspections: List[Dict]) -> List[Dict]:
        """Calculate performance trends."""
        daily_performance = defaultdict(lambda: {
            'processing_times': [],
            'focus_scores': []
        })
        
        for inspection in inspections:
            timestamp = self._parse_timestamp(inspection['timestamp'])
            date_key = timestamp.strftime('%Y-%m-%d')
            
            if inspection['processing_time']:
                daily_performance[date_key]['processing_times'].append(inspection['processing_time'])
            
            if inspection['focus_score']:
                daily_performance[date_key]['focus_scores'].append(inspection['focus_score'])
        
        trends = []
        for date in sorted(daily_performance.keys()):
            data = daily_performance[date]
            
            avg_processing = statistics.mean(data['processing_times']) if data['processing_times'] else 0
            avg_focus = statistics.mean(data['focus_scores']) if data['focus_scores'] else 0
            fps = 1.0 / avg_processing if avg_processing > 0 else 0
            
            trends.append({
                'date': date,
                'avg_processing_time': avg_processing,
                'avg_focus_score': avg_focus,
                'fps': fps
            })
        
        return trends
    
    def _calculate_quality_trends(self, inspections: List[Dict]) -> List[Dict]:
        """Calculate quality trends."""
        daily_quality = defaultdict(lambda: {
            'total_inspections': 0,
            'passed_inspections': 0,
            'focus_scores': []
        })
        
        for inspection in inspections:
            timestamp = self._parse_timestamp(inspection['timestamp'])
            date_key = timestamp.strftime('%Y-%m-%d')
            
            daily_quality[date_key]['total_inspections'] += 1
            if not inspection['has_defects']:
                daily_quality[date_key]['passed_inspections'] += 1
            
            if inspection['focus_score']:
                daily_quality[date_key]['focus_scores'].append(inspection['focus_score'])
        
        trends = []
        for date in sorted(daily_quality.keys()):
            data = daily_quality[date]
            
            pass_rate = data['passed_inspections'] / data['total_inspections'] if data['total_inspections'] > 0 else 0
            avg_focus = statistics.mean(data['focus_scores']) if data['focus_scores'] else 0
            
            trends.append({
                'date': date,
                'pass_rate': pass_rate,
                'avg_focus_score': avg_focus,
                'quality_grade': self._get_quality_grade_from_focus(avg_focus)
            })
        
        return trends
    
    def _calculate_hourly_patterns(self, inspections: List[Dict]) -> List[Dict]:
        """Calculate hourly patterns."""
        hourly_data = defaultdict(lambda: {
            'total_inspections': 0,
            'defect_inspections': 0,
            'total_defects': 0
        })
        
        for inspection in inspections:
            timestamp = self._parse_timestamp(inspection['timestamp'])
            hour = timestamp.hour
            
            hourly_data[hour]['total_inspections'] += 1
            if inspection['has_defects']:
                hourly_data[hour]['defect_inspections'] += 1
                hourly_data[hour]['total_defects'] += inspection['defect_count']
        
        patterns = []
        for hour in range(24):
            data = hourly_data[hour]
            defect_rate = data['defect_inspections'] / data['total_inspections'] if data['total_inspections'] > 0 else 0
            
            patterns.append({
                'hour': hour,
                'total_inspections': data['total_inspections'],
                'defect_rate': defect_rate,
                'total_defects': data['total_defects']
            })
        
        return patterns
    
    def _calculate_weekly_patterns(self, inspections: List[Dict]) -> List[Dict]:
        """Calculate weekly patterns."""
        weekly_data = defaultdict(lambda: {
            'total_inspections': 0,
            'defect_inspections': 0,
            'total_defects': 0
        })
        
        for inspection in inspections:
            timestamp = self._parse_timestamp(inspection['timestamp'])
            weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
            
            weekly_data[weekday]['total_inspections'] += 1
            if inspection['has_defects']:
                weekly_data[weekday]['defect_inspections'] += 1
                weekly_data[weekday]['total_defects'] += inspection['defect_count']
        
        patterns = []
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for i in range(7):
            data = weekly_data[i]
            defect_rate = data['defect_inspections'] / data['total_inspections'] if data['total_inspections'] > 0 else 0
            
            patterns.append({
                'weekday': weekdays[i],
                'weekday_num': i,
                'total_inspections': data['total_inspections'],
                'defect_rate': defect_rate,
                'total_defects': data['total_defects']
            })
        
        return patterns
    
    def _calculate_comparative_analysis(self, inspections: List[Dict], period: str) -> Dict[str, Any]:
        """Calculate comparative analysis with previous period."""
        # Split inspections into current and previous periods
        mid_point = len(inspections) // 2
        current_period = inspections[:mid_point]
        previous_period = inspections[mid_point:]
        
        # Calculate metrics for both periods
        current_metrics = self._calculate_inspection_metrics(current_period)
        previous_metrics = self._calculate_inspection_metrics(previous_period)
        
        # Calculate changes
        changes = {}
        for attr in ['defect_rate', 'avg_processing_time', 'avg_focus_score', 'throughput_per_hour']:
            current_value = getattr(current_metrics, attr, 0)
            previous_value = getattr(previous_metrics, attr, 0)
            
            if previous_value > 0:
                change_percent = ((current_value - previous_value) / previous_value) * 100
            else:
                change_percent = 0
            
            changes[attr] = {
                'current': current_value,
                'previous': previous_value,
                'change_percent': change_percent,
                'trend': 'improving' if change_percent > 5 else 'declining' if change_percent < -5 else 'stable'
            }
        
        return changes
    
    def _calculate_focus_distribution(self, focus_scores: List[float]) -> Dict[str, int]:
        """Calculate focus score distribution."""
        if not focus_scores:
            return {}
        
        distribution = {
            'excellent': 0,    # > 200
            'good': 0,         # 150-200
            'acceptable': 0,   # 100-150
            'poor': 0,         # 50-100
            'very_poor': 0     # < 50
        }
        
        for score in focus_scores:
            if score > 200:
                distribution['excellent'] += 1
            elif score > 150:
                distribution['good'] += 1
            elif score > 100:
                distribution['acceptable'] += 1
            elif score > 50:
                distribution['poor'] += 1
            else:
                distribution['very_poor'] += 1
        
        return distribution
    
    def _calculate_quality_grade(self, focus_scores: List[float]) -> str:
        """Calculate overall quality grade."""
        if not focus_scores:
            return 'Unknown'
        
        avg_score = statistics.mean(focus_scores)
        
        if avg_score > 180:
            return 'A'
        elif avg_score > 150:
            return 'B'
        elif avg_score > 120:
            return 'C'
        elif avg_score > 80:
            return 'D'
        else:
            return 'F'
    
    def _get_quality_grade_from_focus(self, avg_focus: float) -> str:
        """Get quality grade from focus score."""
        if avg_focus > 180:
            return 'A'
        elif avg_focus > 150:
            return 'B'
        elif avg_focus > 120:
            return 'C'
        elif avg_focus > 80:
            return 'D'
        else:
            return 'F'
    
    def _calculate_pass_rate(self, inspections: List[Dict]) -> float:
        """Calculate pass rate (no defects)."""
        if not inspections:
            return 0.0
        
        passed = sum(1 for i in inspections if not i['has_defects'])
        return passed / len(inspections)
    
    def _calculate_defect_severity(self, inspections: List[Dict]) -> Dict[str, int]:
        """Calculate defect severity distribution."""
        severity = {
            'low': 0,      # 1 defect
            'medium': 0,   # 2-3 defects
            'high': 0,     # 4+ defects
            'critical': 0  # Specific defect types
        }
        
        critical_defects = ['Short Circuit', 'Open Circuit']
        
        for inspection in inspections:
            if not inspection['has_defects']:
                continue
            
            defect_count = inspection['defect_count']
            defects = json.loads(inspection['defects']) if isinstance(inspection['defects'], str) else inspection['defects']
            
            # Check for critical defects
            has_critical = any(defect in critical_defects for defect in defects)
            
            if has_critical:
                severity['critical'] += 1
            elif defect_count >= 4:
                severity['high'] += 1
            elif defect_count >= 2:
                severity['medium'] += 1
            else:
                severity['low'] += 1
        
        return severity
    
    def _assess_system_health(self, inspection_metrics: InspectionMetrics,
                             defect_metrics: List[DefectMetrics]) -> Dict[str, Any]:
        """Assess overall system health."""
        health_score = 100.0
        issues = []
        
        # Check defect rate
        if inspection_metrics.defect_rate > 0.1:  # >10% defect rate
            health_score -= 20
            issues.append("High defect rate detected")
        
        # Check processing time
        if inspection_metrics.avg_processing_time > 0.5:  # >0.5s processing time
            health_score -= 10
            issues.append("Slow processing detected")
        
        # Check focus score
        if inspection_metrics.avg_focus_score < 100:  # Low focus
            health_score -= 15
            issues.append("Low focus quality detected")
        
        # Check throughput
        if inspection_metrics.throughput_per_hour < 10:  # <10 inspections/hour
            health_score -= 10
            issues.append("Low throughput detected")
        
        # Check for trending defects
        trending_defects = [dm for dm in defect_metrics if dm.trend == 'increasing']
        if trending_defects:
            health_score -= 10
            issues.append(f"Increasing trend in {len(trending_defects)} defect types")
        
        # Determine health status
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 75:
            status = 'good'
        elif health_score >= 60:
            status = 'fair'
        elif health_score >= 40:
            status = 'poor'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'score': health_score,
            'issues': issues
        }
    
    def _generate_alerts(self, inspection_metrics: InspectionMetrics,
                        defect_metrics: List[DefectMetrics]) -> List[Dict[str, Any]]:
        """Generate system alerts."""
        alerts = []
        
        # High defect rate alert
        if inspection_metrics.defect_rate > 0.15:
            alerts.append({
                'type': 'error',
                'message': f"High defect rate: {inspection_metrics.defect_rate:.1%}",
                'severity': 'high'
            })
        
        # Performance alert
        if inspection_metrics.avg_processing_time > 0.5:
            alerts.append({
                'type': 'warning',
                'message': f"Slow processing: {inspection_metrics.avg_processing_time:.2f}s",
                'severity': 'medium'
            })
        
        # Focus quality alert
        if inspection_metrics.avg_focus_score < 80:
            alerts.append({
                'type': 'warning',
                'message': f"Low focus quality: {inspection_metrics.avg_focus_score:.1f}",
                'severity': 'medium'
            })
        
        # Trending defect alerts
        for dm in defect_metrics:
            if dm.trend == 'increasing' and dm.total_count > 10:
                alerts.append({
                    'type': 'info',
                    'message': f"Increasing trend: {dm.defect_type}",
                    'severity': 'low'
                })
        
        return alerts
    
    def _generate_summary(self, inspection_metrics: InspectionMetrics,
                         defect_metrics: List[DefectMetrics]) -> Dict[str, Any]:
        """Generate system summary."""
        top_defects = sorted(defect_metrics, key=lambda x: x.total_count, reverse=True)[:3]
        
        summary = {
            'total_inspections': inspection_metrics.total_inspections,
            'defect_rate': f"{inspection_metrics.defect_rate:.1%}",
            'avg_processing_time': f"{inspection_metrics.avg_processing_time:.2f}s",
            'throughput': f"{inspection_metrics.throughput_per_hour:.1f}/hour",
            'top_defects': [
                {
                    'type': dm.defect_type,
                    'count': dm.total_count,
                    'percentage': f"{dm.percentage:.1f}%"
                } for dm in top_defects
            ]
        }
        
        return summary
    
    def _parse_timestamp(self, timestamp: str) -> datetime:
        """Parse timestamp string to datetime."""
        if isinstance(timestamp, datetime):
            return timestamp
        
        try:
            # Try different formats
            formats = [
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp, fmt)
                except ValueError:
                    continue
            
            # Fallback to fromisoformat
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
        except Exception:
            return datetime.now()
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache is valid."""
        if key not in self._cache:
            return False
        
        if key not in self._last_cache_update:
            return False
        
        age = (datetime.now() - self._last_cache_update[key]).total_seconds()
        return age < self._cache_timeout
    
    def generate_report(self, period: str = '7d', format: str = 'json') -> Any:
        """
        Generate comprehensive report.
        
        Args:
            period: Time period for report
            format: Output format ('json', 'html', 'pdf')
            
        Returns:
            Report in requested format
        """
        analysis = self.get_time_period_analysis(period)
        
        if format == 'json':
            return analysis
        elif format == 'html':
            return self._generate_html_report(analysis)
        elif format == 'pdf':
            return self._generate_pdf_report(analysis)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_html_report(self, analysis: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PCB Inspection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background: #e0e0e0; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>PCB Inspection System Report</h1>
                <p>Period: {period}</p>
                <p>Generated: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <!-- Add summary content here -->
            </div>
            
            <div class="section">
                <h2>Daily Trends</h2>
                <!-- Add trends content here -->
            </div>
            
            <div class="section">
                <h2>Defect Analysis</h2>
                <!-- Add defect analysis here -->
            </div>
        </body>
        </html>
        """
        
        return html_template.format(
            period=analysis.get('period', ''),
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _generate_pdf_report(self, analysis: Dict[str, Any]) -> bytes:
        """Generate PDF report."""
        # This would require a PDF library like reportlab
        # For now, return placeholder
        return b"PDF report generation not implemented"
    
    def clear_cache(self):
        """Clear analysis cache."""
        self._cache.clear()
        self._last_cache_update.clear()
        self.logger.info("Analysis cache cleared")


# Utility functions
def create_sample_analyzer(db_path: str = None) -> DefectAnalyzer:
    """Create analyzer with sample data."""
    from data.database import PCBDatabase, populate_test_data
    
    if db_path:
        db = PCBDatabase(db_path)
    else:
        db = PCBDatabase("sample_analytics.db")
    
    # Add sample data
    populate_test_data(db, 200)
    
    return DefectAnalyzer(db)


if __name__ == "__main__":
    # Test analyzer
    analyzer = create_sample_analyzer()
    
    # Test real-time analysis
    print("Testing real-time analysis...")
    realtime = analyzer.get_realtime_analysis()
    print(f"Real-time analysis keys: {list(realtime.keys())}")
    
    # Test period analysis
    print("\nTesting period analysis...")
    period_analysis = analyzer.get_time_period_analysis('7d')
    print(f"Period analysis keys: {list(period_analysis.keys())}")
    
    # Test report generation
    print("\nTesting report generation...")
    report = analyzer.generate_report('7d', 'json')
    print(f"Report generated with {len(report)} sections")
    
    print("Analytics test completed successfully!")