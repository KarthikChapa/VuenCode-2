"""
Advanced performance metrics tracking system for VuenCode competition.
Tracks latency, throughput, model performance, and system resources in real-time.
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import statistics
import threading
import psutil
import json
from pathlib import Path

from .config import get_config


@dataclass
class RequestMetrics:
    """Individual request performance metrics."""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    category: Optional[str] = None
    model_used: Optional[str] = None
    frame_count: int = 0
    processing_stages: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    cache_hit: bool = False
    
    @property
    def total_latency_ms(self) -> float:
        """Calculate total request latency in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000
    
    @property
    def is_successful(self) -> bool:
        """Check if request was successful."""
        return self.error is None and self.end_time is not None


@dataclass
class SystemMetrics:
    """System resource utilization metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    active_requests: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_utilization": self.gpu_utilization,
            "active_requests": self.active_requests
        }


class PerformanceTracker:
    """
    Advanced performance tracking system for competition optimization.
    
    Features:
    - Real-time latency tracking with percentiles
    - Category-specific performance analysis
    - System resource monitoring
    - Automatic performance alerts
    - Competition-grade metrics export
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Request tracking
        self.active_requests: Dict[str, RequestMetrics] = {}
        self.completed_requests: deque = deque(maxlen=10000)  # Last 10k requests
        self.request_lock = threading.Lock()
        
        # Category-specific metrics
        self.category_metrics: Dict[str, List[float]] = defaultdict(list)
        self.model_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # System monitoring
        self.system_metrics: deque = deque(maxlen=1000)  # Last 1000 system snapshots
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance alerts
        self.alert_thresholds = {
            "latency_p95_ms": self.config.target_latency_ms * 1.2,  # 20% above target
            "error_rate": 0.05,  # 5% error rate
            "cpu_percent": 90.0,
            "memory_percent": 85.0
        }
        
        # Competition metrics
        self.competition_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "throughput_rps": 0.0,
            "best_category_performance": {},
            "model_usage_stats": defaultdict(int)
        }
        
        self.logger.info(f"PerformanceTracker initialized for {self.config.deployment_mode} mode")
    
    def start_monitoring(self) -> None:
        """Start background system monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._system_monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("System monitoring stopped")
    
    def _system_monitor_loop(self) -> None:
        """Background loop for system resource monitoring."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU metrics (if available)
                gpu_memory_mb = None
                gpu_utilization = None
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(self.config.gpu_device_id)
                    gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_memory_mb = gpu_info.used / 1024 / 1024
                    gpu_utilization = gpu_util.gpu
                except Exception:
                    pass  # GPU monitoring not available
                
                # Create system metrics snapshot
                metrics = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    gpu_memory_mb=gpu_memory_mb,
                    gpu_utilization=gpu_utilization,
                    active_requests=len(self.active_requests)
                )
                
                self.system_metrics.append(metrics)
                
                # Check for performance alerts
                self._check_alerts(metrics)
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
            
            time.sleep(5)  # Monitor every 5 seconds
    
    @asynccontextmanager
    async def track_request(self, request_id: str, category: Optional[str] = None):
        """
        Async context manager for tracking individual request performance.
        
        Usage:
            async with tracker.track_request("req_123", "video_qa") as metrics:
                # Process request
                metrics.frame_count = 30
                metrics.model_used = "gemini-2.5-flash"
        """
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=time.time(),
            category=category
        )
        
        with self.request_lock:
            self.active_requests[request_id] = metrics
        
        try:
            yield metrics
        except Exception as e:
            metrics.error = str(e)
            self.logger.error(f"Request {request_id} failed: {e}")
        finally:
            metrics.end_time = time.time()
            
            with self.request_lock:
                self.active_requests.pop(request_id, None)
                self.completed_requests.append(metrics)
            
            # Update category and model metrics
            if metrics.is_successful:
                latency_ms = metrics.total_latency_ms
                
                if metrics.category:
                    self.category_metrics[metrics.category].append(latency_ms)
                
                if metrics.model_used:
                    self.model_metrics[metrics.model_used].append(latency_ms)
                    self.competition_stats["model_usage_stats"][metrics.model_used] += 1
            
            # Update competition stats
            self._update_competition_stats()
            
            self.logger.debug(f"Request {request_id} completed in {metrics.total_latency_ms:.2f}ms")
    
    def track_processing_stage(self, request_id: str, stage_name: str, duration_ms: float) -> None:
        """Track individual processing stage performance."""
        with self.request_lock:
            if request_id in self.active_requests:
                self.active_requests[request_id].processing_stages[stage_name] = duration_ms
    
    def _update_competition_stats(self) -> None:
        """Update aggregated competition statistics."""
        if not self.completed_requests:
            return
        
        recent_requests = list(self.completed_requests)[-1000:]  # Last 1000 requests
        successful_requests = [r for r in recent_requests if r.is_successful]
        
        if not successful_requests:
            return
        
        latencies = [r.total_latency_ms for r in successful_requests]
        
        self.competition_stats.update({
            "total_requests": len(recent_requests),
            "successful_requests": len(successful_requests),
            "average_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": self._percentile(latencies, 95),
            "p99_latency_ms": self._percentile(latencies, 99),
            "throughput_rps": self._calculate_throughput(recent_requests)
        })
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_throughput(self, requests: List[RequestMetrics]) -> float:
        """Calculate requests per second throughput."""
        if len(requests) < 2:
            return 0.0
        
        time_window = requests[-1].start_time - requests[0].start_time
        if time_window <= 0:
            return 0.0
        
        return len(requests) / time_window
    
    def _check_alerts(self, system_metrics: SystemMetrics) -> None:
        """Check for performance alerts based on current metrics."""
        alerts = []
        
        # Check system resource alerts
        if system_metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {system_metrics.cpu_percent:.1f}%")
        
        if system_metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {system_metrics.memory_percent:.1f}%")
        
        # Check performance alerts
        if self.competition_stats["p95_latency_ms"] > self.alert_thresholds["latency_p95_ms"]:
            alerts.append(f"High P95 latency: {self.competition_stats['p95_latency_ms']:.1f}ms")
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"PERFORMANCE ALERT: {alert}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for competition evaluation."""
        summary = {
            "competition_stats": self.competition_stats.copy(),
            "category_performance": {},
            "model_performance": {},
            "system_health": self._get_system_health(),
            "target_compliance": self._check_target_compliance()
        }
        
        # Category performance breakdown
        for category, latencies in self.category_metrics.items():
            if latencies:
                summary["category_performance"][category] = {
                    "count": len(latencies),
                    "avg_latency_ms": statistics.mean(latencies),
                    "p95_latency_ms": self._percentile(latencies, 95),
                    "min_latency_ms": min(latencies),
                    "max_latency_ms": max(latencies)
                }
        
        # Model performance breakdown
        for model, latencies in self.model_metrics.items():
            if latencies:
                summary["model_performance"][model] = {
                    "count": len(latencies),
                    "avg_latency_ms": statistics.mean(latencies),
                    "p95_latency_ms": self._percentile(latencies, 95)
                }
        
        return summary
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        if not self.system_metrics:
            return {"status": "unknown"}
        
        latest = self.system_metrics[-1]
        return {
            "status": "healthy" if latest.cpu_percent < 80 and latest.memory_percent < 80 else "stressed",
            "cpu_percent": latest.cpu_percent,
            "memory_percent": latest.memory_percent,
            "gpu_memory_mb": latest.gpu_memory_mb,
            "gpu_utilization": latest.gpu_utilization,
            "active_requests": latest.active_requests
        }
    
    def _check_target_compliance(self) -> Dict[str, Any]:
        """Check compliance with competition targets."""
        target_latency = self.config.target_latency_ms
        
        return {
            "target_latency_ms": target_latency,
            "actual_p95_latency_ms": self.competition_stats["p95_latency_ms"],
            "latency_compliance": self.competition_stats["p95_latency_ms"] <= target_latency,
            "error_rate": 1.0 - (self.competition_stats["successful_requests"] / 
                                max(1, self.competition_stats["total_requests"])),
            "throughput_rps": self.competition_stats["throughput_rps"]
        }
    
    def export_metrics(self, filepath: str) -> None:
        """Export performance metrics to JSON file for analysis."""
        summary = self.get_performance_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Performance metrics exported to {filepath}")
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics (useful for testing)."""
        with self.request_lock:
            self.active_requests.clear()
            self.completed_requests.clear()
            self.category_metrics.clear()
            self.model_metrics.clear()
            self.system_metrics.clear()
            
            self.competition_stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "average_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "throughput_rps": 0.0,
                "best_category_performance": {},
                "model_usage_stats": defaultdict(int)
            }
        
        self.logger.info("Performance metrics reset")


# Global performance tracker instance
_tracker: Optional[PerformanceTracker] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get or create the global performance tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
        _tracker.start_monitoring()
    return _tracker


def initialize_tracker(config=None) -> PerformanceTracker:
    """Initialize the global performance tracker with custom config."""
    global _tracker
    if _tracker:
        _tracker.stop_monitoring()
    
    _tracker = PerformanceTracker(config)
    _tracker.start_monitoring()
    return _tracker


# Convenience decorator for tracking function performance
def track_performance(category: Optional[str] = None):
    """Decorator for tracking function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            tracker = get_performance_tracker()
            request_id = f"{func.__name__}_{int(time.time() * 1000)}"
            
            async with tracker.track_request(request_id, category) as metrics:
                result = await func(*args, **kwargs)
                return result
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'll track manually
            tracker = get_performance_tracker()
            request_id = f"{func.__name__}_{int(time.time() * 1000)}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                
                metrics = RequestMetrics(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=end_time,
                    category=category
                )
                
                with tracker.request_lock:
                    tracker.completed_requests.append(metrics)
                
                return result
            except Exception as e:
                end_time = time.time()
                
                metrics = RequestMetrics(
                    request_id=request_id,
                    start_time=start_time,
                    end_time=end_time,
                    category=category,
                    error=str(e)
                )
                
                with tracker.request_lock:
                    tracker.completed_requests.append(metrics)
                
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator