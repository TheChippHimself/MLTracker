"""Performance monitoring for the inference loop."""

import logging
import time

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Tracks FPS and per-stage latency."""

    def __init__(self, target_fps: float = 15.0, log_interval_s: float = 10.0, alpha: float = 0.1) -> None:
        self.target_fps = target_fps
        self.log_interval_s = log_interval_s
        self.alpha = alpha
        
        self.fps = 0.0
        self._latencies: dict[str, float] = {}
        
        self._last_frame_time = time.monotonic()
        self._last_log_time = time.monotonic()
        self._stage_start_time = 0.0

    def mark_frame_start(self) -> None:
        """Called at the very beginning of the loop."""
        now = time.monotonic()
        dt = now - self._last_frame_time
        if dt > 0:
            current_fps = 1.0 / dt
            if self.fps == 0.0:
                self.fps = current_fps
            else:
                self.fps = (current_fps * self.alpha) + (self.fps * (1.0 - self.alpha))
        self._last_frame_time = now

    def start_stage(self) -> None:
        """Start timing a specific stage."""
        self._stage_start_time = time.monotonic()

    def end_stage(self, stage_name: str) -> None:
        """End timing a stage and record its latency."""
        dt = (time.monotonic() - self._stage_start_time) * 1000.0  # ms
        
        if stage_name not in self._latencies:
            self._latencies[stage_name] = dt
        else:
            self._latencies[stage_name] = (dt * self.alpha) + (self._latencies[stage_name] * (1.0 - self.alpha))

    def log_summary_if_needed(self) -> None:
        """Log a performance summary periodically."""
        now = time.monotonic()
        if (now - self._last_log_time) >= self.log_interval_s:
            self._last_log_time = now
            
            lat_str = ", ".join(f"{k}: {v:.1f}ms" for k, v in self._latencies.items())
            logger.info("Perf | FPS: %.1f | %s", self.fps, lat_str)
            
            if self.fps > 0 and self.fps < self.target_fps:
                logger.warning("FPS (%.1f) dropped below target (%.1f)", self.fps, self.target_fps)
