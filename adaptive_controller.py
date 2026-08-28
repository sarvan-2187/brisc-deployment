"""Adaptive model-tier controller: switches the classifier+segmenter pair
between a high-accuracy tier and a lightweight tier based on rolling
inference latency and SoC temperature (via pi_thermal.read_temp_c), with
hysteresis so a single borderline reading doesn't flip the tier. Every
select_pair() call is appended to logs/decision_log.csv for later analysis.

Defaults are set from the Pi 4 numbers in benchmark.py's README table
(EfficientNetB0+DeepLabV3+ mean 1295.5ms/p95 1325.5ms vs MobileNetV3+UNet
mean 1114.6ms) and this codebase's existing 75C thermal-pause point
(pi_thermal.wait_for_safe_temp default) - see README for the full rationale.
"""

import csv
import time
from collections import deque
from pathlib import Path

from pi_thermal import read_temp_c

HIGH_ACCURACY = {"classifier": "efficientnet_b0", "segmenter": "deeplabv3plus"}
LIGHTWEIGHT = {"classifier": "mobilenet_v3", "segmenter": "unet"}

LOG_HEADER = ["timestamp", "temp_c", "avg_latency_ms", "pair", "reason"]


class AdaptiveController:
    def __init__(self, latency_threshold_ms=1350.0, temp_threshold_c=65.0,
                 window_size=5, consecutive_trigger_count=2,
                 log_path=Path("logs") / "decision_log.csv"):
        self.latency_threshold_ms = latency_threshold_ms
        self.temp_threshold_c = temp_threshold_c
        self.consecutive_trigger_count = consecutive_trigger_count
        self.log_path = Path(log_path)

        self._latencies = deque(maxlen=window_size)
        self._tier = "high_accuracy"
        self._trigger_streak = 0
        self._clear_streak = 0

    def record_inference(self, latency_ms):
        self._latencies.append(latency_ms)

    @property
    def avg_latency_ms(self):
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

    @property
    def current_tier(self):
        return self._tier

    def select_pair(self):
        temp = read_temp_c()
        avg_latency = self.avg_latency_ms

        temp_trigger = temp is not None and temp > self.temp_threshold_c
        latency_trigger = avg_latency > self.latency_threshold_ms
        triggered = temp_trigger or latency_trigger
        cause = "temp_trigger" if temp_trigger else ("latency_trigger" if latency_trigger else None)

        if triggered:
            self._trigger_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._trigger_streak = 0

        if self._tier == "high_accuracy":
            if triggered and self._trigger_streak >= self.consecutive_trigger_count:
                self._tier = "lightweight"
                reason = cause
            elif triggered:
                reason = "hysteresis_hold"
            else:
                reason = "nominal"
        else:
            if not triggered and self._clear_streak >= self.consecutive_trigger_count:
                self._tier = "high_accuracy"
                reason = "nominal"
            elif not triggered:
                reason = "hysteresis_hold"
            else:
                reason = cause

        pair = HIGH_ACCURACY if self._tier == "high_accuracy" else LIGHTWEIGHT
        self.last_reason = reason
        self._log_decision(temp, avg_latency, pair, reason)
        return pair

    def _log_decision(self, temp, avg_latency, pair, reason):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(LOG_HEADER)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{temp:.1f}" if temp is not None else "",
                f"{avg_latency:.1f}",
                f"{pair['classifier']}+{pair['segmenter']}",
                reason,
            ])


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "decision_log.csv"
        ctrl = AdaptiveController(
            latency_threshold_ms=1000.0, temp_threshold_c=1e9,  # temp never triggers off-Pi anyway
            window_size=3, consecutive_trigger_count=2, log_path=log_path,
        )

        assert ctrl.select_pair() == HIGH_ACCURACY  # no data yet -> nominal

        # One slow reading isn't enough to switch (hysteresis).
        ctrl.record_inference(2000)
        assert ctrl.select_pair() == HIGH_ACCURACY
        assert ctrl.last_reason == "hysteresis_hold"

        # A second consecutive slow reading flips it.
        ctrl.record_inference(2000)
        assert ctrl.select_pair() == LIGHTWEIGHT
        assert ctrl.last_reason == "latency_trigger"

        # Flush the rolling window with fast readings (window_size=3) so the
        # average actually drops below threshold before hysteresis applies.
        ctrl.record_inference(10)
        ctrl.record_inference(10)
        ctrl.record_inference(10)
        assert ctrl.avg_latency_ms < 1000.0

        # One clear reading isn't enough to switch back.
        assert ctrl.select_pair() == LIGHTWEIGHT
        assert ctrl.last_reason == "hysteresis_hold"

        # A second consecutive clear reading flips it back.
        assert ctrl.select_pair() == HIGH_ACCURACY
        assert ctrl.last_reason == "nominal"

        assert log_path.exists()
        rows = log_path.read_text().strip().splitlines()
        assert len(rows) == 6  # header + 5 select_pair() calls
        assert rows[0].split(",") == LOG_HEADER

    print("adaptive_controller.py self-check passed.")
