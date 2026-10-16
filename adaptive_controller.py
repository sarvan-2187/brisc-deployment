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
    def __init__(self, latency_threshold_ms=1300.0, temp_threshold_c=65.0,
                 window_size=5, consecutive_trigger_count=2,
                 log_path=Path("logs") / "decision_log.csv",
                 temp_release_c=60.0, temp_reader=read_temp_c, latency_release_ms=None):
        # Hysteresis on both signals: switch to lightweight when rolling latency
        # exceeds latency_threshold_ms or temp exceeds temp_threshold_c (engage),
        # but only return to high-accuracy once latency is at or below
        # latency_release_ms (default 95% of the threshold) AND temp is below
        # temp_release_c. Release == engage reproduces single-threshold behaviour.
        if latency_release_ms is None:
            latency_release_ms = 0.95 * latency_threshold_ms
        if temp_release_c > temp_threshold_c or latency_release_ms > latency_threshold_ms:
            raise ValueError("release thresholds must not exceed engage thresholds")
        self.latency_threshold_ms = latency_threshold_ms
        self.latency_release_ms = latency_release_ms
        self.temp_threshold_c = temp_threshold_c
        self.temp_release_c = temp_release_c
        self.window_size = window_size
        self.consecutive_trigger_count = consecutive_trigger_count
        self.log_path = Path(log_path)
        self._read_temp = temp_reader

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
        temp = self._read_temp()
        avg_latency = self.avg_latency_ms

        temp_trigger = temp is not None and temp > self.temp_threshold_c
        latency_trigger = avg_latency > self.latency_threshold_ms
        triggered = temp_trigger or latency_trigger
        cause = "temp_trigger" if temp_trigger else ("latency_trigger" if latency_trigger else None)
        released = avg_latency <= self.latency_release_ms and (temp is None or temp < self.temp_release_c)

        self._trigger_streak = self._trigger_streak + 1 if triggered else 0
        self._clear_streak = self._clear_streak + 1 if released else 0

        if self._tier == "high_accuracy":
            if triggered and self._trigger_streak >= self.consecutive_trigger_count:
                self._tier = "lightweight"
                self._clear_streak = 0
                reason = cause
            elif triggered:
                reason = "hysteresis_hold"
            else:
                reason = "nominal"
        else:
            if released and self._clear_streak >= self.consecutive_trigger_count:
                self._tier = "high_accuracy"
                self._trigger_streak = 0
                reason = "nominal"
            elif released:
                reason = "hysteresis_hold"
            else:
                # Still triggered, or in the band between release and engage.
                reason = cause or "release_pending"

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

        # Temperature hysteresis: engage above 65C, release only below 60C.
        temps = []
        hot = AdaptiveController(temp_threshold_c=65.0, temp_release_c=60.0,
                                 temp_reader=lambda: temps.pop(0), log_path=log_path)

        def step(t):
            temps.append(t)
            return hot.select_pair()

        assert step(70) == HIGH_ACCURACY and hot.last_reason == "hysteresis_hold"   # 1st hot reading
        assert step(70) == LIGHTWEIGHT and hot.last_reason == "temp_trigger"       # 2nd -> engage
        # Between release and engage: below the engage point but not released.
        for t in (62, 63, 64, 62):
            assert step(t) == LIGHTWEIGHT and hot.last_reason == "release_pending"
        # A borderline temperature flickering around 65C must not flip the tier.
        for t in (64.5, 65.5, 64.5, 65.5):
            assert step(t) == LIGHTWEIGHT
        assert step(59) == LIGHTWEIGHT and hot.last_reason == "hysteresis_hold"     # 1st release reading
        assert step(61) == LIGHTWEIGHT and hot.last_reason == "release_pending"     # streak broken
        assert step(59) == LIGHTWEIGHT                                              # streak restarts
        assert step(58) == HIGH_ACCURACY and hot.last_reason == "nominal"           # 2nd -> release
        # Back on high-accuracy, 62C is not a trigger (below 65C engage).
        assert step(62) == HIGH_ACCURACY and hot.last_reason == "nominal"

        # Latency alone holds the lightweight tier even when the SoC is cool.
        slow = AdaptiveController(latency_threshold_ms=1350.0, temp_release_c=60.0,
                                  temp_reader=lambda: 50.0, log_path=log_path)
        for _ in range(5):
            slow.record_inference(1500)
        slow.select_pair()
        assert slow.select_pair() == LIGHTWEIGHT and slow.last_reason == "latency_trigger"
        assert slow.select_pair() == LIGHTWEIGHT  # cool but still slow: not released
        for _ in range(5):
            slow.record_inference(1100)
        slow.select_pair()
        assert slow.select_pair() == HIGH_ACCURACY  # cool and fast for 2 checks

        # Latency hysteresis (the E4 frequency-cap case): a rolling mean just under the
        # 1,300 ms engage threshold, but above the 1,235 ms release point, must not
        # release - otherwise one fast gated slice in the window flips the tier back.
        band = AdaptiveController(latency_threshold_ms=1300.0, temp_reader=lambda: 50.0, log_path=log_path)
        assert band.latency_release_ms == 0.95 * 1300.0
        for _ in range(5):
            band.record_inference(1500)
        band.select_pair()
        assert band.select_pair() == LIGHTWEIGHT
        for ms in (1298, 1290, 1270, 1250, 1240):
            for _ in range(5):
                band.record_inference(ms)
            assert band.select_pair() == LIGHTWEIGHT and band.last_reason == "release_pending"
        for _ in range(5):
            band.record_inference(1150)
        assert band.select_pair() == LIGHTWEIGHT and band.last_reason == "hysteresis_hold"
        assert band.select_pair() == HIGH_ACCURACY

        try:
            AdaptiveController(temp_threshold_c=60.0, temp_release_c=65.0)
            raise AssertionError("release above engage must be rejected")
        except ValueError:
            pass
        try:
            AdaptiveController(latency_threshold_ms=1300.0, latency_release_ms=1400.0)
            raise AssertionError("latency release above engage must be rejected")
        except ValueError:
            pass

    print("adaptive_controller.py self-check passed.")
