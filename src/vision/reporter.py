"""
Report generator.

Two modes:
  - LIVE   : flushes a report every N minutes (20 / 30 / 60 / daily) and also at exit.
  - VIDEO  : flushes one report per video file (call flush() on each video boundary).

Each report is written as both CSV (events) and JSON (summary + events) into the
output directory, with a timestamped filename.
"""

import csv
import json
import os
import time
from datetime import datetime


class Reporter:
    MODE_LIVE = "live"
    MODE_VIDEO = "video"

    def __init__(self, mode, out_dir, interval_minutes=None, enabled=True):
        self.mode = mode
        self.out_dir = out_dir
        self.interval = (interval_minutes * 60) if interval_minutes else None
        self.enabled = enabled
        if enabled:
            os.makedirs(out_dir, exist_ok=True)
        self._reset_window()

    def _reset_window(self):
        self.window_start = time.time()
        self.events = []          # crossing + alert + abnormal events
        self.peak_count = 0
        self.crossings = 0
        self.alerts = 0
        self.label = "session"

    def set_label(self, label):
        self.label = label

    # ---- ingest -----------------------------------------------------------
    def log_crossing(self, ev):
        if not self.enabled:
            return
        self.crossings += 1
        self.events.append({"type": "crossing", **ev})

    def log_alert(self, name, timestamp, snapshot=None):
        if not self.enabled:
            return
        self.alerts += 1
        self.events.append({"type": "watchlist_alert", "name": name,
                            "time": timestamp, "snapshot": snapshot})

    def log_abnormal(self, item, timestamp):
        if not self.enabled:
            return
        self.events.append({"type": "abnormal", "time": timestamp, **item})

    def update_peak(self, count):
        self.peak_count = max(self.peak_count, count)

    # ---- flushing ---------------------------------------------------------
    def maybe_flush_live(self):
        """Called every frame in LIVE mode; flushes when the interval elapses."""
        if not self.enabled or self.mode != self.MODE_LIVE or self.interval is None:
            return
        if time.time() - self.window_start >= self.interval:
            self.flush()

    def flush(self):
        """Write the current window to disk and start a new window."""
        if not self.enabled:
            return None
        if not self.events and self.peak_count == 0:
            self._reset_window()
            return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() else "_" for c in self.label)
        base = os.path.join(self.out_dir, f"report_{safe_label}_{ts}")

        summary = {
            "mode": self.mode,
            "label": self.label,
            "window_start": datetime.fromtimestamp(self.window_start).isoformat(),
            "window_end": datetime.now().isoformat(),
            "total_crossings": self.crossings,
            "peak_persons_in_frame": self.peak_count,
            "watchlist_alerts": self.alerts,
            "events": self.events,
        }

        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["type", "time", "detail"])
            for ev in self.events:
                detail = {k: v for k, v in ev.items() if k not in ("type", "time")}
                w.writerow([ev.get("type"), ev.get("time"), json.dumps(detail)])

        print(f"[REPORT] Saved: {base}.json / .csv  "
              f"(crossings={self.crossings}, peak={self.peak_count}, alerts={self.alerts})")
        self._reset_window()
        return base


def ask_report_config():
    """Interactive prompt -> (enabled, interval_minutes_or_None)."""
    ans = input("Generate reports? (y/n): ").strip().lower()
    if ans != "y":
        return False, None
    print("Report interval (live mode):")
    print("  1 - every 20 minutes")
    print("  2 - every 30 minutes")
    print("  3 - hourly")
    print("  4 - entire day (once)")
    c = input("Enter 1/2/3/4 (default 3): ").strip()
    return True, {"1": 20, "2": 30, "3": 60, "4": 1440}.get(c, 60)
