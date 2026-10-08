"""
Instrumented Dataset
====================
Subclass ของ MinatoLoader's PytTrain ที่เพิ่ม per-sample logging
โดยไม่แก้โค้ดต้นฉบับแม้แต่บรรทัดเดียว

Architecture:
  PytTrain.__getitem__() returns None  → sample ใช้เวลาเกิน timeout → slow path
  PytTrain.__getitem__() returns data  → sample มาทัน → fast path

เราจับ event นี้ตรงนี้แล้ว log ลง JSONL
"""
import sys
import os
import time
import json
import threading
from pathlib import Path

# ── Import MinatoLoader's Dataset (no modification) ───────────
_MINATO_DIR = Path(__file__).parent.parent / "MinatoLoader" / "Minato"
if str(_MINATO_DIR) not in sys.path:
    sys.path.insert(0, str(_MINATO_DIR))

from data_loading.pytorch_loader import PytTrain, PytVal


class InstrumentedPytTrain(PytTrain):
    """
    PytTrain + per-sample timing and fast/slow path tracking.
    
    ใช้แทน PytTrain ใน MinatoLoader โดยตรง — interface เหมือนกันทุกอย่าง
    เพิ่มแค่การ log ลงไฟล์ JSONL
    """
    def __init__(self, *args, sample_trace_path=None, epoch_ref=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace_path  = sample_trace_path
        self._epoch_ref   = epoch_ref or [1]   # mutable reference to current epoch
        self._lock        = threading.Lock()
        self._buf         = []
        self._buf_limit   = 50  # flush every N records

        # timeout_s is set in PytTrain (parent class)
        # Fallback to 2s if attribute not found
        try:
            self._timeout_ms = self.timeout_s * 1000
        except AttributeError:
            self._timeout_ms = 2000.0

    def __getitem__(self, idx):
        t_start = time.perf_counter()
        result  = super().__getitem__(idx)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        if self._trace_path is None:
            return result

        path = "slow" if result is None else "fast"
        record = {
            "epoch":                 self._epoch_ref[0],
            "sample_idx":            int(idx),
            "path":                  path,
            "preprocess_ms":         round(elapsed_ms, 3),
            "timeout_threshold_ms":  self._timeout_ms,
            "timestamp":             time.time(),
        }

        with self._lock:
            self._buf.append(record)
            if len(self._buf) >= self._buf_limit:
                self._flush()

        return result

    def _flush(self):
        """Write buffered records to JSONL (called under lock)."""
        if not self._buf:
            return
        with open(self._trace_path, "a") as f:
            for rec in self._buf:
                f.write(json.dumps(rec) + "\n")
        self._buf.clear()

    def flush(self):
        """Public flush for end-of-epoch calls."""
        with self._lock:
            self._flush()


class QueueMonitor:
    """
    Background thread that polls MinatoLoader queue sizes every N ms.
    
    Usage:
        monitor = QueueMonitor(fast_q, slow_q, log_path, epoch_ref)
        monitor.start()
        # ... training ...
        monitor.stop()
    """
    def __init__(self, fast_queue, slow_processed_queue, log_path,
                 epoch_ref, poll_interval_ms=100):
        self._fast_q    = fast_queue
        self._slow_q    = slow_processed_queue
        self._log_path  = log_path
        self._epoch_ref = epoch_ref
        self._interval  = poll_interval_ms / 1000.0
        self._stop_evt  = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True)
        self._t0        = None

    def start(self):
        self._t0 = time.time()
        self._thread.start()

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=2.0)

    def _qsize_safe(self, q):
        """qsize() can raise NotImplementedError on some platforms."""
        try:
            return q.qsize()
        except Exception:
            return -1

    def _run(self):
        import psutil
        while not self._stop_evt.is_set():
            elapsed_ms = (time.time() - self._t0) * 1000.0
            record = {
                "elapsed_ms":              round(elapsed_ms, 1),
                "epoch":                   self._epoch_ref[0],
                "fast_q_size":             self._qsize_safe(self._fast_q),
                "slow_processed_q_size":   self._qsize_safe(self._slow_q),
                "cpu_percent":             psutil.cpu_percent(interval=None),
                "ram_gb":                  round(
                    psutil.virtual_memory().used / 1e9, 3
                ),
            }
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self._stop_evt.wait(self._interval)
