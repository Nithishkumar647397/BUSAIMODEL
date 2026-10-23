"""
SmartBus TN - AI People Counter (Stage 1)
Science-Expo Prototype: Standalone AI Passenger Counting System

Architecture & Fixes in this version:
  1. Setup Guard: Verifies running inside virtual environment ('venv' in sys.prefix).
  2. Latency-Free Camera: ThreadedCamera reads continuously in background (CAP_PROP_BUFFERSIZE=1),
     preventing inference lag.
  3. Pre-Tracking Detection Filtering: Filters partial detections (MIN_BOX_HEIGHT_FRAC=0.12),
     counts rejections, and displays rejected boxes in grey with reasons during debug mode.
  4. Robust ID Stitching Layer (Configurable via USE_STITCHING):
     - When USE_STITCHING=False: stable_id = raw_id (stitching bypassed for baseline comparison).
     - When USE_STITCHING=True:
       * Rule 1: One stable ID per visible track (no sharing across visible tracks).
       * Rule 2: Candidates are lost tracks lost for <= STITCH_WINDOW_FRAMES.
       * Rule 3: Predicted position = last_pos + velocity * dt within STITCH_MAX_DIST_FRAC of width.
       * Rule 4: Box width and height within STITCH_SIZE_TOLERANCE (+/- 30%).
       * Rule 5: Ambiguity check: rejected if another visible track is closer to predicted position.
       * Rule 6: Crossing check: never stitch across opposite confirmed sides of the line.
       * Rule 7: Choose closest candidate; if none pass, create NEW stable ID.
       * Rule 8: Stable ID stitched to at most one new raw ID at a time.
     - Raw ID reuse fix: raw_to_stable mapping expires after RAW_ID_EXPIRY_FRAMES without detection.
     - Detailed rejection logging: [NO STITCH] id N: <reason>.
  5. Hysteresis Dead-Zone Line Crossing: Dead-zone band (+/- LINE_MARGIN_PX) prevents jitter oscillation.
  6. Horizontal & Vertical Line Orientation: Key 't' toggles orientation, 'w'/'s' moves line.
  7. Ground-Truth Accuracy: Manual 'i'/'o' tallies with evaluation summary upon exit.
"""

import argparse
import csv
import json
import math
import os
import queue
import random
import string
import sys
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
from collections import deque
from datetime import datetime

# ------------------------------------------------------------------------------
# 1. SETUP ROBUSTNESS: Ensure script is running inside the virtual environment
# ------------------------------------------------------------------------------
if "venv" not in sys.prefix.lower():
    print("Run venv\\Scripts\\activate first")
    sys.exit(1)

import cv2
import numpy as np
from ultralytics import YOLO

import config
from status import get_status

# Saved line position per door file
DOOR_LINES_FILE = "door_lines.json"


def load_door_line(door_name: str, default_pos: float) -> float:
    """Loads saved line position for the specified door from door_lines.json."""
    if os.path.exists(DOOR_LINES_FILE):
        try:
            with open(DOOR_LINES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if door_name in data:
                    return float(data[door_name])
        except Exception as e:
            print(f"[WARNING] Could not read {DOOR_LINES_FILE}: {e}")
    return default_pos


def save_door_line(door_name: str, pos: float):
    """Saves updated line position for the specified door to door_lines.json."""
    data = {}
    if os.path.exists(DOOR_LINES_FILE):
        try:
            with open(DOOR_LINES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[door_name] = round(pos, 2)
    try:
        with open(DOOR_LINES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not write to {DOOR_LINES_FILE}: {e}")


# ==============================================================================
# 2. CAMERA: Threaded Frame Capture (Zero Buffer Latency)
# ==============================================================================
class ThreadedCamera:
    """
    Background worker that continuously captures the newest frame from the webcam.
    Ensures that processing delays never cause the AI to evaluate stale/lagged frames.
    """
    def __init__(self, src=0):
        # Open camera with cv2.CAP_DSHOW on Windows for fast and robust access
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            self.is_opened = False
            return

        self.is_opened = True

        capture_w = getattr(config, "CAPTURE_WIDTH", 640)
        capture_h = getattr(config, "CAPTURE_HEIGHT", 480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.running = True
        self.ret = False
        self.frame = None

        # Prime with the very first frame
        self.ret, self.frame = self.cap.read()

        # Launch reader thread
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                with self.lock:
                    self.ret = False
                break
            with self.lock:
                self.ret = True
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return self.ret, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


# ==============================================================================
# LOGGING INITIALIZATION PER DOOR
# ==============================================================================
def get_log_filenames(door: str) -> tuple[str, str]:
    """Returns the CSV file paths specific to this door."""
    return f"crossings_log_{door}.csv", f"ground_truth_log_{door}.csv"


def init_csv_logs(door: str):
    """Initializes CSV logs per door with headers including door column."""
    crossings_file, gt_file = get_log_filenames(door)
    if not os.path.exists(crossings_file):
        with open(crossings_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "door", "track_id", "direction", "in_total", "out_total", "occupancy", "clamped"
            ])

    if not os.path.exists(gt_file):
        with open(gt_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "door", "direction"])


def log_crossing(door: str, stable_id: int, direction: str, in_total: int, out_total: int, occupancy: int, clamped: bool):
    """Logs an AI detected line crossing event for this door."""
    crossings_file, _ = get_log_filenames(door)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(crossings_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, door, stable_id, direction, in_total, out_total, occupancy, clamped])


def log_ground_truth(door: str, direction: str):
    """Logs a manual ground truth event for this door."""
    _, gt_file = get_log_filenames(door)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(gt_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, door, direction])



# ==============================================================================
# REMOTE INGEST TELEMETRY (Stage 3)
# ==============================================================================
# INGEST_KEY is read from the environment variable — never stored in code.
_INGEST_KEY = os.environ.get("INGEST_KEY", "")
_key_warned = False
if not _INGEST_KEY:
    print("[WARNING] INGEST_KEY environment variable is not set. "
          "Events and heartbeat will NOT be sent. Set it with:\n"
          "  Windows: set INGEST_KEY=your_key   (or use run_entrance.bat)\n"
          "  Linux:   export INGEST_KEY=your_key")
    _key_warned = True


def _make_run_id() -> str:
    """Generates a random 8-character alphanumeric run ID at startup."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=8))


def _http_post(url: str, payload: dict, key: str, timeout: float = 0.5):
    """
    Posts payload as JSON with X-Ingest-Key header.
    Returns (status_code, response_dict_or_None).
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Ingest-Key": key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            try:
                return resp.status, json.loads(body)
            except Exception:
                return resp.status, None
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, None
    except Exception:
        return None, None  # network failure


class IngestSender:
    """
    Background worker that drains an event queue with retry logic.
    - Retries the same event_id for up to 30 seconds on network failure.
    - Stops retrying on duplicate / mode_not_ai / overflow replies.
    - Decrements cumulative_boards or cumulative_alights on mode_not_ai/overflow.
    - Shows BAD KEY flag on 401; drops on 422.
    - Also owns the 1-Hz heartbeat thread and the optional frame sender thread.
    """

    def __init__(self, door: str, run_id: str, cumulative_ref: dict):
        """
        cumulative_ref is a shared mutable dict with keys:
          'boards', 'alights', 'camera_ok', 'fps',
          'delivered', 'rejected', 'dropped',
          'server_ok', 'queue_len', 'last_reason', 'bad_key'
        """
        self._door = door
        self._run_id = run_id
        self._cum = cumulative_ref
        self._key = _INGEST_KEY
        self._base = getattr(config, "SERVER_BASE", "http://127.0.0.1:8000")
        self._send = getattr(config, "SEND_EVENTS", True)

        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()

        # Worker thread: drains the event queue
        self._worker_thread = threading.Thread(
            target=self._event_loop, daemon=True, name=f"ingest-events-{door}"
        )
        self._worker_thread.start()

        # Heartbeat thread: sends 1 Hz heartbeat regardless of events
        self._hb_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name=f"ingest-hb-{door}"
        )
        self._hb_thread.start()

        # Optional frame sender thread
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._send_frames = getattr(config, "SEND_FRAMES", False)
        if self._send_frames:
            self._frame_thread = threading.Thread(
                target=self._frame_loop, daemon=True, name=f"ingest-frames-{door}"
            )
            self._frame_thread.start()

    def queue_event(self, event_type: str, track_id: int, event_id: str):
        """Called from the counting loop — never blocks the AI loop."""
        if not self._send or not self._key:
            return
        # Increment cumulative counter when event is queued
        if event_type == "board":
            self._cum["boards"] += 1
        elif event_type == "alight":
            self._cum["alights"] += 1
        self._q.put({
            "event_id": event_id,
            "door": self._door,
            "type": event_type,
            "track_id": track_id,
            "timestamp": time.time(),
        })
        self._cum["queue_len"] = self._q.qsize()

    def push_frame(self, frame_bgr):
        """Called from the main loop; frame is kept only if frame sender is running."""
        if not self._send_frames:
            return
        with self._frame_lock:
            self._latest_frame = frame_bgr

    def stop(self):
        """Signal background threads to stop."""
        self._stop.set()

    # ------------------------------------------------------------------
    # Internal: event worker
    # ------------------------------------------------------------------
    def _event_loop(self):
        url = f"{self._base}/ingest/event"
        while not self._stop.is_set():
            try:
                payload = self._q.get(timeout=0.5)
            except queue.Empty:
                self._cum["queue_len"] = 0
                continue

            self._cum["queue_len"] = self._q.qsize()
            deadline = time.time() + 30.0
            delay = 0.5

            while time.time() < deadline and not self._stop.is_set():
                status, resp = _http_post(url, payload, self._key)

                if status is None:
                    # Network failure — retry with backoff
                    self._cum["server_ok"] = False
                    time.sleep(min(delay, deadline - time.time()))
                    delay = min(delay * 2, 8.0)
                    continue

                self._cum["server_ok"] = (200 <= status < 300 or status == 401)

                if status == 401:
                    self._cum["bad_key"] = True
                    self._cum["last_reason"] = "401 BAD KEY"
                    # Keep retrying within the 30 s window
                    time.sleep(min(delay, deadline - time.time()))
                    delay = min(delay * 2, 8.0)
                    continue

                if status == 422:
                    print(f"[INGEST] 422 Unprocessable for event {payload['event_id']} — dropped.")
                    self._cum["dropped"] += 1
                    self._cum["last_reason"] = "422 bad payload"
                    break

                if resp is None:
                    break

                applied = resp.get("applied", False)
                reason = resp.get("reason", "")
                self._cum["bad_key"] = False
                self._cum["last_reason"] = reason if reason else ("ok" if applied else "?")

                if applied:
                    # delivered (including warnings alight_at_zero / door_type_mismatch)
                    self._cum["delivered"] += 1
                    self._cum["server_ok"] = True
                    break
                else:
                    if reason == "duplicate":
                        # Already delivered; stop retrying, no change to counter
                        self._cum["delivered"] += 1
                        self._cum["server_ok"] = True
                        break
                    elif reason in ("mode_not_ai", "overflow"):
                        # Rejected: decrement matching cumulative counter
                        self._cum["rejected"] += 1
                        evt_type = payload.get("type", "")
                        if evt_type == "board":
                            self._cum["boards"] = max(0, self._cum["boards"] - 1)
                        elif evt_type == "alight":
                            self._cum["alights"] = max(0, self._cum["alights"] - 1)
                        print(f"[INGEST] Event {payload['event_id']} REJECTED: {reason}. Cumulative adjusted.")
                        self._cum["server_ok"] = True
                        break
                    else:
                        # Unknown rejection — retry
                        time.sleep(min(delay, deadline - time.time()))
                        delay = min(delay * 2, 8.0)
                        continue

            else:
                # 30-second deadline expired
                self._cum["dropped"] += 1
                self._cum["last_reason"] = "timeout 30s"
                print(f"[INGEST] Event {payload['event_id']} dropped after 30 s — still counted.")

            self._cum["queue_len"] = self._q.qsize()

    # ------------------------------------------------------------------
    # Internal: 1-Hz heartbeat
    # ------------------------------------------------------------------
    def _heartbeat_loop(self):
        url = f"{self._base}/ingest/heartbeat"
        while not self._stop.is_set():
            payload = {
                "door": self._door,
                "run_id": self._run_id,
                "cumulative_boards": self._cum["boards"],
                "cumulative_alights": self._cum["alights"],
                "fps": round(self._cum["fps"], 1),
                "camera_ok": self._cum["camera_ok"],
                "timestamp": time.time(),
            }
            if self._key:
                try:
                    status, resp = _http_post(url, payload, self._key)
                    if status == 200 and resp:
                        match = resp.get("match")
                        if match is not None and not match:
                            print(f"[INGEST] Heartbeat MISMATCH: server says {resp}")
                        self._cum["server_ok"] = True
                        self._cum["bad_key"] = False
                    elif status == 401:
                        self._cum["bad_key"] = True
                        self._cum["server_ok"] = False
                    elif status is None:
                        self._cum["server_ok"] = False
                except Exception:
                    self._cum["server_ok"] = False
            self._stop.wait(1.0)

    # ------------------------------------------------------------------
    # Internal: frame sender (≤5 fps, no disk write)
    # ------------------------------------------------------------------
    def _frame_loop(self):
        url = f"{self._base}/ingest/frame?door={urllib.parse.quote(self._door)}"
        interval = 1.0 / 5.0  # max 5 fps
        while not self._stop.is_set():
            t0 = time.time()
            if self._key:
                with self._frame_lock:
                    frame = self._latest_frame
                    self._latest_frame = None  # drop if not consumed in time
                if frame is not None:
                    try:
                        small = cv2.resize(frame, (640, 480))
                        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                        if ok:
                            jpeg_bytes = buf.tobytes()
                            req = urllib.request.Request(
                                url,
                                data=jpeg_bytes,
                                headers={
                                    "Content-Type": "image/jpeg",
                                    "X-Ingest-Key": self._key,
                                },
                                method="POST",
                            )
                            try:
                                urllib.request.urlopen(req, timeout=0.5)
                            except Exception:
                                pass  # drop frame silently
                    except Exception:
                        pass
            elapsed = time.time() - t0
            self._stop.wait(max(0.0, interval - elapsed))


# ==============================================================================
# VISUAL HUD & OVERLAYS
# ==============================================================================
def draw_hud(frame, door: str, door_label: str, count_direction: str,
             in_total: int, out_total: int, ignored_total: int,
             occupancy: int, capacity: int,
             fps: float, line_orientation: str, line_ratio: float, in_direction: str,
             raw_ids_created: int, stable_ids_created: int, stitches_done: int,
             rej_too_far: int, rej_size: int, rej_ambig: int, rej_side: int, rej_in_use: int,
             rejected_short: int, rejected_wide: int,
             gt_in: int, gt_out: int, debug_mode: bool, paused: bool,
             ingest_cum: dict | None = None):
    """
    Renders top status banner, diagnostic metrics, and control prompts onto frame.
    Adjusted per door mode (shows IN for entrance, OUT for exit, and ignored count).
    """
    h, w = frame.shape[:2]

    # Calculate status from single source of truth in status.py
    status, percent = get_status(occupancy, capacity)
    if status == "GREEN":
        status_color = (60, 210, 60)
    elif status == "YELLOW":
        status_color = (0, 215, 255)
    elif status == "RED":
        status_color = (40, 40, 240)
    else:  # OVERFLOW
        status_color = (180, 0, 220)

    # Top banner background (semi-transparent dark)
    banner_h = 78 if not debug_mode else 118
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (18, 18, 22), -1)
    # Bottom key guide bar
    cv2.rectangle(overlay, (0, h - 28), (w, h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    pt_name = getattr(config, "TRACK_POINT", "center").upper()

    # Line 1: Primary Passenger Metrics per door
    if door == "entrance":
        cv2.putText(frame, "DOOR:", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "ENTRANCE", (70, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.putText(frame, "IN:", (195, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        cv2.putText(frame, f"{in_total}", (235, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 230, 80), 2)

        cv2.putText(frame, f"ignored: {ignored_total}", (310, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (140, 140, 140), 1)

        gt_text = f"Real [I]: {gt_in} IN   Point: {pt_name}   ({door_label})"
        cv2.putText(frame, gt_text, (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 120), 1)

    elif door == "exit":
        cv2.putText(frame, "DOOR:", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "EXIT", (70, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)

        cv2.putText(frame, "OUT:", (155, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        cv2.putText(frame, f"{out_total}", (205, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 120, 255), 2)

        cv2.putText(frame, f"ignored: {ignored_total}", (280, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (140, 140, 140), 1)

        gt_text = f"Real [O]: {gt_out} OUT   Point: {pt_name}   ({door_label})"
        cv2.putText(frame, gt_text, (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 120), 1)

    else:  # "both" (single-door test mode)
        cv2.putText(frame, "IN:", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        cv2.putText(frame, f"{in_total}", (55, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 230, 80), 2)

        cv2.putText(frame, "OUT:", (125, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        cv2.putText(frame, f"{out_total}", (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 120, 255), 2)

        cv2.putText(frame, "ON BOARD:", (255, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
        occ_text = f"{occupancy} / {capacity}"
        cv2.putText(frame, occ_text, (360, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        status_badge = f"[{status}] {percent:.0f}%"
        cv2.putText(frame, status_badge, (255, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.putText(frame, f"ignored: {ignored_total}", (470, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)

        gt_text = f"Real [I/O]: {gt_in} IN | {gt_out} OUT   Point: {pt_name}"
        cv2.putText(frame, gt_text, (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 120), 1)

    # Top right: FPS and line info
    fps_text = f"FPS: {fps:4.1f}"
    cv2.putText(frame, fps_text, (w - 140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    line_info = f"Line: {line_orientation[:4].upper()} {int(line_ratio * 100)}% ({in_direction.upper()}=IN)"
    cv2.putText(frame, line_info, (w - 330, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    # Line 3, 4 & 5 (if debug mode active): Tracking Stability, Stitch Rejections, Filter Rejections
    if debug_mode:
        inflation = raw_ids_created / max(1, stable_ids_created)
        diag_text = (f"DEBUG | Raw IDs: {raw_ids_created}  Stable IDs: {stable_ids_created}  "
                     f"Stitches: {stitches_done}  ID Inflation: {inflation:.2f}x")
        cv2.putText(frame, diag_text, (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

        if config.USE_STITCHING:
            st_rej_text = (f"STITCH REJ: far={rej_too_far} size={rej_size} ambig={rej_ambig} "
                           f"side={rej_side} in_use={rej_in_use}")
            cv2.putText(frame, st_rej_text, (15, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 180, 100), 1)
        else:
            cv2.putText(frame, "STITCHING DISABLED: stable_id == raw_id", (15, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)

        rej_text = f"DET REJECTED: short={rejected_short} wide={rejected_wide} (AspectFilter={'ON' if config.USE_ASPECT_FILTER else 'OFF'})"
        cv2.putText(frame, rej_text, (15, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 200, 255), 1)

    # Ingest status bar (always visible, one line above the bottom guide)
    if ingest_cum is not None and getattr(config, "SEND_EVENTS", True):
        server_state = "OK" if ingest_cum.get("server_ok") else "DOWN"
        srv_color = (60, 220, 60) if ingest_cum.get("server_ok") else (40, 40, 230)
        bad_key = ingest_cum.get("bad_key", False)
        last_reason = ingest_cum.get("last_reason", "")
        ingest_text = (
            f"SRV:{server_state}  Q:{ingest_cum.get('queue_len', 0)} "
            f"D:{ingest_cum.get('delivered', 0)} "
            f"REJ:{ingest_cum.get('rejected', 0)} "
            f"DROP:{ingest_cum.get('dropped', 0)} "
            f"last:{last_reason}"
        )
        cv2.putText(frame, ingest_text, (15, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, srv_color, 1)
        if bad_key:
            cv2.putText(frame, "BAD KEY", (w - 110, h - 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (40, 40, 230), 2)

    # Bottom Guide Text
    debug_state = "ON" if debug_mode else "OFF"
    if door == "entrance":
        guide_text = f"[Q] Quit  [T] Toggle Line  [W/S] Move Line (saved)  [I] Real IN  [D] Debug ({debug_state})  [R] Reset  [SPACE] Pause"
    elif door == "exit":
        guide_text = f"[Q] Quit  [T] Toggle Line  [W/S] Move Line (saved)  [O] Real OUT  [D] Debug ({debug_state})  [R] Reset  [SPACE] Pause"
    else:
        guide_text = f"[Q] Quit  [T] Toggle Line  [W/S] Move Line (saved)  [I/O] Real  [D] Debug ({debug_state})  [R] Reset  [SPACE] Pause"
    cv2.putText(frame, guide_text, (15, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 215, 215), 1)

    # Pause Banner
    if paused:
        p_overlay = frame.copy()
        cv2.rectangle(p_overlay, (w // 2 - 170, h // 2 - 35), (w // 2 + 170, h // 2 + 35), (0, 0, 0), -1)
        cv2.addWeighted(p_overlay, 0.70, frame, 0.30, 0, frame)
        cv2.putText(frame, "PAUSED - PRESS SPACE", (w // 2 - 150, h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)


# ==============================================================================
# ACCURACY EVALUATION SUMMARY
# ==============================================================================
def print_summary(door: str, door_label: str, count_direction: str,
                  gt_in: int, gt_out: int, in_total: int, out_total: int, ignored_total: int,
                  raw_ids_created: int, stable_ids_created: int, stitches_done: int):
    """Prints a structured accuracy comparison table between ground-truth and AI counts upon exit."""
    inflation = raw_ids_created / max(1, stable_ids_created)

    print("\n" + "=" * 76)
    print(f" SmartBus TN - Evaluation Summary: {door_label}")
    print("=" * 76)

    if count_direction == "IN":  # ENTRANCE
        counted = in_total
        real = gt_in
        print(f" Door Name:                 {door} ({door_label})")
        print(f" Monitored Direction:       IN only")
        print(f" AI Counted Total:          {counted}")
        print(f" Ignored (OUT) Total:       {ignored_total}")
        print(f" Real Ground Truth Total:   {real}")
        if real == 0:
            print(" Missed:                    -")
            print(f" Extra:                     {counted}")
            print(" Accuracy:                  No ground truth recorded")
        else:
            missed = max(0, real - counted)
            extra = max(0, counted - real)
            acc = max(0.0, 100.0 * (1.0 - abs(counted - real) / real))
            print(f" Missed:                    {missed}")
            print(f" Extra:                     {extra}")
            print(f" Accuracy:                  {acc:.1f}%")

    elif count_direction == "OUT":  # EXIT
        counted = out_total
        real = gt_out
        print(f" Door Name:                 {door} ({door_label})")
        print(f" Monitored Direction:       OUT only")
        print(f" AI Counted Total:          {counted}")
        print(f" Ignored (IN) Total:        {ignored_total}")
        print(f" Real Ground Truth Total:   {real}")
        if real == 0:
            print(" Missed:                    -")
            print(f" Extra:                     {counted}")
            print(" Accuracy:                  No ground truth recorded")
        else:
            missed = max(0, real - counted)
            extra = max(0, counted - real)
            acc = max(0.0, 100.0 * (1.0 - abs(counted - real) / real))
            print(f" Missed:                    {missed}")
            print(f" Extra:                     {extra}")
            print(f" Accuracy:                  {acc:.1f}%")

    else:  # BOTH (single-door test)
        tot_real = gt_in + gt_out
        tot_counted = in_total + out_total
        print(f" Door Name:                 {door} ({door_label})")
        print(f" Monitored Direction:       IN and OUT")
        print(f" Ignored Total:             {ignored_total}")
        print(" " + "-" * 74)
        if tot_real == 0:
            print(f" {'Direction':<10} {'Real (Tally)':<14} {'AI Counted':<12} {'Missed':<8} {'Extra':<8} {'Accuracy':<26}")
            print(" " + "-" * 74)
            print(f" {'IN':<10} {gt_in:<14} {in_total:<12} {'-':<8} {in_total:<8} {'No ground truth recorded':<26}")
            print(f" {'OUT':<10} {gt_out:<14} {out_total:<12} {'-':<8} {out_total:<8} {'No ground truth recorded':<26}")
            print(" " + "-" * 74)
            print(f" {'TOTAL':<10} {tot_real:<14} {tot_counted:<12} {'-':<8} {tot_counted:<8} {'No ground truth recorded':<26}")
        else:
            def calc_acc(r, c):
                if r == 0:
                    return 100.0 if c == 0 else 0.0
                return max(0.0, 100.0 * (1.0 - abs(c - r) / r))

            in_missed = max(0, gt_in - in_total)
            in_extra = max(0, in_total - gt_in)
            in_acc = calc_acc(gt_in, in_total)

            out_missed = max(0, gt_out - out_total)
            out_extra = max(0, out_total - gt_out)
            out_acc = calc_acc(gt_out, out_total)

            tot_missed = max(0, tot_real - tot_counted)
            tot_extra = max(0, tot_counted - tot_real)
            tot_acc = calc_acc(tot_real, tot_counted)

            print(f" {'Direction':<10} {'Real (Tally)':<14} {'AI Counted':<12} {'Missed':<8} {'Extra':<8} {'Accuracy':<10}")
            print(" " + "-" * 74)
            print(f" {'IN':<10} {gt_in:<14} {in_total:<12} {in_missed:<8} {in_extra:<8} {in_acc:>6.1f}%")
            print(f" {'OUT':<10} {gt_out:<14} {out_total:<12} {out_missed:<8} {out_extra:<8} {out_acc:>6.1f}%")
            print(" " + "-" * 74)
            print(f" {'TOTAL':<10} {tot_real:<14} {tot_counted:<12} {tot_missed:<8} {tot_extra:<8} {tot_acc:>6.1f}%")

    print("=" * 76)
    print(" Tracking Diagnostics:")
    print(f"   Stitching Mode:             {'ENABLED' if config.USE_STITCHING else 'DISABLED'}")
    print(f"   Raw ByteTrack IDs Created:  {raw_ids_created}")
    print(f"   Stable Person IDs Created:  {stable_ids_created}")
    print(f"   Stitches Performed:         {stitches_done}")
    print(f"   ID Inflation Ratio:         {inflation:.2f}x (Raw IDs / Stable IDs)")
    print("=" * 76 + "\n")


# ==============================================================================
# MAIN APPLICATION LOOP
# ==============================================================================
def parse_arguments():
    """Parses command-line arguments for per-door operation."""
    parser = argparse.ArgumentParser(description="SmartBus TN - AI People Counter (Door Counter)")
    parser.add_argument("--door", choices=["entrance", "exit", "both"], required=True,
                        help="Door to monitor: 'entrance' (counts IN only), 'exit' (counts OUT only), or 'both' (testing)")
    parser.add_argument("--camera", type=int, default=getattr(config, "CAMERA_INDEX", 0),
                        help="Camera device index (default: 0)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    door = args.door.lower()

    # Retrieve per-door configuration from config.DOOR_SETTINGS
    door_settings = getattr(config, "DOOR_SETTINGS", {})
    door_cfg = door_settings.get(door, {
        "count_direction": "BOTH",
        "line_mode": "horizontal",
        "line_pos": 0.5,
        "in_direction": "down",
        "label": f"DOOR - {door.upper()}",
    })

    door_label = door_cfg.get("label", f"DOOR - {door.upper()}")
    count_direction = door_cfg.get("count_direction", "BOTH")
    line_orientation = door_cfg.get("line_mode", "horizontal")
    line_ratio = float(door_cfg.get("line_pos", 0.5))
    in_direction = door_cfg.get("in_direction", "down")

    # Load saved line position from door_lines.json (remembers line per door)
    line_ratio = load_door_line(door, line_ratio)

    # Initialize CSV logs for this specific door
    init_csv_logs(door)

    # Operational passenger counters
    in_total = 0
    out_total = 0
    ignored_total = 0
    occupancy = 0

    paused = False
    debug_mode = True

    # Ground-truth tallies
    gt_in = 0
    gt_out = 0

    # Detection filter diagnostics counters
    rejected_short = 0
    rejected_wide = 0

    # Stitching rejection diagnostic counters
    rej_too_far = 0
    rej_size = 0
    rej_ambig = 0
    rej_side = 0
    rej_in_use = 0

    # Startup Banner (shows Door, Direction, Line, Stitching)
    print("=" * 76)
    print(f" SmartBus TN - Standalone AI People Counter ({door_label})")
    print(f" Door Mode: {door.upper()} | Counts: {count_direction} ONLY" if count_direction != "BOTH" else f" Door Mode: {door.upper()} | Counts: IN and OUT")
    print(f" Model: {config.MODEL} | ImgSz: {config.IMGSZ} | Conf: {config.CONF}")
    print(f" Capacity: {config.CAPACITY} | Line: {line_orientation.upper()} at {int(line_ratio * 100)}% ({in_direction.upper()}=IN)")
    print(f" Stitching: {'ENABLED' if config.USE_STITCHING else 'DISABLED (raw_id mode)'} | "
          f"Window: {config.STITCH_WINDOW_FRAMES}f | DistFrac: {config.STITCH_MAX_DIST_FRAC}")
    print(f" Point: {getattr(config, 'TRACK_POINT', 'center').upper()} | Dead-zone Margin: +/- {config.LINE_MARGIN_PX}px")
    print(f" Logs: crossings_log_{door}.csv | ground_truth_log_{door}.csv")
    print("=" * 76)

    # Initialize Threaded Camera
    camera_idx = args.camera
    print(f"[INFO] Initializing Camera (Index {camera_idx}) for {door.upper()} door...")
    cam = ThreadedCamera(camera_idx)
    if not cam.is_opened:
        print("=" * 76)
        print(f"[ERROR] Could not open camera with index {camera_idx}.")
        print("Troubleshooting:")
        print("  1. Verify your USB camera or laptop webcam is connected.")
        print("  2. Ensure no other application (Camera app, Zoom, Teams) is using it.")
        print("  3. Run 'python list_cameras.py' to discover available camera indexes.")
        print("=" * 76)
        sys.exit(1)

    # Load YOLO Model
    print(f"[INFO] Loading YOLO model '{config.MODEL}'...")
    try:
        model = YOLO(config.MODEL)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        cam.release()
        sys.exit(1)

    # --------------------------------------------------------------------------
    # ID STITCHING & TRACKING STATE
    # --------------------------------------------------------------------------
    raw_to_stable: dict[int, int] = {}
    raw_last_seen: dict[int, int] = {}  # raw_id -> last_frame_idx seen
    next_stable_id: int = 1

    # Diagnostics counters
    raw_ids_created: int = 0
    stable_ids_created: int = 0
    stitches_done: int = 0

    # Retained lost tracks for stitching:
    # key: stable_id -> dict(last_pos, velocity, box_w, box_h, lost_frame, last_raw_id, last_confirmed_side)
    lost_tracks: dict[int, dict] = {}

    # State per stable_id
    stable_frame_count: dict[int, int] = {}          # Total frames observed
    last_confirmed_side: dict[int, str] = {}         # "ABOVE", "BELOW", "LEFT", "RIGHT", or None
    last_crossing_frame: dict[int, int] = {}         # Frame index of last crossing
    track_last_pos: dict[int, tuple[int, int]] = {}     # Previous (cx, cy)
    track_last_box: dict[int, tuple[float, float]] = {} # Previous (box_w, box_h)
    track_velocity: dict[int, tuple[float, float]] = {} # Smoothed (vx, vy)
    trails: dict[int, deque] = {}                    # Deque of last 20 foot points
    last_no_cross_report: dict[int, tuple[int, str]] = {}  # Throttle for [NO CROSS] prints

    # FPS smoothing
    frame_times = deque(maxlen=30)
    prev_time = time.time()
    fps = 0.0
    frame_idx = 0
    last_frame_time = time.time()  # for camera_ok heartbeat field

    # ---- Ingest sender setup ----
    run_id = _make_run_id()
    # Local event ID counter — starts at 1, never resets between r-resets
    event_local_counter = [0]  # mutable list so inner function can write

    # Shared mutable state dict read by IngestSender and draw_hud
    ingest_cum = {
        "boards": 0,
        "alights": 0,
        "fps": 0.0,
        "camera_ok": True,
        "delivered": 0,
        "rejected": 0,
        "dropped": 0,
        "server_ok": False,
        "queue_len": 0,
        "last_reason": "",
        "bad_key": False,
    }

    sender = IngestSender(door, run_id, ingest_cum)

    def queue_event(event_type: str, track_id: int):
        """Generates a unique event_id and queues it. Called on each counted crossing."""
        event_local_counter[0] += 1
        eid = f"{door}-{int(time.time())}-{event_local_counter[0]}"
        sender.queue_event(event_type, track_id, eid)

    print(f"[INGEST] run_id={run_id} | server={getattr(config, 'SERVER_BASE', '-')} "
          f"| key={'set' if _INGEST_KEY else 'MISSING'}")
    print("[INFO] Camera stream started. Press 'q' to quit, 't' to toggle H/V line, 'd' for debug overlay.")
    print("-" * 76)

    try:
        while True:
            if not paused:
                ret, frame = cam.read()
                if not ret or frame is None:
                    print("\n[ERROR] Failed to read frame from camera. Camera disconnected.")
                    break

                frame_idx += 1
                h, w = frame.shape[:2]

                # Smoothed FPS calculation (average of last 30 frames)
                curr_time = time.time()
                frame_times.append(curr_time - prev_time)
                prev_time = curr_time
                avg_time = sum(frame_times) / max(1, len(frame_times))
                fps = (1.0 / avg_time) if avg_time > 0 else 0.0

                # Update shared ingest state for heartbeat
                last_frame_time = curr_time
                ingest_cum["fps"] = fps
                ingest_cum["camera_ok"] = True

                # Push current frame to frame sender (no-op if SEND_FRAMES=False)
                sender.push_frame(frame)

                # --------------------------------------------------------------
                # RUN YOLO TRACKING WITH BYTETRACK
                # --------------------------------------------------------------
                results = model.track(
                    source=frame,
                    persist=True,
                    classes=[0],             # Person class only
                    conf=config.CONF,
                    imgsz=config.IMGSZ,
                    tracker=config.TRACKER,  # "bytetrack_custom.yaml"
                    verbose=False
                )

                # --------------------------------------------------------------
                # DETECTION FILTERING & DIAGNOSTICS
                # --------------------------------------------------------------
                valid_detections = []   # list of (raw_id, box, conf, is_clipped, cx, cy)
                rejected_boxes = []     # list of (box, reason)

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    if boxes.id is not None:
                        track_ids = boxes.id.int().cpu().tolist()
                        xyxys = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(track_ids)

                        for raw_id, box, conf in zip(track_ids, xyxys, confs):
                            x1, y1, x2, y2 = box
                            box_w = x2 - x1
                            box_h = y2 - y1

                            # Filter 1: Check minimum box height (12% of frame height)
                            if box_h < (h * config.MIN_BOX_HEIGHT_FRAC):
                                rejected_short += 1
                                rejected_boxes.append((box, f"short ({box_h/h:.2f}<{config.MIN_BOX_HEIGHT_FRAC})"))
                                continue

                            # Filter 2: Optional aspect ratio filter (disabled by default)
                            if config.USE_ASPECT_FILTER and ((box_w / max(1.0, box_h)) > config.MAX_ASPECT_RATIO):
                                rejected_wide += 1
                                rejected_boxes.append((box, f"wide ({box_w/box_h:.1f}>{config.MAX_ASPECT_RATIO})"))
                                continue

                            # Track Point Calculation based on config.TRACK_POINT
                            # options: "center" (box center), "chest" (30% down from top), "feet" (bottom-center)
                            track_mode = getattr(config, "TRACK_POINT", "center").lower()
                            cx = int((x1 + x2) / 2)
                            is_clipped = False

                            if track_mode == "center":
                                cy = int((y1 + y2) / 2)
                            elif track_mode == "chest":
                                cy = int(y1 + 0.30 * box_h)
                            else:  # "feet"
                                is_clipped = (y2 >= (h - 3))
                                if is_clipped:
                                    cy = int(y2 - 0.15 * box_h)
                                else:
                                    cy = int(y2)

                            valid_detections.append((raw_id, box, conf, is_clipped, cx, cy))

                # --------------------------------------------------------------
                # RAW ID REUSE EXPIRY & LOST TRACK PRUNING
                # --------------------------------------------------------------
                # If a raw ID has not been seen for > RAW_ID_EXPIRY_FRAMES, delete its mapping
                expired_raws = [
                    r for r, last_f in raw_last_seen.items()
                    if (frame_idx - last_f) > config.RAW_ID_EXPIRY_FRAMES
                ]
                for r in expired_raws:
                    if r in raw_to_stable:
                        del raw_to_stable[r]
                    del raw_last_seen[r]

                # Delete lost-track memory older than STITCH_WINDOW_FRAMES
                expired_lost = [
                    s for s, item in lost_tracks.items()
                    if (frame_idx - item["lost_frame"]) > config.STITCH_WINDOW_FRAMES
                ]
                for s in expired_lost:
                    del lost_tracks[s]

                # Current visible raw IDs in this frame
                current_raw_ids = set(d[0] for d in valid_detections)

                # Check which mapped raw IDs disappeared this frame and move their stable IDs to lost_tracks
                for r_id in list(raw_to_stable.keys()):
                    if r_id not in current_raw_ids:
                        s_id = raw_to_stable[r_id]
                        # Only add to lost_tracks if this stable_id is not already active
                        if s_id not in lost_tracks:
                            lost_tracks[s_id] = {
                                "last_pos": track_last_pos.get(s_id, (0, 0)),
                                "velocity": track_velocity.get(s_id, (0.0, 0.0)),
                                "box_w": track_last_box.get(s_id, (0.0, 0.0))[0],
                                "box_h": track_last_box.get(s_id, (0.0, 0.0))[1],
                                "lost_frame": frame_idx,
                                "last_raw_id": r_id,
                                "last_confirmed_side": last_confirmed_side.get(s_id, "UNKNOWN")
                            }

                # --------------------------------------------------------------
                # ID STITCHING & ASSIGNMENT FOR CURRENT FRAME
                # --------------------------------------------------------------
                # Track stable IDs assigned to visible tracks in this frame
                # (Rule 1 & Rule 8: One stable ID per visible track)
                assigned_sids_this_frame = set()
                processed_tracks = [] # (stable_id, raw_id, box, conf, is_clipped, cx, cy)

                # Compute line coordinates for side checks
                is_horiz = (line_orientation == "horizontal")
                line_pos = int((h if is_horiz else w) * line_ratio)
                margin = config.LINE_MARGIN_PX
                dead_zone_min = line_pos - margin
                dead_zone_max = line_pos + margin
                stitch_max_dist = config.STITCH_MAX_DIST_FRAC * w

                for raw_id, box, conf, is_clipped, cx, cy in valid_detections:
                    new_w = box[2] - box[0]
                    new_h = box[3] - box[1]

                    # Determine side for crossing check (Rule 6)
                    track_coord = cy if is_horiz else cx
                    if is_horiz:
                        new_side = "ABOVE" if track_coord < dead_zone_min else ("BELOW" if track_coord > dead_zone_max else "UNKNOWN")
                    else:
                        new_side = "LEFT" if track_coord < dead_zone_min else ("RIGHT" if track_coord > dead_zone_max else "UNKNOWN")

                    now_str = datetime.now().strftime("%H:%M:%S")

                    if not config.USE_STITCHING:
                        # ------------------------------------------------------
                        # BASELINE MODE: Stitching disabled (stable_id = raw_id)
                        # ------------------------------------------------------
                        stable_id = raw_id
                        if raw_id not in raw_to_stable:
                            raw_ids_created += 1
                            stable_ids_created += 1
                            raw_to_stable[raw_id] = stable_id
                        raw_last_seen[raw_id] = frame_idx
                        assigned_sids_this_frame.add(stable_id)

                    else:
                        # ------------------------------------------------------
                        # STITCHING MODE (USE_STITCHING = True)
                        # ------------------------------------------------------
                        # Check if raw_id is already in raw_to_stable and valid
                        if raw_id in raw_to_stable:
                            candidate_sid = raw_to_stable[raw_id]
                            # Rule 1: Never give a stable ID if already assigned to another visible track
                            if candidate_sid not in assigned_sids_this_frame:
                                stable_id = candidate_sid
                                assigned_sids_this_frame.add(stable_id)
                                raw_last_seen[raw_id] = frame_idx
                                if stable_id in lost_tracks:
                                    del lost_tracks[stable_id]
                            else:
                                # Conflict: raw ID mapped to an already visible stable ID
                                # Treat as a NEW raw ID to resolve conflict
                                del raw_to_stable[raw_id]
                                stable_id = None
                        else:
                            stable_id = None

                        if stable_id is None:
                            # This is a NEW raw ID! Apply Rules 1 to 8
                            raw_ids_created += 1
                            raw_last_seen[raw_id] = frame_idx

                            passed_candidates = []

                            for cand_sid, info in list(lost_tracks.items()):
                                # Rule 1 & Rule 8: Never stitch to an ID already assigned this frame
                                if cand_sid in assigned_sids_this_frame:
                                    rej_in_use += 1
                                    print(f"[{now_str}] [NO STITCH] id #{cand_sid}: already in use")
                                    continue

                                # Rule 2: Lost for at most STITCH_WINDOW_FRAMES
                                dt = frame_idx - info["lost_frame"]
                                if dt > config.STITCH_WINDOW_FRAMES:
                                    continue

                                # Rule 3: Predicted position within STITCH_MAX_DIST
                                pred_x = info["last_pos"][0] + info["velocity"][0] * dt
                                pred_y = info["last_pos"][1] + info["velocity"][1] * dt
                                dist = math.hypot(cx - pred_x, cy - pred_y)
                                if dist > stitch_max_dist:
                                    rej_too_far += 1
                                    print(f"[{now_str}] [NO STITCH] id #{cand_sid}: too far ({dist:.1f}px > {stitch_max_dist:.1f}px)")
                                    continue

                                # Rule 4: Size check (width and height within STITCH_SIZE_TOLERANCE)
                                old_w = info["box_w"]
                                old_h = info["box_h"]
                                tol = config.STITCH_SIZE_TOLERANCE
                                w_diff = abs(new_w - old_w) / max(1.0, old_w)
                                h_diff = abs(new_h - old_h) / max(1.0, old_h)
                                if (w_diff > tol) or (h_diff > tol):
                                    rej_size += 1
                                    print(f"[{now_str}] [NO STITCH] id #{cand_sid}: size mismatch (w_diff={w_diff:.2f}, h_diff={h_diff:.2f} > {tol})")
                                    continue

                                # Rule 5: Ambiguity check (no other visible track is closer to predicted position)
                                is_ambiguous = False
                                for other_det in valid_detections:
                                    if other_det[0] == raw_id:
                                        continue
                                    other_dist = math.hypot(other_det[4] - pred_x, other_det[5] - pred_y)
                                    if other_dist < dist:
                                        is_ambiguous = True
                                        break
                                if is_ambiguous:
                                    rej_ambig += 1
                                    print(f"[{now_str}] [NO STITCH] id #{cand_sid}: ambiguous (another visible track is closer)")
                                    continue

                                # Rule 6: Crossing check (new track confirmed side must not be opposite of lost side)
                                lost_side = info.get("last_confirmed_side", "UNKNOWN")
                                is_opposite = (
                                    (lost_side == "ABOVE" and new_side == "BELOW") or
                                    (lost_side == "BELOW" and new_side == "ABOVE") or
                                    (lost_side == "LEFT" and new_side == "RIGHT") or
                                    (lost_side == "RIGHT" and new_side == "LEFT")
                                )
                                if is_opposite:
                                    rej_side += 1
                                    print(f"[{now_str}] [NO STITCH] id #{cand_sid}: opposite side ({lost_side} -> {new_side})")
                                    continue

                                # Candidate passed all checks!
                                passed_candidates.append((cand_sid, dist, dt, info))

                            # Rule 7: Choose closest passing candidate; if none pass, create NEW stable ID
                            if passed_candidates:
                                # Closest candidate
                                best_sid, best_dist, best_dt, best_info = min(passed_candidates, key=lambda c: c[1])
                                stable_id = best_sid
                                raw_to_stable[raw_id] = stable_id
                                assigned_sids_this_frame.add(stable_id)
                                del lost_tracks[stable_id]
                                stitches_done += 1

                                prev_raw = best_info["last_raw_id"]
                                size_ratio = new_h / max(1.0, best_info["box_h"])

                                # Only print [STITCH] when raw ID actually changed
                                if raw_id != prev_raw:
                                    print(f"[{now_str}] [STITCH] Raw ID #{raw_id} -> Stitched to Stable ID #{stable_id} "
                                          f"(old raw #{prev_raw}, dist: {best_dist:.1f}px, lost: {best_dt}f, size_ratio: {size_ratio:.2f})")
                            else:
                                stable_id = next_stable_id
                                next_stable_id += 1
                                raw_to_stable[raw_id] = stable_id
                                assigned_sids_this_frame.add(stable_id)
                                stable_ids_created += 1
                                print(f"[{now_str}] [NEW TRACK] Raw ID #{raw_id} -> New Stable ID #{stable_id} (conf: {conf:.2f})")

                    # Record processed track
                    processed_tracks.append((stable_id, raw_id, box, conf, is_clipped, cx, cy, new_w, new_h))

                # --------------------------------------------------------------
                # UPDATE STATE & EVALUATE CROSSINGS FOR PROCESSED TRACKS
                # --------------------------------------------------------------
                span_max_bound = w if is_horiz else h
                span_min_px = int(span_max_bound * config.LINE_SPAN_MIN)
                span_max_px = int(span_max_bound * config.LINE_SPAN_MAX)

                for stable_id, raw_id, box, conf, is_clipped, cx, cy, new_w, new_h in processed_tracks:
                    # Update position, box size, and velocity
                    if stable_id in track_last_pos:
                        prev_cx, prev_cy = track_last_pos[stable_id]
                        inst_vx = cx - prev_cx
                        inst_vy = cy - prev_cy
                        old_vx, old_vy = track_velocity.get(stable_id, (0.0, 0.0))
                        track_velocity[stable_id] = (0.7 * old_vx + 0.3 * inst_vx, 0.7 * old_vy + 0.3 * inst_vy)
                    else:
                        track_velocity[stable_id] = (0.0, 0.0)

                    track_last_pos[stable_id] = (cx, cy)
                    track_last_box[stable_id] = (new_w, new_h)
                    stable_frame_count[stable_id] = stable_frame_count.get(stable_id, 0) + 1

                    # Update foot trail (last 20 points)
                    if stable_id not in trails:
                        trails[stable_id] = deque(maxlen=20)
                    trails[stable_id].append((cx, cy))

                    # ----------------------------------------------------------
                    # CROSSING LOGIC WITH HYSTERESIS DEAD ZONE (H or V)
                    # ----------------------------------------------------------
                    track_coord = cy if is_horiz else cx
                    span_coord = cx if is_horiz else cy

                    # Determine side
                    if is_horiz:
                        if track_coord < dead_zone_min:
                            current_side = "ABOVE"
                        elif track_coord > dead_zone_max:
                            current_side = "BELOW"
                        else:
                            current_side = "UNKNOWN"
                    else:
                        if track_coord < dead_zone_min:
                            current_side = "LEFT"
                        elif track_coord > dead_zone_max:
                            current_side = "RIGHT"
                        else:
                            current_side = "UNKNOWN"

                    # Check for crossing attempt / rejection
                    if stable_id in last_confirmed_side:
                        prev_side = last_confirmed_side[stable_id]
                        moved_across_line = (current_side != "UNKNOWN" and current_side != prev_side)
                        inside_dead_zone = (current_side == "UNKNOWN")

                        # If inside dead zone but foot physically crossed center line
                        if inside_dead_zone:
                            if (prev_side in ("ABOVE", "LEFT") and track_coord > line_pos) or \
                               (prev_side in ("BELOW", "RIGHT") and track_coord < line_pos):
                                reason = f"inside dead zone (+/-{margin}px around line)"
                                last_rep_f, last_rep_r = last_no_cross_report.get(stable_id, (-999, ""))
                                if (frame_idx - last_rep_f) >= 30 or reason != last_rep_r:
                                    last_no_cross_report[stable_id] = (frame_idx, reason)
                                    now_str = datetime.now().strftime("%H:%M:%S")
                                    print(f"[{now_str}] [NO CROSS] id #{stable_id}: {reason}")

                        elif moved_across_line:
                            inside_span = (span_min_px <= span_coord <= span_max_px)
                            frames_seen = stable_frame_count.get(stable_id, 0)
                            frames_since_crossing = frame_idx - last_crossing_frame.get(stable_id, -99999)

                            if not inside_span:
                                reason = f"outside line span ({span_coord} not in [{span_min_px}..{span_max_px}])"
                                now_str = datetime.now().strftime("%H:%M:%S")
                                print(f"[{now_str}] [NO CROSS] id #{stable_id}: {reason}")
                            elif frames_seen < config.MIN_TRACK_FRAMES:
                                reason = f"too few frames ({frames_seen} < {config.MIN_TRACK_FRAMES})"
                                now_str = datetime.now().strftime("%H:%M:%S")
                                print(f"[{now_str}] [NO CROSS] id #{stable_id}: {reason}")
                            elif frames_since_crossing < config.COOLDOWN_FRAMES:
                                cd_left = config.COOLDOWN_FRAMES - frames_since_crossing
                                reason = f"cooldown active ({cd_left} frames left)"
                                now_str = datetime.now().strftime("%H:%M:%S")
                                print(f"[{now_str}] [NO CROSS] id #{stable_id}: {reason}")
                            else:
                                # ALL CONDITIONS PASSED: REGISTER CROSSING!
                                if is_horiz:
                                    if prev_side == "ABOVE" and current_side == "BELOW":
                                        direction = "IN" if in_direction == "down" else "OUT"
                                    else:
                                        direction = "OUT" if in_direction == "down" else "IN"
                                else:
                                    if prev_side == "LEFT" and current_side == "RIGHT":
                                        direction = "IN" if in_direction == "right" else "OUT"
                                    else:
                                        direction = "OUT" if in_direction == "right" else "IN"

                                # Check per-door direction filter:
                                # The entrance counts only crossings whose direction is IN.
                                # The exit counts only crossings whose direction is OUT.
                                # Crossings in the other direction are ignored.
                                if count_direction != "BOTH" and direction != count_direction:
                                    ignored_total += 1
                                    last_crossing_frame[stable_id] = frame_idx
                                    now_str = datetime.now().strftime("%H:%M:%S")
                                    print(f"[{now_str}] [IGNORED] door {door}: {direction} crossing (Stable ID #{stable_id})")
                                else:
                                    # Update counts & clamp occupancy [0, CAPACITY]
                                    clamped = False
                                    if direction == "IN":
                                        in_total += 1
                                        target_occ = occupancy + 1
                                        if target_occ > config.CAPACITY:
                                            clamped = True
                                            occupancy = config.CAPACITY
                                            print(f"[WARNING] Bus capacity reached! Clamped to {config.CAPACITY}.")
                                        else:
                                            occupancy = target_occ
                                    else:
                                        out_total += 1
                                        target_occ = occupancy - 1
                                        if target_occ < 0:
                                            clamped = True
                                            occupancy = 0
                                            print(f"[WARNING] Bus empty! Clamped to 0 (tried {target_occ}).")
                                        else:
                                            occupancy = target_occ

                                    last_crossing_frame[stable_id] = frame_idx

                                    # Terminal notification
                                    now_str = datetime.now().strftime("%H:%M:%S")
                                    clamp_msg = " [CLAMPED]" if clamped else ""
                                    print(f"[{now_str}] Crossing: Stable ID #{stable_id:<3} | Dir: {direction:<3} | "
                                          f"IN: {in_total:<3} | OUT: {out_total:<3} | ON BOARD: {occupancy}/{config.CAPACITY}"
                                          f"{clamp_msg}")

                                    # Write to crossings_log_<door>.csv
                                    log_crossing(door, stable_id, direction, in_total, out_total, occupancy, clamped)

                                    # Queue ingest event (board / alight)
                                    evt_type = "board" if direction == "IN" else "alight"
                                    queue_event(evt_type, stable_id)

                            # Update confirmed side
                            last_confirmed_side[stable_id] = current_side

                    else:
                        if current_side != "UNKNOWN":
                            last_confirmed_side[stable_id] = current_side

                    # ----------------------------------------------------------
                    # DRAW DETECTIONS, TRAILS & LABELS
                    # ----------------------------------------------------------
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 130, 0), 2)

                    # Clearly visible dot at track point (yellow with black outline)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)
                    cv2.circle(frame, (cx, cy), 7, (0, 0, 0), 1)

                    # Box label
                    if debug_mode:
                        clip_tag = " [clipped]" if is_clipped else ""
                        conf_side = last_confirmed_side.get(stable_id, "NONE")
                        box_label = f"S:#{stable_id} R:#{raw_id} ({conf:.2f}){clip_tag}"
                        cv2.putText(frame, box_label, (int(x1), max(20, int(y1) - 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 130, 0), 2)

                        if is_horiz:
                            coord_label = f"y:{cy} line:{line_pos} [{conf_side}]"
                        else:
                            coord_label = f"x:{cx} line:{line_pos} [{conf_side}]"
                        cv2.putText(frame, coord_label, (int(x1), max(36, int(y1) - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 235, 255), 1)
                    else:
                        box_label = f"ID: #{stable_id}"
                        cv2.putText(frame, box_label, (int(x1), max(22, int(y1) - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 130, 0), 2)

                    # Draw trail of last 20 foot points
                    if debug_mode and stable_id in trails:
                        pts = list(trails[stable_id])
                        for p_i in range(1, len(pts)):
                            cv2.line(frame, pts[p_i - 1], pts[p_i], (0, 230, 255), 2)

                # --------------------------------------------------------------
                # DRAW REJECTED BOXES IN GREY (Filter Diagnostics)
                # --------------------------------------------------------------
                if debug_mode:
                    for r_box, r_reason in rejected_boxes:
                        rx1, ry1, rx2, ry2 = r_box
                        cv2.rectangle(frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (128, 128, 128), 1)
                        cv2.putText(frame, f"REJ: {r_reason}", (int(rx1), max(15, int(ry1) - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

                # --------------------------------------------------------------
                # DRAW COUNTING LINE & HYSTERESIS DEAD-ZONE BAND (H or V)
                # --------------------------------------------------------------
                dz_min = line_pos - margin
                dz_max = line_pos + margin

                if is_horiz:
                    span_p1 = (int(w * config.LINE_SPAN_MIN), line_pos)
                    span_p2 = (int(w * config.LINE_SPAN_MAX), line_pos)
                    dz_rect_p1 = (span_p1[0], dz_min)
                    dz_rect_p2 = (span_p2[0], dz_max)
                else:
                    span_p1 = (line_pos, int(h * config.LINE_SPAN_MIN))
                    span_p2 = (line_pos, int(h * config.LINE_SPAN_MAX))
                    dz_rect_p1 = (dz_min, span_p1[1])
                    dz_rect_p2 = (dz_max, span_p2[1])

                if debug_mode:
                    dz_overlay = frame.copy()
                    cv2.rectangle(dz_overlay, dz_rect_p1, dz_rect_p2, (0, 215, 255), -1)
                    cv2.addWeighted(dz_overlay, 0.18, frame, 0.82, 0, frame)

                    if is_horiz:
                        cv2.line(frame, (span_p1[0], dz_min), (span_p2[0], dz_min), (0, 180, 230), 1, cv2.LINE_AA)
                        cv2.line(frame, (span_p1[0], dz_max), (span_p2[0], dz_max), (0, 180, 230), 1, cv2.LINE_AA)
                    else:
                        cv2.line(frame, (dz_min, span_p1[1]), (dz_min, span_p2[1]), (0, 180, 230), 1, cv2.LINE_AA)
                        cv2.line(frame, (dz_max, span_p1[1]), (dz_max, span_p2[1]), (0, 180, 230), 1, cv2.LINE_AA)

                # Center counting line
                cv2.line(frame, span_p1, span_p2, (0, 255, 255), 2)
                line_tag = f"COUNTING LINE ({in_direction.upper()}=IN) [+/-{margin}px]"
                if is_horiz:
                    cv2.putText(frame, line_tag, (span_p1[0] + 15, line_pos - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
                else:
                    cv2.putText(frame, line_tag, (line_pos + 6, span_p1[1] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                # Render top HUD banner
                draw_hud(frame, door, door_label, count_direction,
                         in_total, out_total, ignored_total, occupancy, config.CAPACITY,
                         fps, line_orientation, line_ratio, in_direction,
                         raw_ids_created, stable_ids_created, stitches_done,
                         rej_too_far, rej_size, rej_ambig, rej_side, rej_in_use,
                         rejected_short, rejected_wide,
                         gt_in, gt_out, debug_mode, paused,
                         ingest_cum=ingest_cum)

            else:
                # Paused state display
                draw_hud(frame, door, door_label, count_direction,
                         in_total, out_total, ignored_total, occupancy, config.CAPACITY,
                         fps, line_orientation, line_ratio, in_direction,
                         raw_ids_created, stable_ids_created, stitches_done,
                         rej_too_far, rej_size, rej_ambig, rej_side, rej_in_use,
                         rejected_short, rejected_wide,
                         gt_in, gt_out, debug_mode, paused,
                         ingest_cum=ingest_cum)

            # Display main frame with door-specific window title
            cv2.imshow(f"SmartBus TN - AI People Counter ({door_label})", frame)

            # ------------------------------------------------------------------
            # KEYBOARD CONTROLS
            # ------------------------------------------------------------------
            wait_delay = 30 if paused else 1
            key = cv2.waitKey(wait_delay) & 0xFF

            if key == ord('q'):
                print("\n[INFO] 'q' pressed. Terminating session...")
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"[INFO] Debug overlay: {'ENABLED' if debug_mode else 'DISABLED'}")
            elif key == ord('t'):
                if line_orientation == "horizontal":
                    line_orientation = "vertical"
                    in_direction = "right" if in_direction == "down" else "left"
                else:
                    line_orientation = "horizontal"
                    in_direction = "down" if in_direction == "right" else "up"

                last_confirmed_side.clear()
                print(f"[INFO] Line toggled to: {line_orientation.upper()} ({in_direction.upper()} = IN) | Use W/S to move line")
            elif key == ord('i'):
                if door == "exit":
                    print("[INFO] Key 'i' (IN) ignored: EXIT door counts OUT only.")
                else:
                    gt_in += 1
                    log_ground_truth(door, "IN")
                    now_str = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now_str}] [GROUND TRUTH] Real person went IN (Total Real IN: {gt_in})")
            elif key == ord('o'):
                if door == "entrance":
                    print("[INFO] Key 'o' (OUT) ignored: ENTRANCE door counts IN only.")
                else:
                    gt_out += 1
                    log_ground_truth(door, "OUT")
                    now_str = datetime.now().strftime("%H:%M:%S")
                    print(f"[{now_str}] [GROUND TRUTH] Real person went OUT (Total Real OUT: {gt_out})")
            elif key == ord('r'):
                in_total = 0
                out_total = 0
                ignored_total = 0
                occupancy = 0
                gt_in = 0
                gt_out = 0
                rejected_short = 0
                rejected_wide = 0
                rej_too_far = 0
                rej_size = 0
                rej_ambig = 0
                rej_side = 0
                rej_in_use = 0
                raw_to_stable.clear()
                raw_last_seen.clear()
                lost_tracks.clear()
                stable_frame_count.clear()
                last_confirmed_side.clear()
                last_crossing_frame.clear()
                track_last_pos.clear()
                track_last_box.clear()
                track_velocity.clear()
                trails.clear()
                last_no_cross_report.clear()
                raw_ids_created = 0
                stable_ids_created = 0
                stitches_done = 0
                next_stable_id = 1
                # NOTE: ingest cumulative_boards/alights are NOT reset — they are session-total
                # for the heartbeat. Only the local UI counters reset.
                print("[INFO] All counters, tracking states, and ground-truth tallies reset to 0.")
            elif key == ord('w'):
                line_ratio = max(0.10, round(line_ratio - 0.02, 2))
                direction_label = "UP" if line_orientation == "horizontal" else "LEFT"
                print(f"[INFO] Counting line moved {direction_label} to {int(line_ratio * 100)}% (saved to {DOOR_LINES_FILE})")
                save_door_line(door, line_ratio)
            elif key == ord('s'):
                line_ratio = min(0.90, round(line_ratio + 0.02, 2))
                direction_label = "DOWN" if line_orientation == "horizontal" else "RIGHT"
                print(f"[INFO] Counting line moved {direction_label} to {int(line_ratio * 100)}% (saved to {DOOR_LINES_FILE})")
                save_door_line(door, line_ratio)
            elif key == ord(' ') or key == 32:
                paused = not paused
                print(f"[INFO] System {'PAUSED' if paused else 'RESUMED'}.")

    finally:
        sender.stop()
        cam.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows closed cleanly.")
        # Print final accuracy summary table
        print_summary(door, door_label, count_direction,
                      gt_in, gt_out, in_total, out_total, ignored_total,
                      raw_ids_created, stable_ids_created, stitches_done)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Exiting cleanly...")
