"""
SmartBus TN - Camera Discovery & Validation Tool
Stage 1: Multi-Door Camera Setup

Tests camera indexes 0 to 5. For each working camera:
  - Prints index and resolution
  - Displays a 3-second live preview with index overlaid
  - Closes cleanly and prints a final summary of operational indexes
"""

import os
import sys
import time

# Verify running inside virtual environment
if "venv" not in sys.prefix.lower():
    print("Run venv\\Scripts\\activate first")
    sys.exit(1)

import cv2


def test_camera_index(idx: int) -> tuple[bool, int, int]:
    """
    Attempts to open a camera index.
    Returns (success, width, height).
    """
    cap = None
    try:
        # On Windows, cv2.CAP_DSHOW avoids MSMF hangs and opens USB cameras faster
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            if cap is not None:
                cap.release()
            return False, 0, 0

        # Try reading one frame to confirm actual video feed
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, 0, 0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w == 0 or h == 0:
            h, w = frame.shape[:2]

        # Camera successfully validated! Show 3-second preview
        print(f"[INFO] Camera index {idx}: OPENED (Resolution: {w}x{h}). Showing 3-second preview...")

        win_name = f"Camera Index {idx} Preview - SmartBus TN"
        start_time = time.time()
        preview_duration = 3.0

        while (time.time() - start_time) < preview_duration:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            remaining = max(0.0, preview_duration - (time.time() - start_time))

            # Overlay diagnostic text
            cv2.rectangle(frame, (10, 10), (460, 95), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 10), (460, 95), (0, 255, 255), 2)
            cv2.putText(frame, f"CAMERA INDEX: {idx} ({w}x{h})", (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(frame, f"Testing live stream... {remaining:.1f}s remaining", (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
            cv2.putText(frame, "Press 'q' or SPACE to skip to next camera", (20, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 100), 1)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 32, 27):  # 'q', space, or ESC to skip preview
                break

        cv2.destroyAllWindows()
        cap.release()
        return True, w, h

    except Exception as e:
        print(f"[ERROR] Error probing camera index {idx}: {e}")
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        return False, 0, 0


def main():
    print("=" * 64)
    print(" SmartBus TN - Camera Hardware Discovery (Indexes 0 to 5)")
    print("=" * 64)

    working_cameras = []

    for idx in range(6):
        print(f"\n[PROBING] Checking camera index {idx}...")
        success, w, h = test_camera_index(idx)
        if success:
            working_cameras.append((idx, w, h))
        else:
            print(f"[INFO] Camera index {idx}: Not available / No video feed.")

    print("\n" + "=" * 64)
    print(" Camera Discovery Summary")
    print("=" * 64)
    if working_cameras:
        print(f" Found {len(working_cameras)} available camera(s):")
        for idx, w, h in working_cameras:
            print(f"   -> Camera Index {idx}: {w}x{h}")
        print("\n Recommended configuration:")
        print(f"   Entrance Door: --camera {working_cameras[0][0]}")
        if len(working_cameras) > 1:
            print(f"   Exit Door:     --camera {working_cameras[1][0]}")
        else:
            print("   (Only 1 camera found. For 2 doors, connect a second USB camera)")
    else:
        print(" [WARNING] No operational cameras detected on indexes 0 to 5.")
        print(" Please verify that your webcam/USB camera is plugged in and")
        print(" not being used by another application (Zoom, Teams, Camera app).")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Discovery cancelled by user.")
        cv2.destroyAllWindows()
