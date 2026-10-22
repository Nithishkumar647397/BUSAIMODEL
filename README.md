# SmartBus TN - Stage 1: Standalone AI People Counter

A science-expo prototype that counts passengers boarding and alighting across a bus door line using a standard camera and pre-trained AI.

---

## Project Structure

```text
BUSAIMODEL/
│
├── venv/                   # Virtual environment
├── config.py               # Central configuration (model, camera, thresholds, line position, stitching)
├── bytetrack_custom.yaml   # Custom ByteTrack tracker tuning (stability, track buffer)
├── status.py               # Crowd status calculation engine (GREEN, YELLOW, RED, OVERFLOW)
├── test_status.py          # Automated test suite for crowd status thresholds
├── counter.py              # Main application (YOLO11n + ByteTrack + ID Stitching + HUD)
├── crossings_log.csv       # System crossing audit log
├── ground_truth_log.csv    # Manual ground-truth tally log (recorded via 'I' and 'O' keys)
├── requirements.txt        # Python dependencies (ultralytics, opencv-python, numpy, lapx)
└── README.md               # Setup and run instructions
```

---

## Prerequisites

- **OS**: Windows 10 or 11
- **Python**: Python 3.11, 3.12, or 3.13 (64-bit)
- **Hardware**: CPU only (no GPU needed), USB or built-in webcam

---

## Setup Instructions (Windows)

Open **PowerShell** or **Command Prompt** inside the `BUSAIMODEL` folder:

1. **Activate the Virtual Environment**:
   - **PowerShell**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
     *(If script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*
   - **Command Prompt (CMD)**:
     ```cmd
     venv\Scripts\activate.bat
     ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Run the Status Engine Tests
Verify that all crowd status thresholds (49%, 50%, 79%, 80%, 100%, 110%) pass:
```powershell
python test_status.py
```

### 2. Run the AI People Counter
Launch the live camera tracking application:
```powershell
python counter.py
```

---

## Keyboard Controls

| Key | Action |
| :--- | :--- |
| `Q` | **Quit** application cleanly and print the accuracy evaluation summary |
| `T` | **Toggle Line Orientation**: Switch between **Horizontal** (up/down) and **Vertical** (left/right) |
| `W` | Move counting line **UP** (horizontal) or **LEFT** (vertical) |
| `S` | Move counting line **DOWN** (horizontal) or **RIGHT** (vertical) |
| `D` | **Toggle Debug Overlay** (shows Stable ID, Raw ID, Confidence, Foot Trails, Rejection counts & Dead-Zone) |
| `I` | **Real Person IN**: Manual ground-truth count for accuracy testing (logs to `ground_truth_log.csv`) |
| `O` | **Real Person OUT**: Manual ground-truth count for accuracy testing (logs to `ground_truth_log.csv`) |
| `R` | **Reset** all system counts, ground-truth tallies, and tracking states |
| `SPACE` | **Pause / Resume** video stream |

---

## Stability & Accuracy Diagnostics

1. **ID Stitching (`raw_id` vs `stable_id`)**:
   - When a person briefly turns away, ByteTrack might drop their raw ID and assign a new one.
   - The **ID Stitching Layer** remembers the person's last position and velocity for 45 frames (`STITCH_WINDOW_FRAMES`). When a new raw ID appears nearby, it automatically re-stitches it to the previous `stable_id`.
   - The HUD displays **ID Inflation** (`Raw IDs / Stable IDs`). A high inflation ratio with stable counts demonstrates that stitching is successfully recovering IDs!

2. **Hysteresis Dead-Zone**:
   - A `+/- 20px` band around the line prevents double counting when people pause or shuffle right on the door line.
   - State flips only register when a confirmed `ABOVE` flips to `BELOW` (or vice-versa).

3. **Ground-Truth Evaluation Table**:
   - Press `I` when a real person boards and `O` when a real person alights.
   - When you press `Q` to quit, a full accuracy comparison table (Real vs Counted, Missed, Extra, and Accuracy %) is printed directly to the terminal.
