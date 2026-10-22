"""
SmartBus TN - Configuration Settings
Stage 1: Standalone AI People Counter

All operational parameters are centralized here for easy tuning during the science expo.
"""

# Model & Inference Settings
MODEL = "yolo11n.pt"         # Pre-trained YOLO11 nano model
IMGSZ = 416                  # Inference image size (416 gives fast CPU inference >= 12 FPS)
CONF = 0.35                  # Minimum detection confidence threshold (tuned to 0.35)

# Custom ByteTrack Tracker Settings (configured in bytetrack_custom.yaml)
#   tracker_type: bytetrack
#   track_high_thresh: 0.4   # First-stage match threshold
#   track_low_thresh: 0.1    # Second-stage threshold for low-score matches
#   new_track_thresh: 0.5    # Minimum confidence to initialize a new track ID (prevents spurious IDs)
#   track_buffer: 60         # Keeps lost tracks alive for 60 frames (~2-4s) across brief occlusions
#   match_thresh: 0.8        # Association similarity threshold (IoU/cost)
#   fuse_score: True         # Fuses score with motion
TRACKER = "bytetrack_custom.yaml"

# Camera Settings
CAMERA_INDEX = 0             # Default camera index (can be overridden via --camera argument)
CAPTURE_WIDTH = 640          # Default capture width (640x480 standard)
CAPTURE_HEIGHT = 480         # Default capture height

# Per-Door Settings (Stage 1 Separation: ENTRANCE vs EXIT)
# Each door has its own line position, line orientation, and in_direction
# to easily adjust for reversed/angled cameras without affecting the other door.
DOOR_SETTINGS = {
    "entrance": {
        "count_direction": "IN",
        "line_mode": "horizontal",
        "line_pos": 0.5,
        "in_direction": "down",
        "label": "ENTRANCE - counts IN only",
    },
    "exit": {
        "count_direction": "OUT",
        "line_mode": "horizontal",
        "line_pos": 0.5,
        "in_direction": "down",
        "label": "EXIT - counts OUT only",
    },
    "both": {
        "count_direction": "BOTH",
        "line_mode": "horizontal",
        "line_pos": 0.5,
        "in_direction": "down",
        "label": "SINGLE DOOR - counts IN and OUT",
    },
}

# Bus Capacity & Crowd Settings
CAPACITY = 10                # Maximum passenger capacity of the bus prototype

# Counting Line Settings
LINE_ORIENTATION = "horizontal"  # "horizontal" (people move up/down) or "vertical" (people move left/right)
LINE_POSITION = 0.5              # Line position as fraction of frame dimension (0.0 to 1.0)
LINE_Y = 0.5                     # Default position for horizontal line (fraction of height)
LINE_X = 0.5                     # Default position for vertical line (fraction of width)

# Crossing Direction:
# For Horizontal Line: "down" (top-to-bottom = IN) or "up" (bottom-to-top = IN)
# For Vertical Line  : "right" (left-to-right = IN) or "left" (right-to-left = IN)
IN_DIRECTION = "down"

LINE_MARGIN_PX = 20          # Dead-zone (hysteresis) margin on either side of the line in pixels
LINE_SPAN_MIN = 0.0          # Line span along perpendicular axis start (0.0 = edge)
LINE_SPAN_MAX = 1.0          # Line span along perpendicular axis end (1.0 = edge)
LINE_X_MIN = 0.0             # Backward compatibility for horizontal span start
LINE_X_MAX = 1.0             # Backward compatibility for horizontal span end

# Track Point Settings:
# options:
#   "center" = centre of the body box (mid-torso)
#   "chest"  = horizontal centre of the box, 30% down from the box top (upper body / face area)
#   "feet"   = bottom-centre of the box (ground contact)
TRACK_POINT = "center"

# Detection Filtering Constants (loosened to capture distant/partial persons)
MIN_BOX_HEIGHT_FRAC = 0.12   # Ignore bounding boxes shorter than 12% of frame height (lowered from 0.25)
USE_ASPECT_FILTER = False    # Disabled by default so walking/turned poses are not rejected
MAX_ASPECT_RATIO = 1.2       # Maximum allowed width/height ratio if USE_ASPECT_FILTER is enabled

# ID Stitching Settings & Constants
USE_STITCHING = True         # When False, stable_id = raw_id and stitching code is skipped entirely
STITCH_WINDOW_FRAMES = 20    # Max frames (~0.7-1.5s) to retain lost track history for potential stitching
STITCH_MAX_DIST_FRAC = 0.06  # Max distance fraction of frame width (~60px at 1000px width)
STITCH_SIZE_TOLERANCE = 0.30 # Box width and height must each be within +/- 30% of lost track size
RAW_ID_EXPIRY_FRAMES = 60    # If a raw ID is absent for >60 frames, discard mapping and treat as brand new

# Tracking Stability & Debounce Filters
MIN_TRACK_FRAMES = 3         # Minimum frames a stable track must be seen before registering a crossing
COOLDOWN_FRAMES = 15         # Number of frames to ignore subsequent crossings for the same stable ID (prevents jitter)

# Local Server Broadcasting Settings (Stage 2)
SEND_TO_SERVER = True        # Transmit telemetry JSON to local FastAPI server
SERVER_URL = "http://127.0.0.1:8000/update"

