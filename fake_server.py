"""
SmartBus TN - Fake Ingest Server (Test Only)
=============================================
Mimics the friend's remote server for local testing.
Runs on http://127.0.0.1:8000

Endpoints:
  POST /ingest/event      - Accept board/alight events
  POST /ingest/heartbeat  - Accept 1-Hz heartbeats
  POST /ingest/frame      - Accept JPEG frames (when SEND_FRAMES=True)

Terminal keys (type + Enter while running):
  m  - Toggle mode_not_ai replies
  o  - Toggle overflow replies
  d  - Toggle 20% random drops (simulates network failure)
  k  - Toggle 401 BAD KEY replies

Run with:
  python fake_server.py
"""

import random
import sys
import threading
import time

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("[ERROR] FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn[standard]")
    sys.exit(1)

app = FastAPI(title="SmartBus TN Fake Ingest Server")

# ---- Mutable server state (toggled via keyboard) ----
state = {
    "mode_not_ai": False,
    "overflow":    False,
    "drop_20pct":  False,
    "bad_key":     False,
    "seen_events": {},   # event_id -> True  (duplicate tracking)
    "boards":  0,
    "alights": 0,
}
_lock = threading.Lock()


def _print_state():
    print(
        f"  [STATE] mode_not_ai={state['mode_not_ai']}  "
        f"overflow={state['overflow']}  "
        f"drop20={state['drop_20pct']}  "
        f"bad_key={state['bad_key']}  "
        f"boards={state['boards']}  alights={state['alights']}"
    )


# ---- Keyboard listener thread ----
def _keyboard_loop():
    print("\n[fake_server] Keys: m=mode_not_ai  o=overflow  d=drop20%  k=bad401  q=quit")
    print("[fake_server] Type a key and press Enter.\n")
    while True:
        try:
            ch = input().strip().lower()
        except EOFError:
            break
        with _lock:
            if ch == "m":
                state["mode_not_ai"] = not state["mode_not_ai"]
                print(f"  >> mode_not_ai = {state['mode_not_ai']}")
            elif ch == "o":
                state["overflow"] = not state["overflow"]
                print(f"  >> overflow = {state['overflow']}")
            elif ch == "d":
                state["drop_20pct"] = not state["drop_20pct"]
                print(f"  >> drop_20pct = {state['drop_20pct']}")
            elif ch == "k":
                state["bad_key"] = not state["bad_key"]
                print(f"  >> bad_key = {state['bad_key']}")
            elif ch == "q":
                import os
                os._exit(0)
            _print_state()


def _check_key(request: Request) -> bool:
    """Accept any non-empty key unless bad_key mode is on."""
    if state["bad_key"]:
        return False
    return bool(request.headers.get("X-Ingest-Key", ""))


def _should_drop() -> bool:
    return state["drop_20pct"] and random.random() < 0.20


# ==============================================================================
# POST /ingest/event
# ==============================================================================
@app.post("/ingest/event")
async def ingest_event(request: Request):
    with _lock:
        if _should_drop():
            return Response(status_code=503, content=b"dropped (simulated)")
        if not _check_key(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=422, content={"detail": "invalid JSON"})

    event_id  = body.get("event_id", "")
    evt_type  = body.get("type", "")
    door      = body.get("door", "?")
    track_id  = body.get("track_id", "?")

    with _lock:
        # Duplicate check
        if event_id in state["seen_events"]:
            print(f"  [EVENT] DUPLICATE   id={event_id} type={evt_type} door={door} track={track_id}")
            return JSONResponse({"applied": False, "reason": "duplicate"})

        # mode_not_ai rejection
        if state["mode_not_ai"]:
            print(f"  [EVENT] mode_not_ai id={event_id} type={evt_type} door={door} track={track_id}")
            return JSONResponse({"applied": False, "reason": "mode_not_ai"})

        # overflow rejection (only for board events)
        if state["overflow"] and evt_type == "board":
            print(f"  [EVENT] overflow    id={event_id} type={evt_type} door={door} track={track_id}")
            return JSONResponse({"applied": False, "reason": "overflow"})

        # Mark as seen
        state["seen_events"][event_id] = True

        # Update server-side counters
        if evt_type == "board":
            state["boards"] += 1
        elif evt_type == "alight":
            if state["alights"] <= 0:
                print(f"  [EVENT] alight_at_zero (applied with warning) id={event_id}")
                return JSONResponse({"applied": True, "reason": "alight_at_zero"})
            state["alights"] -= 1

        print(
            f"  [EVENT] OK          id={event_id} type={evt_type} door={door} track={track_id}"
            f"  | boards={state['boards']} alights={state['alights']}"
        )
        return JSONResponse({"applied": True, "reason": ""})


# ==============================================================================
# POST /ingest/heartbeat
# ==============================================================================
@app.post("/ingest/heartbeat")
async def ingest_heartbeat(request: Request):
    with _lock:
        if not _check_key(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=422, content={"detail": "invalid JSON"})

    door        = body.get("door", "?")
    run_id      = body.get("run_id", "?")
    cam_boards  = body.get("cumulative_boards", 0)
    cam_alights = body.get("cumulative_alights", 0)
    fps         = body.get("fps", 0.0)
    camera_ok   = body.get("camera_ok", True)

    with _lock:
        srv_boards  = state["boards"]
        srv_alights = state["alights"]
        match = (cam_boards == srv_boards)

    flag = "MISMATCH" if not match else "ok      "
    print(
        f"  [HB]    {flag} door={door} run={run_id} "
        f"cam_boards={cam_boards} srv_boards={srv_boards} fps={fps} cam_ok={camera_ok}"
    )

    return JSONResponse({
        "match":          match,
        "server_boards":  srv_boards,
        "server_alights": srv_alights,
    })


# ==============================================================================
# POST /ingest/frame
# ==============================================================================
@app.post("/ingest/frame")
async def ingest_frame(request: Request):
    with _lock:
        if not _check_key(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        if _should_drop():
            return Response(status_code=503, content=b"dropped")

    door = request.query_params.get("door", "?")
    body = await request.body()
    kb   = len(body) / 1024
    print(f"  [FRAME] door={door} size={kb:.1f} KB")
    return JSONResponse({"received": True})


# ==============================================================================
# Startup
# ==============================================================================
if __name__ == "__main__":
    kb_thread = threading.Thread(target=_keyboard_loop, daemon=True)
    kb_thread.start()
    print("=" * 64)
    print(" SmartBus TN - Fake Ingest Server")
    print(" Listening: http://127.0.0.1:8000")
    print(" Endpoints: /ingest/event  /ingest/heartbeat  /ingest/frame")
    print("=" * 64)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
