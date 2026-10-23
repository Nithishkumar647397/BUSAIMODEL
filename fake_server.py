"""
SmartBus TN - Fake Ingest Server (Test Only)
=============================================
Mimics the friend's remote server for local testing.
Runs on http://127.0.0.1:8000

Endpoints:
  POST /ingest/event      - Accept board/alight events
  POST /ingest/heartbeat  - Accept 1-Hz heartbeats
  POST /ingest/frame      - Accept JPEG frames

Terminal keys (type + Enter):
  m  - Toggle mode_not_ai   replies for new events
  o  - Toggle overflow      replies for new events (board only)
  d  - Toggle 20% pre-processing DROP (connection closed before storing)
  l  - Toggle 30% "lost reply" mode (event stored normally, reply not sent)
  k  - Toggle 401 BAD KEY   replies
  q  - Quit

Dedupe rules
  * applied event_id  -> duplicate reply: {"applied":false,"reason":"duplicate"}
  * rejected event_id -> same rejection reason as the first time
  * lost-reply events ARE stored; retried event gets "duplicate" reply
"""

import random
import sys
import threading

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("[ERROR] FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn[standard]")
    sys.exit(1)

app = FastAPI(title="SmartBus TN Fake Ingest Server")

# ---------------------------------------------------------------------------
# Shared server state
# ---------------------------------------------------------------------------
state = {
    # Toggle flags
    "mode_not_ai": False,
    "overflow":    False,
    "drop_20pct":  False,
    "lost_reply":  False,
    "bad_key":     False,

    # Dedupe store: event_id -> outcome dict
    #   {"applied": bool, "reason": str}
    "seen_events": {},

    # Per-door applied totals for heartbeat comparison
    # door -> {"boards": int, "alights": int}
    "door_totals": {},

    # Live stats
    "cnt_applied":    0,
    "cnt_rejected":   0,
    "cnt_duplicates": 0,
    "cnt_lost":       0,
}

_lock = threading.Lock()


def _stats_line() -> str:
    return (
        f"  [STATS] applied={state['cnt_applied']}  "
        f"rejected={state['cnt_rejected']}  "
        f"dupes={state['cnt_duplicates']}  "
        f"lost={state['cnt_lost']}"
    )


def _print_state():
    flags = (
        f"mode_not_ai={state['mode_not_ai']}  "
        f"overflow={state['overflow']}  "
        f"drop20={state['drop_20pct']}  "
        f"lostReply={state['lost_reply']}  "
        f"bad_key={state['bad_key']}"
    )
    print(f"  [TOGGLE] {flags}")
    print(_stats_line())


# ---------------------------------------------------------------------------
# Keyboard listener thread
# ---------------------------------------------------------------------------
def _keyboard_loop():
    print("\n[fake_server] Keys: m=mode_not_ai  o=overflow  d=drop20%  l=lostReply  k=bad401  q=quit")
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
            elif ch == "l":
                state["lost_reply"] = not state["lost_reply"]
                print(f"  >> lost_reply = {state['lost_reply']}")
            elif ch == "k":
                state["bad_key"] = not state["bad_key"]
                print(f"  >> bad_key = {state['bad_key']}")
            elif ch == "q":
                import os
                os._exit(0)
            _print_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_key(request: Request) -> bool:
    """Accept any non-empty key unless bad_key mode is active."""
    if state["bad_key"]:
        return False
    return bool(request.headers.get("X-Ingest-Key", ""))


def _get_door_totals(door: str) -> dict:
    """Return (and lazily create) the per-door applied totals dict."""
    if door not in state["door_totals"]:
        state["door_totals"][door] = {"boards": 0, "alights": 0}
    return state["door_totals"][door]


# ---------------------------------------------------------------------------
# POST /ingest/event
# ---------------------------------------------------------------------------
@app.post("/ingest/event")
async def ingest_event(request: Request):
    # --- Auth check first (before any processing) ---
    with _lock:
        if not _check_key(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})

        # --- Pre-processing DROP (d key) — close before storing ---
        if state["drop_20pct"] and random.random() < 0.20:
            # Return 503; event is NOT stored
            return Response(status_code=503, content=b"dropped (simulated)")

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=422, content={"detail": "invalid JSON"})

    event_id = body.get("event_id", "")
    evt_type = body.get("type", "")
    door     = body.get("door", "?")
    track_id = body.get("track_id", "?")

    with _lock:
        # ----------------------------------------------------------------
        # DEDUPE CHECK
        # ----------------------------------------------------------------
        if event_id in state["seen_events"]:
            prev = state["seen_events"][event_id]
            state["cnt_duplicates"] += 1

            if prev["applied"]:
                # Previously applied -> standard duplicate reply
                print(
                    f"  [EVENT] DUPLICATE (was applied)  "
                    f"id={event_id} type={evt_type} door={door} track={track_id}"
                )
                print(_stats_line())
                return JSONResponse({"applied": False, "reason": "duplicate"})
            else:
                # Previously rejected -> repeat the SAME rejection reason
                same_reason = prev["reason"]
                print(
                    f"  [EVENT] DUPLICATE (was rejected:{same_reason})  "
                    f"id={event_id} type={evt_type} door={door} track={track_id}"
                )
                print(_stats_line())
                return JSONResponse({"applied": False, "reason": same_reason})

        # ----------------------------------------------------------------
        # NEW EVENT: decide outcome
        # ----------------------------------------------------------------
        if state["mode_not_ai"]:
            outcome = {"applied": False, "reason": "mode_not_ai"}
        elif state["overflow"] and evt_type == "board":
            outcome = {"applied": False, "reason": "overflow"}
        else:
            # Apply the event
            outcome = {"applied": True, "reason": ""}
            totals = _get_door_totals(door)
            if evt_type == "board":
                totals["boards"] += 1
            elif evt_type == "alight":
                if totals["boards"] <= 0:
                    # alight_at_zero: applied=True with warning reason
                    outcome = {"applied": True, "reason": "alight_at_zero"}
                else:
                    totals["alights"] += 1

        # Store outcome in dedupe store
        state["seen_events"][event_id] = outcome

        # Update live stats
        if outcome["applied"]:
            state["cnt_applied"] += 1
        else:
            state["cnt_rejected"] += 1

        # ----------------------------------------------------------------
        # LOST REPLY mode (l key): store event, close without replying
        # ----------------------------------------------------------------
        if state["lost_reply"] and random.random() < 0.30:
            state["cnt_lost"] += 1
            print(
                f"  [EVENT] LOST_REPLY  "
                f"id={event_id} type={evt_type} door={door} track={track_id} "
                f"outcome={outcome}"
            )
            print(_stats_line())
            # Disconnect without sending a body (counter will retry same event_id)
            return Response(status_code=200, content=b"")  # empty body -> JSON decode fails -> retry

        # ----------------------------------------------------------------
        # Normal reply
        # ----------------------------------------------------------------
        tag = "OK       " if outcome["applied"] else f"REJECTED ({outcome['reason']})"
        print(
            f"  [EVENT] {tag}  "
            f"id={event_id} type={evt_type} door={door} track={track_id}"
        )
        print(_stats_line())
        return JSONResponse(outcome)


# ---------------------------------------------------------------------------
# POST /ingest/heartbeat
# ---------------------------------------------------------------------------
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
        totals      = _get_door_totals(door)
        srv_boards  = totals["boards"]
        srv_alights = totals["alights"]

    diff_b = cam_boards  - srv_boards
    diff_a = cam_alights - srv_alights
    match  = (diff_b == 0 and diff_a == 0)

    if match:
        print(
            f"  [HB]    match ok    door={door} run={run_id} "
            f"boards={cam_boards} alights={cam_alights} fps={fps} cam_ok={camera_ok}"
        )
    else:
        print(
            f"  [HB]    MISMATCH    door={door} run={run_id} "
            f"cam_boards={cam_boards}(diff {diff_b:+d}) "
            f"cam_alights={cam_alights}(diff {diff_a:+d}) fps={fps}"
        )

    return JSONResponse({
        "match":          match,
        "server_boards":  srv_boards,
        "server_alights": srv_alights,
    })


# ---------------------------------------------------------------------------
# POST /ingest/frame
# ---------------------------------------------------------------------------
@app.post("/ingest/frame")
async def ingest_frame(request: Request):
    with _lock:
        if not _check_key(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        if state["drop_20pct"] and random.random() < 0.20:
            return Response(status_code=503, content=b"dropped")

    door = request.query_params.get("door", "?")
    body = await request.body()
    kb   = len(body) / 1024
    print(f"  [FRAME] door={door} size={kb:.1f} KB")
    return JSONResponse({"received": True})


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    kb_thread = threading.Thread(target=_keyboard_loop, daemon=True)
    kb_thread.start()
    print("=" * 64)
    print(" SmartBus TN - Fake Ingest Server")
    print(" Listening: http://127.0.0.1:8000")
    print(" Endpoints: /ingest/event  /ingest/heartbeat  /ingest/frame")
    print("=" * 64)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
