"""
SmartBus TN - Local Broadcast Server (Stage 2)
FastAPI Server for Telemetry Ingestion and Real-time WebSocket Broadcasting

Endpoints:
  - POST /update : Ingests JSON telemetry from counter.py, updates state, and broadcasts via WebSocket.
  - WebSocket /ws: Streams live state to connected browser dashboards immediately on connect and on every update.
  - GET /state   : Returns latest state JSON (for browser/curl testing).
  - GET /        : Serves test.html visual monitor.
  - Heartbeat    : Automatically marks camera_ok = false and broadcasts if no update arrives for > 3.0s.
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from status import get_status


# ------------------------------------------------------------------------------
# JSON CONTRACT SPECIFICATION
# ------------------------------------------------------------------------------
class UpdatePayload(BaseModel):
    boarded: int
    alighted: int
    occupancy: int
    capacity: int = config.CAPACITY
    percent: float = 0.0
    status: str = "GREEN"
    camera_ok: bool = True
    timestamp: float = 0.0


# Initialize latest state with single source of truth from status.py
init_status, init_pct = get_status(0, config.CAPACITY)
latest_state = {
    "boarded": 0,
    "alighted": 0,
    "occupancy": 0,
    "capacity": config.CAPACITY,
    "percent": init_pct,
    "status": init_status,
    "camera_ok": False,
    "timestamp": time.time()
}

last_update_time = 0.0


# ------------------------------------------------------------------------------
# WEBSOCKET CONNECTION MANAGER
# ------------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        dead_connections = set()
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)

        for dead in dead_connections:
            self.active_connections.discard(dead)


manager = ConnectionManager()


# ------------------------------------------------------------------------------
# HEARTBEAT BACKGROUND WORKER
# ------------------------------------------------------------------------------
async def heartbeat_worker():
    """
    Checks if camera telemetry has stopped arriving for 3.0 seconds.
    If timeout occurs, marks camera_ok = false and broadcasts to all clients.
    """
    global last_update_time
    while True:
        await asyncio.sleep(0.5)
        now = time.time()
        if latest_state["camera_ok"] and (now - last_update_time > 3.0):
            latest_state["camera_ok"] = False
            latest_state["timestamp"] = now
            await manager.broadcast(latest_state)
            print("[HEARTBEAT] No telemetry received for > 3.0s. Marked camera_ok = False.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start heartbeat monitor in the background
    heartbeat_task = asyncio.create_task(heartbeat_worker())
    yield
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="SmartBus TN Stage 2 Broadcast Server", lifespan=lifespan)


# ------------------------------------------------------------------------------
# REST & WEBSOCKET ROUTES
# ------------------------------------------------------------------------------
@app.get("/")
async def get_test_page():
    """Serves the plain test.html WebSocket monitor."""
    html_path = os.path.join(os.path.dirname(__file__), "test.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"error": "test.html not found"}


@app.get("/state")
async def get_state():
    """Returns the latest state JSON for testing in a browser or curl."""
    return latest_state


@app.post("/update")
async def update_state(payload: UpdatePayload):
    """
    Ingests telemetry from counter.py.
    Calculates status and percent strictly through status.py:get_status().
    """
    global last_update_time
    last_update_time = time.time()

    # Calculate status and percent using the single source of truth in status.py
    computed_status, computed_pct = get_status(payload.occupancy, payload.capacity)

    latest_state["boarded"] = payload.boarded
    latest_state["alighted"] = payload.alighted
    latest_state["occupancy"] = payload.occupancy
    latest_state["capacity"] = payload.capacity
    latest_state["percent"] = computed_pct
    latest_state["status"] = computed_status
    latest_state["camera_ok"] = True
    latest_state["timestamp"] = payload.timestamp if payload.timestamp > 0 else last_update_time

    # Broadcast to all connected WebSocket clients
    await manager.broadcast(latest_state)
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint: sends latest state immediately on connect,
    and streams every subsequent update.
    """
    await manager.connect(websocket)
    try:
        # Send current state immediately upon connection
        await websocket.send_json(latest_state)
        while True:
            # Keep connection open; clients can ping or send messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
