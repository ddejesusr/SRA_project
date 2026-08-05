#!/usr/bin/env python3
"""
SRA Web Bridge Node

Runs a FastAPI web server alongside a ROS 2 node in a background thread.
Subscribes to system topics and pushes live events to browser clients
over WebSocket. Exposes REST endpoints for initial data load.

Architecture:
  Thread 1 (main): asyncio event loop — FastAPI + uvicorn + WebSocket manager
  Thread 2 (bg):   rclpy.spin() — ROS 2 subscriptions
  Bridge:           asyncio.run_coroutine_threadsafe() puts ROS 2 messages
                    into an asyncio.Queue that the main loop broadcasts from.

Dashboard: http://localhost:8000
"""

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Set

import psycopg2
import rclpy
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host":     os.getenv("SRA_DB_HOST",     "localhost"),
    "dbname":   os.getenv("SRA_DB_NAME",     "sra_db"),
    "user":     os.getenv("SRA_DB_USER",     "sra_user"),
    "password": os.getenv("SRA_DB_PASSWORD", ""),
    "port":     int(os.getenv("SRA_DB_PORT", "5432")),
}

WEB_PORT = int(os.getenv("SRA_WEB_PORT", "8000"))

# Path to the static index.html — resolved relative to this file
STATIC_DIR = Path(__file__).parent.parent / "static"


# ---------------------------------------------------------------------------
# WebSocket connection manager
# Keeps track of all connected browsers and broadcasts to all of them.
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: dict):
        dead = set()
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        self.active -= dead


# ---------------------------------------------------------------------------
# Global state shared between threads
# ---------------------------------------------------------------------------
manager    = ConnectionManager()
ros_queue: asyncio.Queue                  = None   # set on startup
main_loop: asyncio.AbstractEventLoop      = None   # set on startup

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ros_queue, main_loop
    main_loop = asyncio.get_running_loop()
    ros_queue  = asyncio.Queue()
    asyncio.create_task(broadcast_loop())
    yield

app = FastAPI(title="SRA Dashboard", docs_url=None, redoc_url=None, lifespan=lifespan)


async def broadcast_loop():
    """
    Runs forever in the asyncio event loop.
    Reads messages enqueued by the ROS 2 thread and broadcasts them
    to all connected WebSocket clients.
    """
    while True:
        msg = await ros_queue.get()
        await manager.broadcast(msg)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def dashboard():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/inventory")
async def get_inventory():
    """Return current stock for all 24 configurations."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.config_code, c.back_color, c.front_color,
                       c.fuse_position, i.quantity, i.low_stock_threshold
                FROM configurations c
                JOIN inventory i ON i.config_id = c.id
                ORDER BY c.back_color, c.front_color, c.fuse_position
            """)
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "config_code":        r[0],
                "back_color":         r[1],
                "front_color":        r[2],
                "fuse_position":      r[3],
                "quantity":           r[4],
                "low_stock_threshold": r[5],
            }
            for r in rows
        ]
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/tasks")
async def get_tasks():
    """Return the 20 most recent tasks."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, config_code, quantity, destination,
                       status, requested_at, completed_at
                FROM tasks
                ORDER BY id DESC
                LIMIT 20
            """)
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "id":           r[0],
                "config_code":  r[1],
                "quantity":     r[2],
                "destination":  r[3],
                "status":       r[4],
                "requested_at": r[5].isoformat() if r[5] else None,
                "completed_at": r[6].isoformat() if r[6] else None,
            }
            for r in rows
        ]
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/api/stats")
async def get_stats():
    """Return aggregate production statistics."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                          AS total,
                    COUNT(*) FILTER (WHERE status = 'completed')     AS completed,
                    COUNT(*) FILTER (WHERE status = 'failed')        AS failed,
                    COALESCE(SUM(quantity) FILTER
                             (WHERE status = 'completed'), 0)        AS boxes_delivered
                FROM tasks
            """)
            row = cur.fetchone()
        conn.close()
        total     = row[0] or 0
        completed = row[1] or 0
        failed    = row[2] or 0
        delivered = row[3] or 0
        rate = round(completed / total * 100) if total > 0 else 0
        return {
            "total":           total,
            "completed":       completed,
            "failed":          failed,
            "boxes_delivered": delivered,
            "success_rate":    rate,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send full initial state so the page populates immediately on connect
        inv   = await get_inventory()
        tasks = await get_tasks()
        stats = await get_stats()
        await ws.send_json({
            "type":      "init",
            "inventory": inv,
            "tasks":     tasks,
            "stats":     stats,
        })

        # Hold the connection open; real-time updates arrive via broadcast_loop
        while True:
            await asyncio.sleep(30)
            await ws.send_json({"type": "ping"})

    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# ROS 2 bridge node
# ---------------------------------------------------------------------------

class WebBridgeNode(Node):
    """
    Lightweight ROS 2 node that subscribes to system topics and
    forwards each message into the asyncio queue for WebSocket broadcast.
    """

    def __init__(self):
        super().__init__("sra_web_bridge")

        self.create_subscription(
            String, "/sra/alerts/events",  self.on_alert,          10
        )
        self.create_subscription(
            String, "/sra/tasks/active",   self.on_task_active,    10
        )
        self.create_subscription(
            String, "/sra/tasks/completed",self.on_task_completed, 10
        )
        self.create_subscription(
            String, "/sra/tasks/status",   self.on_task_status,    10
        )

        self.get_logger().info(
            f"Web bridge ready. Dashboard → http://localhost:{WEB_PORT}"
        )

    def _push(self, msg: dict):
        """Thread-safe: schedule a queue.put() on the main asyncio loop."""
        if main_loop and ros_queue:
            asyncio.run_coroutine_threadsafe(ros_queue.put(msg), main_loop)

    def on_alert(self, msg: String):
        try:
            data = json.loads(msg.data)
            self._push({"type": "alert", **data})
        except Exception:
            pass

    def on_task_active(self, msg: String):
        try:
            data = json.loads(msg.data)
            self._push({"type": "task_active", **data})
        except Exception:
            pass

    def on_task_completed(self, msg: String):
        try:
            data = json.loads(msg.data)
            self._push({"type": "task_completed", **data})
        except Exception:
            pass

    def on_task_status(self, msg: String):
        try:
            data = json.loads(msg.data)
            self._push({"type": "task_status", **data})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _ros2_thread():
    rclpy.init()
    node     = WebBridgeNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def main():
    thread = threading.Thread(target=_ros2_thread, daemon=True)
    thread.start()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=WEB_PORT,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
