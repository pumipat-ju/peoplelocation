from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path


class MapStore:
    """SQLite-backed floorplan storage. Images are stored as BLOBs; no static files are required."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS map_database (
                    map_date TEXT NOT NULL,
                    location TEXT NOT NULL,
                    room TEXT NOT NULL,
                    image_data BLOB NOT NULL,
                    image_mime_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (map_date, location, room)
                )
            """)
            conn.commit()

    def _connect(self):
        return sqlite3.connect(str(self.db_path), timeout=30)

    @staticmethod
    def make_ref(map_date, location, room):
        return f"{map_date}|{location}|{room}"

    @staticmethod
    def parse_ref(map_ref):
        parts = str(map_ref or "").split("|", 2)
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ValueError("Invalid map reference")
        return tuple(part.strip() for part in parts)

    def add(self, location, room, image_data, mime_type, map_date=None):
        location = str(location or "").strip()
        room = str(room or "").strip()
        if not location or not room:
            raise ValueError("location and room are required")
        if "|" in location or "|" in room:
            raise ValueError("สถานที่และห้องห้ามมีเครื่องหมาย |")
        map_date = map_date or datetime.now().astimezone().date().isoformat()
        created_at = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO map_database (map_date, location, room, image_data, image_mime_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (map_date, location, room, sqlite3.Binary(image_data), mime_type, created_at),
                )
                conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("มี Map ของสถานที่และห้องนี้ในวันนี้แล้ว") from exc
        return self.make_ref(map_date, location, room)

    def list(self):
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT map_date, location, room, image_mime_type, created_at FROM map_database ORDER BY map_date DESC, location, room"
            ).fetchall()
        return [dict(row, map_ref=self.make_ref(row["map_date"], row["location"], row["room"])) for row in rows]

    def get(self, map_ref):
        map_date, location, room = self.parse_ref(map_ref)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT map_date, location, room, image_data, image_mime_type, created_at FROM map_database WHERE map_date=? AND location=? AND room=?",
                (map_date, location, room),
            ).fetchone()
        return dict(row) if row else None

    def delete(self, map_ref):
        map_date, location, room = self.parse_ref(map_ref)
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM map_database WHERE map_date=? AND location=? AND room=?",
                (map_date, location, room),
            )
            conn.commit()
        return cur.rowcount > 0
