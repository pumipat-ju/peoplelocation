from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

router = APIRouter(prefix="/api", tags=["maps"])

_map_store = None
_cameras_using_callback = None
_invalidate_callback = None


def configure_map_store(store, cameras_using_callback=None, invalidate_callback=None):
    """Attach the SQLite MapStore and optional application integration callbacks."""
    global _map_store, _cameras_using_callback, _invalidate_callback
    _map_store = store
    _cameras_using_callback = cameras_using_callback
    _invalidate_callback = invalidate_callback


def _store():
    if _map_store is None:
        raise RuntimeError("Map store has not been configured")
    return _map_store


def _json(success, message, data=None, status_code=200):
    payload = {"success": success, "message": message}
    if data:
        payload.update(data)
    return JSONResponse(payload, status_code=status_code)


def _get_record(map_ref):
    try:
        return _store().get(map_ref)
    except ValueError:
        return None


@router.post("/upload_floorplan")
async def upload_floorplan(
    file: UploadFile = File(...),
    location: str = Form(...),
    room: str = Form(...),
):
    try:
        contents = await file.read()
        decoded = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return _json(False, "ไฟล์ที่อัปโหลดไม่ใช่รูปภาพที่อ่านได้", status_code=400)

        mime_type = (file.content_type or "image/png").lower()
        if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
            mime_type = "image/png"

        map_ref = _store().add(location, room, contents, mime_type)
        if _invalidate_callback:
            _invalidate_callback(map_ref)
        return _json(True, "เพิ่ม Map ลง Database สำเร็จ", {"floorplan_name": map_ref})
    except ValueError as exc:
        return _json(False, str(exc), status_code=409)
    except Exception:
        return _json(False, "เพิ่ม Map ไม่สำเร็จ", status_code=500)


@router.get("/floorplans")
async def list_floorplans():
    return {"floorplans": [dict(item, name=item["map_ref"]) for item in _store().list()]}


@router.get("/map-database")
async def list_map_database():
    return {"items": _store().list()}


@router.get("/map-database/image")
async def map_database_image(ref: str):
    record = _get_record(ref)
    if not record:
        return JSONResponse({"error": "Map not found"}, status_code=404)
    return Response(content=record["image_data"], media_type=record["image_mime_type"])


@router.get("/get_floorplan")
async def get_floorplan(name: str = None):
    if not name:
        items = _store().list()
        name = items[0]["map_ref"] if items else None
    record = _get_record(name) if name else None
    if not record:
        return JSONResponse({"error": "No floorplan uploaded"}, status_code=404)
    return {
        "image_base64": base64.b64encode(record["image_data"]).decode("ascii"),
        "floorplan_name": name,
    }


@router.delete("/floorplans/{floorplan_name:path}")
async def delete_floorplan(floorplan_name: str):
    if not _get_record(floorplan_name):
        return _json(False, "Floorplan not found", status_code=404)

    cameras_using_map = (
        _cameras_using_callback(floorplan_name)
        if _cameras_using_callback else []
    )
    if cameras_using_map:
        return _json(
            False,
            "ไม่สามารถลบ Floorplan ที่ยังมี Calibration ใช้งานอยู่",
            {"cameras": cameras_using_map},
            status_code=409,
        )

    _store().delete(floorplan_name)
    if _invalidate_callback:
        _invalidate_callback(floorplan_name)
    return _json(True, "ลบ Floorplan สำเร็จ", {"floorplan_name": floorplan_name})
