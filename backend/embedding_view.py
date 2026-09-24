"""Read-only browser page for database/embeddings.sqlite3.

In backend/main.py, after creating `app`, add:
    try:
        from .embedding_view import router as embedding_view_router
    except ImportError:
        from embedding_view import router as embedding_view_router
    app.include_router(embedding_view_router)
"""

import os
from pathlib import Path
import sqlite3
import struct

import cv2
import numpy as np

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse


router = APIRouter()
DB_PATH = Path(os.getenv("EMBEDDING_DB_PATH", "database/embeddings.sqlite3")).resolve()
_store = None
_embedding_extractor = None


def configure_store(store):
    """Connect the controls to the same store used by the tracking pipeline."""
    global _store, DB_PATH
    _store = store
    DB_PATH = Path(store.db_path).resolve()


def configure_embedding_extractor(extractor):
    global _embedding_extractor
    _embedding_extractor = extractor


def active_store():
    if _store is None:
        raise HTTPException(status_code=503, detail="Embedding store is not configured")
    return _store


@router.post("/api/embeddings/search-image")
async def search_embedding_image(
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=100),
    min_similarity: float = Query(0.0, ge=-1.0, le=1.0),
):
    if _embedding_extractor is None:
        raise HTTPException(status_code=503, detail="OSNet extractor is not configured")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="File must be an image")
    contents = await file.read()
    if not contents or len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image is empty or larger than 15 MB")
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    try:
        embedding = _embedding_extractor.extract(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="OSNet extraction failed") from exc
    if embedding is None:
        raise HTTPException(status_code=422, detail="OSNet could not extract an embedding")
    matches = active_store().search_similar(embedding, top_k, min_similarity)
    return {
        "query_embedding_dim": int(np.asarray(embedding).size),
        "total_matches": len(matches),
        "matches": matches,
    }


@router.get("/api/embeddings/selected-ids")
def selected_ids():
    return {"selected_ids": active_store().selected_ids()}


@router.put("/api/embeddings/selected-ids/{global_id}")
def select_id(global_id: int):
    if global_id <= 0:
        raise HTTPException(status_code=422, detail="Global ID must be positive")
    store = active_store()
    store.select_id(global_id)
    return {"selected_ids": store.selected_ids()}


@router.delete("/api/embeddings/selected-ids/{global_id}")
def unselect_id(global_id: int):
    if global_id <= 0:
        raise HTTPException(status_code=422, detail="Global ID must be positive")
    store = active_store()
    store.unselect_id(global_id)
    return {"selected_ids": store.selected_ids()}


@router.get("/api/embeddings")
def list_embeddings(
    global_id: int | None = Query(None, ge=1),
    captured_date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
):
    """Metadata only: embedding BLOB is never sent to the browser."""
    if not DB_PATH.is_file():
        return {"items": [], "total": 0, "page": page, "page_size": page_size}
    filters, params = [], []
    if global_id is not None:
        filters.append("global_id = ?")
        params.append(global_id)
    if captured_date is not None:
        filters.append("captured_date = ?")
        params.append(captured_date)
    where = " WHERE " + " AND ".join(filters) if filters else ""
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            total = conn.execute("SELECT COUNT(*) FROM embeddings" + where, params).fetchone()[0]
            rows = conn.execute(
                "SELECT id, global_id, camera_name, embedding_dim, captured_date, captured_time "
                "FROM embeddings" + where + " ORDER BY captured_date DESC, global_id ASC, captured_time DESC, id DESC "
                "LIMIT ? OFFSET ?", (*params, page_size, (page - 1) * page_size),
            ).fetchall()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail="Cannot read embeddings database") from exc
    return {"items": [dict(row) for row in rows], "total": total,
            "page": page, "page_size": page_size}


@router.get("/api/embeddings/{record_id}/vector")
def get_embedding_vector(record_id: int):
    """Return the float32 vector of one stored record on explicit request."""
    if record_id <= 0:
        raise HTTPException(status_code=422, detail="Record ID must be positive")
    if not DB_PATH.is_file():
        raise HTTPException(status_code=404, detail="Record not found")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as conn:
            row = conn.execute(
                "SELECT global_id, embedding_dim, embedding FROM embeddings WHERE id = ?",
                (record_id,),
            ).fetchone()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail="Cannot read embeddings database") from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Record not found")
    global_id, dimension, raw = row
    if dimension <= 0 or len(raw) != dimension * 4:
        raise HTTPException(status_code=500, detail="Invalid stored embedding size")
    vector = struct.unpack(f"<{dimension}f", raw)
    return {"id": record_id, "global_id": global_id,
            "embedding_dim": dimension, "embedding": vector}


@router.delete("/api/embeddings/{record_id}")
def delete_embedding(record_id: int):
    if record_id <= 0:
        raise HTTPException(status_code=422, detail="Record ID must be positive")
    if not active_store().delete_record(record_id):
        raise HTTPException(status_code=404, detail="Record not found")
    return {"deleted": True, "id": record_id}


PAGE = """<!doctype html>
<html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>รายการ Embedding</title>
<style>
:root{font-family:system-ui,'Segoe UI',sans-serif;color:#e9efff;background:#101827}
body{max-width:1100px;margin:0 auto;padding:36px 20px}h1{font-size:1.7rem;margin-bottom:5px}
p{color:#aab9d1}.panel{background:#1b2940;border:1px solid #34445c;border-radius:14px;padding:20px;margin-top:25px}
form{display:flex;gap:12px;align-items:end;flex-wrap:wrap}label{display:grid;gap:6px;color:#cbd5e1;font-size:.9rem}
input,button{font:inherit;border-radius:8px;padding:10px;border:1px solid #53657d;color:#edf3ff;background:#101827}
button{cursor:pointer;background:#366fea;border-color:#366fea}button:disabled{opacity:.45;cursor:default}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;margin-top:15px}th,td{padding:12px;text-align:left;border-bottom:1px solid #34445c}
th{color:#afc6f6}#status{min-height:24px}.pager{display:flex;align-items:center;gap:12px;margin-top:16px}
</style></head><body><h1>รายการ Embedding ที่บันทึก</h1><p>ดูข้อมูลตาม Global ID และวันที่ · แสดงเฉพาะรายละเอียดรายการ ไม่แสดงเวกเตอร์</p>
<div class="panel"><form id="filters"><label>Global ID<input id="gid" type="number" min="1" step="1" placeholder="ทั้งหมด"></label>
<label>วันที่บันทึก<input id="date" type="date"></label><button type="submit">ค้นหา</button><button type="button" id="clear">ล้างตัวกรอง</button></form></div>
<div class="panel"><div id="status" role="status"></div><div class="table-wrap"><table><thead><tr><th>Global ID</th><th>วันที่</th><th>เวลา</th><th>กล้อง</th><th>ขนาดเวกเตอร์</th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="pager"><button id="prev" type="button">ก่อนหน้า</button><span id="paging"></span><button id="next" type="button">ถัดไป</button></div></div>
<script>
const rows=document.querySelector('#rows'),status=document.querySelector('#status');let page=1,total=0;
async function load(){status.textContent='กำลังโหลด...';rows.replaceChildren();const q=new URLSearchParams({page:String(page),page_size:'25'});
const gid=document.querySelector('#gid').value,date=document.querySelector('#date').value;if(gid)q.set('global_id',gid);if(date)q.set('captured_date',date);
try{const response=await fetch('/api/embeddings?'+q);if(!response.ok)throw Error('HTTP '+response.status);
const result=await response.json();total=result.total;for(const entry of result.items){const tr=document.createElement('tr');
for(const value of [entry.global_id,entry.captured_date,entry.captured_time,entry.camera_name??'—',entry.embedding_dim]){
const td=document.createElement('td');td.textContent=String(value);tr.append(td)}rows.append(tr)}
status.textContent=total?'พบ '+total+' รายการ':'ยังไม่มีรายการตามเงื่อนไขนี้';
document.querySelector('#paging').textContent='หน้า '+page+' / '+Math.max(1,Math.ceil(total/25));
document.querySelector('#prev').disabled=page<=1;document.querySelector('#next').disabled=page*25>=total;
}catch(err){status.textContent='อ่านข้อมูลไม่สำเร็จ: '+err.message;document.querySelector('#prev').disabled=true;document.querySelector('#next').disabled=true}}
document.querySelector('#filters').onsubmit=e=>{e.preventDefault();page=1;load()};
document.querySelector('#clear').onclick=()=>{document.querySelector('#filters').reset();page=1;load()};
document.querySelector('#prev').onclick=()=>{page--;load()};document.querySelector('#next').onclick=()=>{page++;load()};load();
</script></body></html>"""


@router.get("/embeddings", response_class=HTMLResponse)
def embedding_page():
    return PAGE
