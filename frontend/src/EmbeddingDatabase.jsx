import { useEffect, useState } from 'react'

// Place in frontend/src/ and render <EmbeddingDatabase /> from App.jsx.
// If Vite does not proxy /api to FastAPI, set VITE_API_BASE_URL to the backend URL.
const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

const css = `
.embedding-db { color:#e9efff; background:#101827; padding:28px; border-radius:16px; font-family:system-ui,sans-serif; }
.embedding-db h2 { margin:0 0 6px; }.embedding-db p { color:#afbdd2; }
.embedding-db .db-filters { display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin:24px 0; }
.embedding-db label { display:grid; gap:6px; font-size:14px; }
.embedding-db input,.embedding-db button { font:inherit; border-radius:8px; padding:9px 12px; border:1px solid #60708a; }
.embedding-db input { background:#1d2b40; color:white; color-scheme:dark; }
.embedding-db button { color:white; background:#315fc9; cursor:pointer; }
.embedding-db button:disabled { opacity:.45; cursor:default; }
.embedding-db .db-scroll { overflow-x:auto; }
.embedding-db table { border-collapse:collapse; width:100%; text-align:left; min-width:550px; }
.embedding-db th,.embedding-db td { padding:12px; border-bottom:1px solid #384960; }
.embedding-db th { color:#afc6f6; }.embedding-db .db-pager { display:flex; align-items:center; gap:12px; margin-top:18px; }
.embedding-db .db-vector { white-space:pre-wrap; overflow-wrap:anywhere; max-height:260px; overflow:auto; background:#0d1726; padding:12px; border-radius:8px; font-family:monospace; font-size:12px; }
`

export default function EmbeddingDatabase() {
  const [id, setId] = useState('')
  const [date, setDate] = useState('')
  const [filters, setFilters] = useState({ id: '', date: '' })
  const [page, setPage] = useState(1)
  const [result, setResult] = useState({ items: [], total: 0 })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [refreshKey, setRefreshKey] = useState(0)
  const [selected, setSelected] = useState([])
  const [newId, setNewId] = useState('')
  const [openVector, setOpenVector] = useState(null)
  const [vector, setVector] = useState(null)
  const [vectorError, setVectorError] = useState('')
  const [vectorLoading, setVectorLoading] = useState(false)

  async function toggleVector(recordId) {
    if (openVector === recordId) { setOpenVector(null); return }
    setOpenVector(recordId)
    setVector(null)
    setVectorError('')
    setVectorLoading(true)
    try {
      const response = await fetch(`${API_BASE}/api/embeddings/${recordId}/vector`)
      if (!response.ok) throw new Error(`API ${response.status}`)
      setVector((await response.json()).embedding)
    } catch (e) { setVectorError(`อ่าน embedding ไม่สำเร็จ: ${e.message}`) }
    finally { setVectorLoading(false) }
  }

  async function refreshSelected() {
    const response = await fetch(`${API_BASE}/api/embeddings/selected-ids`)
    if (!response.ok) throw new Error(`API ${response.status}`)
    setSelected((await response.json()).selected_ids)
  }

  useEffect(() => {
    const refresh = () => refreshSelected().catch(e => setError(`อ่าน ID ที่เลือกไม่สำเร็จ: ${e.message}`))
    refresh()
    const interval = setInterval(refresh, 5000)
    return () => clearInterval(interval)
  }, [])

  async function changeSelection(globalId) {
    try {
      const response = await fetch(`${API_BASE}/api/embeddings/selected-ids/${globalId}`, { method: 'PUT' })
      if (!response.ok) throw new Error(`API ${response.status}`)
      setSelected((await response.json()).selected_ids)
      setNewId('')
      setError('')
      setRefreshKey(key => key + 1)
    } catch (e) { setError(`เปลี่ยนรายการ ID ไม่สำเร็จ: ${e.message}`) }
  }

  async function deleteRecord(recordId) {
    if (!window.confirm(`ลบข้อมูล embedding รายการ #${recordId} ถาวรหรือไม่?`)) return
    try {
      const response = await fetch(`${API_BASE}/api/embeddings/${recordId}`, { method: 'DELETE' })
      if (!response.ok) throw new Error(`API ${response.status}`)
      if (openVector === recordId) setOpenVector(null)
      setRefreshKey(key => key + 1)
    } catch (e) { setError(`ลบข้อมูลไม่สำเร็จ: ${e.message}`) }
  }

  useEffect(() => {
    const controller = new AbortController()
    const params = new URLSearchParams({ page: String(page), page_size: '25' })
    if (filters.id) params.set('global_id', filters.id)
    if (filters.date) params.set('captured_date', filters.date)
    const load = () => fetch(`${API_BASE}/api/embeddings?${params}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`API ${response.status}`)
        return response.json()
      })
      .then(data => { setResult(data); setError('') })
      .catch((e) => { if (e.name !== 'AbortError') setError(`อ่านข้อมูลไม่สำเร็จ: ${e.message}`) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    setLoading(true)
    load()
    const interval = setInterval(load, 5000)
    return () => { clearInterval(interval); controller.abort() }
  }, [filters, page, refreshKey])

  function search(event) {
    event.preventDefault()
    setPage(1)
    setFilters({ id, date })
    setRefreshKey(key => key + 1)
  }

  function clear() {
    setId('')
    setDate('')
    setPage(1)
    setFilters({ id: '', date: '' })
    setRefreshKey(key => key + 1)
  }

  return <section className="embedding-db">
    <style>{css}</style>
    <h2>รายการ Embedding</h2>
    <p>ข้อมูลที่บันทึกแยกตาม Global ID และวันที่ · แสดงเฉพาะรายละเอียด ไม่แสดงเวกเตอร์</p>
    <form className="db-filters" onSubmit={e => { e.preventDefault(); if (newId && Number(newId) > 0) changeSelection(newId) }}>
      <label>Global ID ที่ต้องการเก็บ<input type="number" min="1" step="1" value={newId} onChange={e => setNewId(e.target.value)} placeholder="เช่น 3" /></label>
      <button type="submit" disabled={!newId || Number(newId) < 1}>เริ่มเก็บ ID นี้</button>
    </form>
    <p>รอบันทึก: {selected.length ? selected.map(gid => <span key={gid} style={{ marginRight: 12 }}>ID {gid}</span>) : 'ยังไม่ได้เลือก ID'} · เมื่อบันทึกได้แล้วระบบจะหยุดเก็บ ID นั้นอัตโนมัติ</p>
    <form className="db-filters" onSubmit={search}>
      <label>Global ID<input type="number" min="1" step="1" value={id} onChange={e => setId(e.target.value)} placeholder="ทั้งหมด" /></label>
      <label>วันที่<input type="date" value={date} onChange={e => setDate(e.target.value)} /></label>
      <button type="submit">ค้นหา</button><button type="button" onClick={clear}>ล้างตัวกรอง</button>
      <button type="button" onClick={() => setRefreshKey(key => key + 1)}>รีเฟรชรายการ</button>
    </form>
    <p role="status">{error || (loading ? 'กำลังโหลด...' : `พบ ${result.total} รายการ`)}</p>
    <div className="db-scroll"><table><thead><tr><th>Global ID</th><th>วันที่</th><th>เวลา</th><th>กล้อง</th><th>ขนาดเวกเตอร์</th><th>ข้อมูล</th><th>ลบ</th></tr></thead>
      <tbody>{result.items.map(row => <FragmentRow key={row.id} row={row} expanded={openVector === row.id}
        toggle={() => toggleVector(row.id)} remove={() => deleteRecord(row.id)} vector={vector} vectorLoading={vectorLoading} vectorError={vectorError} />)}</tbody></table></div>
    <div className="db-pager"><button type="button" disabled={page <= 1 || loading} onClick={() => setPage(p => p - 1)}>ก่อนหน้า</button>
      <span>หน้า {page} / {Math.max(1, Math.ceil(result.total / 25))}</span>
      <button type="button" disabled={page * 25 >= result.total || loading} onClick={() => setPage(p => p + 1)}>ถัดไป</button></div>
  </section>
}

function FragmentRow({ row, expanded, toggle, remove, vector, vectorLoading, vectorError }) {
  return <>
    <tr><td>{row.global_id}</td><td>{row.captured_date}</td><td>{row.captured_time}</td>
      <td>{row.camera_name || '—'}</td><td>{row.embedding_dim}</td>
      <td><button type="button" onClick={toggle}>{expanded ? 'ซ่อน embedding' : 'ดู embedding'}</button></td>
      <td><button type="button" onClick={remove}>ลบ</button></td></tr>
    {expanded && <tr><td colSpan="7"><strong>Embedding ของ Global ID {row.global_id}</strong>
      {vectorLoading ? <p>กำลังโหลด...</p> : vectorError ? <p role="alert">{vectorError}</p>
        : vector && <pre className="db-vector">{vector.map((value, index) => `${index}: ${value}`).join('\n')}</pre>}
    </td></tr>}
  </>
}
