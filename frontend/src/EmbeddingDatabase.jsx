import { useEffect, useState } from 'react'

// Place in frontend/src/ and render <EmbeddingDatabase /> from App.jsx.
const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8899')
  .trim()
  .replace(/\/+$/, '')
  .replace(/(?:\/api)+$/i, '')

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
.embedding-db .search-results { margin:20px 0 30px; }.embedding-db .score { font-weight:700; color:#79a7ff; }
.embedding-db .query-preview { width:min(260px,100%); max-height:320px; object-fit:contain; display:block; margin:12px 0 18px; border:1px solid #384960; border-radius:10px; background:#0d1726; }
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
  const [searchFile, setSearchFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [ambiguousResult, setAmbiguousResult] = useState(null)
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchMessage, setSearchMessage] = useState('')

  useEffect(() => {
    if (!searchFile) { setPreviewUrl(''); return undefined }
    const url = URL.createObjectURL(searchFile)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [searchFile])

  async function searchByImage(event) {
    event.preventDefault()
    if (!searchFile) return
    setSearchLoading(true)
    setSearchMessage('')
    setSearchResults([])
    setAmbiguousResult(null)
    const body = new FormData()
    body.append('file', searchFile)
    try {
      const response = await fetch(`${API_BASE}/api/embeddings/search-image?top_k=3`, {
        method: 'POST', body,
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || `API ${response.status}`)
      setSearchResults(data.matches)
      setAmbiguousResult(data.status === 'AMBIGUOUS' ? data : null)
      const skipped = Object.values(data.skipped_records || {}).reduce((sum, count) => sum + count, 0)
      setSearchMessage(data.status === 'AMBIGUOUS' ? '' : data.status === 'MATCH'
        ? `MATCH — พบ ${data.matches.length} identity${data.person_count > 1 ? ' (เลือกบุคคลที่กรอบใหญ่ที่สุด)' : ''}`
        : data.status === 'NO_PERSON_DETECTED' ? 'ไม่พบบุคคลในภาพ'
            : `UNKNOWN / NO MATCH${skipped ? ` — ข้าม ${skipped} record ที่โมเดลหรือข้อมูลไม่เข้ากัน` : ''}`)
    } catch (e) { setSearchMessage(`ค้นหาไม่สำเร็จ: ${e.message}`) }
    finally { setSearchLoading(false) }
  }

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
    <form className="db-filters" onSubmit={searchByImage}>
      <label>ค้นหาบุคคลจากรูป<input type="file" accept="image/*" onChange={e => setSearchFile(e.target.files?.[0] || null)} /></label>
      <button type="submit" disabled={!searchFile || searchLoading}>{searchLoading ? 'กำลังตรวจสอบ...' : 'ตรวจสอบกับ Database'}</button>
    </form>
    {previewUrl && <><p>รูปที่ใช้ตรวจสอบ</p><img className="query-preview" src={previewUrl} alt="รูปบุคคลที่เลือกเพื่อตรวจสอบ" /></>}
    {searchMessage && <p role="status">{searchMessage}</p>}
    {ambiguousResult && <AmbiguousCandidates result={ambiguousResult} />}
    {searchResults.length > 0 && <div className="db-scroll search-results"><table>
      <thead><tr><th>อันดับ (Top 3)</th><th>Session</th><th>Global ID</th><th>Similarity</th><th>วันที่</th><th>เวลา</th><th>กล้อง</th></tr></thead>
      <tbody>{searchResults.map((match, index) => <tr key={`${match.identity_session_id}:${match.global_id}`}><td>{index + 1}</td><td>{match.session_display_name}</td><td>{match.global_id}</td>
        <td className="score">{match.similarity.toFixed(4)}</td><td>{match.captured_date}</td>
        <td>{match.captured_time}</td><td>{match.camera_name || '—'}</td></tr>)}</tbody>
    </table></div>}
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
    <div className="db-scroll"><table><thead><tr><th>Session</th><th>Global ID</th><th>วันที่</th><th>เวลา</th><th>กล้อง</th><th>ขนาดเวกเตอร์</th><th>ข้อมูล</th><th>ลบ</th></tr></thead>
      <tbody>{result.items.map(row => <FragmentRow key={row.id} row={row} expanded={openVector === row.id}
        toggle={() => toggleVector(row.id)} remove={() => deleteRecord(row.id)} vector={vector} vectorLoading={vectorLoading} vectorError={vectorError} />)}</tbody></table></div>
    <div className="db-pager"><button type="button" disabled={page <= 1 || loading} onClick={() => setPage(p => p - 1)}>ก่อนหน้า</button>
      <span>หน้า {page} / {Math.max(1, Math.ceil(result.total / 25))}</span>
      <button type="button" disabled={page * 25 >= result.total || loading} onClick={() => setPage(p => p + 1)}>ถัดไป</button></div>
  </section>
}

export function AmbiguousCandidates({ result }) {
  if (result?.status !== 'AMBIGUOUS') return null
  return <div className="db-scroll search-results">
    <p role="status">AMBIGUOUS — พบ identity ที่ใกล้เคียงกัน ยังไม่สามารถยืนยันว่าเป็นบุคคลใด</p>
    <p>Top-1 / Top-2 margin: {result.margin.toFixed(4)} · Required margin: {result.required_margin.toFixed(4)}</p>
    <table><thead><tr><th>อันดับ</th><th>Session</th><th>Global ID</th><th>Similarity</th></tr></thead>
      <tbody>{result.candidates.map((candidate, index) =>
        <tr key={`${candidate.identity_session_id}:${candidate.global_id}`}>
          <td>{index + 1}</td><td>{candidate.session_display_name}</td>
          <td>{candidate.global_id}</td><td className="score">{candidate.similarity.toFixed(4)}</td>
        </tr>)}</tbody>
    </table>
  </div>
}

function FragmentRow({ row, expanded, toggle, remove, vector, vectorLoading, vectorError }) {
  return <>
    <tr><td>{row.session_display_name}</td><td>{row.global_id}</td><td>{row.captured_date}</td><td>{row.captured_time}</td>
      <td>{row.camera_name || '—'}</td><td>{row.embedding_dim}</td>
      <td><button type="button" onClick={toggle}>{expanded ? 'ซ่อน embedding' : 'ดู embedding'}</button></td>
      <td><button type="button" onClick={remove}>ลบ</button></td></tr>
    {expanded && <tr><td colSpan="8"><strong>Embedding ของ {row.session_display_name} / Global ID {row.global_id}</strong>
      {vectorLoading ? <p>กำลังโหลด...</p> : vectorError ? <p role="alert">{vectorError}</p>
        : vector && <pre className="db-vector">{vector.map((value, index) => `${index}: ${value}`).join('\n')}</pre>}
    </td></tr>}
  </>
}
