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
.embedding-db input,.embedding-db select,.embedding-db button { font:inherit; border-radius:8px; padding:9px 12px; border:1px solid #60708a; }
.embedding-db input,.embedding-db select { background:#1d2b40; color:white; color-scheme:dark; }
.embedding-db button { color:white; background:#315fc9; cursor:pointer; }
.embedding-db button:disabled { opacity:.45; cursor:default; }
.embedding-db .db-scroll { overflow-x:auto; }
.embedding-db table { border-collapse:collapse; width:100%; text-align:left; min-width:550px; }
.embedding-db th,.embedding-db td { padding:12px; border-bottom:1px solid #384960; }
.embedding-db th { color:#afc6f6; }.embedding-db .db-pager { display:flex; align-items:center; gap:12px; margin-top:18px; }
.embedding-db .db-vector { white-space:pre-wrap; overflow-wrap:anywhere; max-height:260px; overflow:auto; background:#0d1726; padding:12px; border-radius:8px; font-family:monospace; font-size:12px; }
.embedding-db .search-results { margin:20px 0 30px; }.embedding-db .score { font-weight:700; color:#79a7ff; }
.embedding-db .crop-thumbs { display:flex; gap:12px; flex-wrap:wrap; align-items:flex-start; }
.embedding-db .crop-card { margin:0; color:#afbdd2; font-size:12px; }
.embedding-db .crop-card img { width:110px; height:150px; object-fit:contain; display:block; margin-bottom:5px; border:1px solid #384960; border-radius:8px; background:#0d1726; }
.embedding-db .query-preview { width:min(260px,100%); max-height:320px; object-fit:contain; display:block; margin:12px 0 18px; border:1px solid #384960; border-radius:10px; background:#0d1726; }
.embedding-db .db-tabs { display:flex; gap:8px; margin-bottom:22px; border-bottom:1px solid #384960; padding-bottom:10px; }
.embedding-db .db-tabs button.active { background:#e9efff; color:#101827; }
.embedding-db.compact-mode > .compact-summary { position:relative; cursor:pointer; list-style:none; user-select:none; display:flex; align-items:center; gap:10px; padding-right:44px; min-height:34px; }
.embedding-db.compact-mode > .compact-summary::-webkit-details-marker { display:none; }
.embedding-db.compact-mode > .compact-summary::after { content:'+'; position:absolute; right:2px; top:50%; transform:translateY(-50%); width:34px; height:34px; display:grid; place-items:center; border-radius:9px; border:1px solid var(--border); background:rgba(15,23,42,.7); color:var(--text); font-size:27px; font-weight:700; line-height:1; }
.embedding-db.compact-mode[open] > .compact-summary::after { content:'−'; }
.embedding-db.compact-mode > .compact-summary h2 { margin:0; }
.embedding-db .map-thumb { width:150px; height:100px; object-fit:contain; background:#0d1726; border:1px solid #384960; border-radius:8px; }
`

export default function EmbeddingDatabase({ compact = false, compactMode = 'both', language = 'en' }) {
  const isTH = language === 'th';
  const compactStorageKey = compactMode === 'save' ? 'panel:id-save' : 'panel:image-compare'
  const [compactOpen, setCompactOpen] = useState(() => sessionStorage.getItem(compactStorageKey) === '1');

  const handleCompactToggle = (event) => {
    if (!compact) return;
    const nextOpen = event.currentTarget.open;
    setCompactOpen(nextOpen);
    sessionStorage.setItem(compactStorageKey, nextOpen ? '1' : '0');
  };
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
  const [databaseTab, setDatabaseTab] = useState('embeddings')
  const [mapRows, setMapRows] = useState([])
  const [mapError, setMapError] = useState('')
  const [mapRefreshKey, setMapRefreshKey] = useState(0)
  const [sourceType, setSourceType] = useState('')
  const [mapDate, setMapDate] = useState('')
  const [mapLocation, setMapLocation] = useState('')
  const [mapRoom, setMapRoom] = useState('')

  useEffect(() => {
    if (!searchFile) { setPreviewUrl(''); return undefined }
    const url = URL.createObjectURL(searchFile)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [searchFile])

  function finishImageCompare() {
    setSearchFile(null)
    setPreviewUrl('')
    setSearchResults([])
    setAmbiguousResult(null)
    setSearchMessage('')
  }

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
    } catch (e) { setSearchMessage(`${isTH ? 'ค้นหาไม่สำเร็จ' : 'Search failed'}: ${e.message}`) }
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
    } catch (e) { setVectorError(`${isTH ? 'อ่าน embedding ไม่สำเร็จ' : 'Failed to load embedding'}: ${e.message}`) }
    finally { setVectorLoading(false) }
  }

  async function refreshSelected() {
    const response = await fetch(`${API_BASE}/api/embeddings/selected-ids`)
    if (!response.ok) throw new Error(`API ${response.status}`)
    setSelected((await response.json()).selected_ids)
  }

  useEffect(() => {
    const refresh = () => refreshSelected().catch(e => setError(`${isTH ? 'อ่าน ID ที่เลือกไม่สำเร็จ' : 'Failed to load selected IDs'}: ${e.message}`))
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
    } catch (e) { setError(`${isTH ? 'เปลี่ยนรายการ ID ไม่สำเร็จ' : 'Failed to update selected IDs'}: ${e.message}`) }
  }

  async function deleteRecord(recordId) {
    if (!window.confirm(isTH ? `ลบข้อมูล embedding รายการ #${recordId} ถาวรหรือไม่?` : `Permanently delete embedding #${recordId}?`)) return
    try {
      const response = await fetch(`${API_BASE}/api/embeddings/${recordId}`, { method: 'DELETE' })
      if (!response.ok) throw new Error(`API ${response.status}`)
      if (openVector === recordId) setOpenVector(null)
      setRefreshKey(key => key + 1)
    } catch (e) { setError(`${isTH ? 'ลบข้อมูลไม่สำเร็จ' : 'Delete failed'}: ${e.message}`) }
  }

  useEffect(() => {
    const controller = new AbortController()
    const params = new URLSearchParams({ page: String(page), page_size: '25' })
    if (filters.id) params.set('global_id', filters.id)
    if (filters.date) params.set('captured_date', filters.date)
    if (sourceType) params.set('source_type', sourceType)
    const load = () => fetch(`${API_BASE}/api/embeddings?${params}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`API ${response.status}`)
        return response.json()
      })
      .then(data => { setResult(data); setError('') })
      .catch((e) => { if (e.name !== 'AbortError') setError(`${isTH ? 'อ่านข้อมูลไม่สำเร็จ' : 'Failed to load data'}: ${e.message}`) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    setLoading(true)
    load()
    const interval = setInterval(load, 5000)
    return () => { clearInterval(interval); controller.abort() }
  }, [filters, page, refreshKey, sourceType])

  useEffect(() => {
    if (compact || databaseTab !== 'maps') return undefined
    const controller = new AbortController()
    fetch(`${API_BASE}/api/map-database`, { signal: controller.signal })
      .then(async response => {
        if (!response.ok) throw new Error(`API ${response.status}`)
        return response.json()
      })
      .then(data => { setMapRows(data.items || []); setMapError('') })
      .catch(e => { if (e.name !== 'AbortError') setMapError(`${isTH ? 'อ่าน Map Database ไม่สำเร็จ' : 'Failed to load Map Database'}: ${e.message}`) })
    return () => controller.abort()
  }, [compact, databaseTab, mapRefreshKey])

  async function deleteMap(mapRef) {
    if (!window.confirm(isTH ? 'ลบ Map รายการนี้ถาวรหรือไม่?' : 'Permanently delete this map?')) return
    try {
      const response = await fetch(`${API_BASE}/api/floorplans/${encodeURIComponent(mapRef)}`, { method: 'DELETE' })
      const data = await response.json()
      if (!response.ok || !data.success) throw new Error(data.message || `API ${response.status}`)
      setMapRefreshKey(key => key + 1)
    } catch (e) { setMapError(`${isTH ? 'ลบ Map ไม่สำเร็จ' : 'Failed to delete map'}: ${e.message}`) }
  }

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
    setSourceType('')
    setRefreshKey(key => key + 1)
  }

  const filteredMapRows = mapRows.filter(row =>
    (!mapDate || row.map_date === mapDate) &&
    (!mapLocation || String(row.location || '').toLowerCase().includes(mapLocation.toLowerCase())) &&
    (!mapRoom || String(row.room || '').toLowerCase().includes(mapRoom.toLowerCase()))
  )

  if (!compact && databaseTab === 'maps') return <section className="embedding-db">
    <style>{css}</style>
    <div className="db-tabs">
      <button type="button" onClick={() => setDatabaseTab('embeddings')}>{isTH ? 'ฐานข้อมูล Embedding' : 'Embedding Database'}</button>
      <button type="button" className="active" onClick={() => setDatabaseTab('maps')}>{isTH ? 'ฐานข้อมูล Map' : 'Map Database'}</button>
    </div>
    <h2>{isTH ? 'ฐานข้อมูล Map' : 'Map Database'}</h2>
    <div className="db-filters">
      <label>{isTH ? 'วันที่' : 'Date'}<input type="date" value={mapDate} onChange={e => setMapDate(e.target.value)} /></label>
      <label>{isTH ? 'สถานที่' : 'Location'}<input type="text" value={mapLocation} onChange={e => setMapLocation(e.target.value)} placeholder={isTH ? 'ทั้งหมด' : 'All'} /></label>
      <label>{isTH ? 'ห้อง' : 'Room'}<input type="text" value={mapRoom} onChange={e => setMapRoom(e.target.value)} placeholder={isTH ? 'ทั้งหมด' : 'All'} /></label>
      <button type="button" onClick={() => { setMapDate(''); setMapLocation(''); setMapRoom('') }}>{isTH ? 'ล้างตัวกรอง' : 'Clear Filters'}</button>
    </div>
    <p role="status">{mapError || (isTH ? `พบ ${filteredMapRows.length} รายการ` : `${filteredMapRows.length} records`)}</p>
    <div className="db-scroll"><table>
      <thead><tr><th>{isTH ? 'วันที่' : 'Date'}</th><th>{isTH ? 'สถานที่' : 'Location'}</th><th>{isTH ? 'ห้อง' : 'Room'}</th><th>{isTH ? 'รูปแมพ' : 'Map Image'}</th><th>{isTH ? 'ลบ' : 'Delete'}</th></tr></thead>
      <tbody>{filteredMapRows.map(row => <tr key={row.map_ref}>
        <td>{row.map_date}</td><td>{row.location}</td><td>{row.room}</td>
        <td><a href={`${API_BASE}/api/map-database/image?ref=${encodeURIComponent(row.map_ref)}`} target="_blank" rel="noreferrer"><img className="map-thumb" src={`${API_BASE}/api/map-database/image?ref=${encodeURIComponent(row.map_ref)}`} alt={`Map ${row.location} ${row.room}`} /></a></td>
        <td><button type="button" onClick={() => deleteMap(row.map_ref)}>{isTH ? 'ลบ' : 'Delete'}</button></td>
      </tr>)}</tbody>
    </table></div>
  </section>

  const Root = compact ? 'details' : 'section'

  return <Root
    className={`embedding-db ${compact ? 'compact-mode' : ''}`}
    {...(compact ? { open: compactOpen, onToggle: handleCompactToggle } : {})}
  >
    <style>{css}</style>
    {compact && <summary className="compact-summary"><h2>{compactMode === 'save' ? (isTH ? 'บันทึก ID' : 'Save ID') : (isTH ? 'เปรียบเทียบรูป' : 'Compare Image')}</h2></summary>}
    {!compact && <div className="db-tabs">
      <button type="button" className="active" onClick={() => setDatabaseTab('embeddings')}>{isTH ? 'ฐานข้อมูล Embedding' : 'Embedding Database'}</button>
      <button type="button" onClick={() => setDatabaseTab('maps')}>{isTH ? 'ฐานข้อมูล Map' : 'Map Database'}</button>
    </div>}
    {!compact && <h2>{isTH ? 'รายการ Embedding' : 'Saved Embeddings'}</h2>}
    <p>{compact
      ? (compactMode === 'save' ? (isTH ? 'เลือก Global ID ที่ต้องการเก็บ' : 'Choose a Global ID to save') : (isTH ? 'ค้นหาบุคคลจากรูปและเปรียบเทียบกับ Database' : 'Find a person by image and compare with the database'))
      : (isTH ? 'ข้อมูลที่บันทึกแยกตาม Global ID และวันที่' : 'Saved records by Global ID and date')}</p>
    {(!compact || compactMode !== 'save') && <>
    <form className="db-filters" onSubmit={searchByImage}>
      <label>{isTH ? 'ค้นหาบุคคลจากรูป' : 'Person image'}<input type="file" accept="image/*" onChange={e => setSearchFile(e.target.files?.[0] || null)} /></label>
      <button type="submit" disabled={!searchFile || searchLoading}>{searchLoading ? (isTH ? 'กำลังตรวจสอบ...' : 'Checking...') : (isTH ? 'ตรวจสอบกับ Database' : 'Compare with Database')}</button>
    </form>
    {previewUrl && <><p>{isTH ? 'รูปที่ใช้ตรวจสอบ' : 'Selected image'}</p><img className="query-preview" src={previewUrl} alt="รูปบุคคลที่เลือกเพื่อตรวจสอบ" /></>}
    {searchMessage && <p role="status">{searchMessage}</p>}
    {ambiguousResult && <AmbiguousCandidates result={ambiguousResult} language={language} />}
    {searchResults.length > 0 && <div className="db-scroll search-results"><table>
      <thead><tr><th>{isTH ? 'อันดับ (Top 3)' : 'Rank (Top 3)'}</th><th>{isTH ? 'รูป' : 'Image'}</th><th>Session</th><th>Global ID</th><th>Similarity</th><th>{isTH ? 'วันที่' : 'Date'}</th><th>{isTH ? 'เวลา' : 'Time'}</th><th>{isTH ? 'กล้อง' : 'Camera'}</th></tr></thead>
      <tbody>{searchResults.map((match, index) => <tr key={`${match.identity_session_id}:${match.global_id}`}><td>{index + 1}</td><td>{match.id ? <img style={{width:70,height:95,objectFit:'contain'}} src={`${API_BASE}/api/embeddings/${match.id}/crops/1/image`} alt={`Global ID ${match.global_id}`} /> : '—'}</td><td>{match.session_display_name}</td><td>{match.global_id}</td>
        <td className="score">{match.similarity.toFixed(4)}</td><td>{match.captured_date}</td>
        <td>{match.captured_time}</td><td>{match.camera_name || '—'}</td></tr>)}</tbody>
    </table></div>}
    {(searchFile || previewUrl || searchMessage || ambiguousResult || searchResults.length > 0) && (
      <div style={{display: 'flex', justifyContent: 'flex-end', marginTop: 12, marginBottom: 12}}>
        <button type="button" onClick={finishImageCompare}>{isTH ? 'เสร็จสิ้น' : 'Done'}</button>
      </div>
    )}
    </>}
    {compact && compactMode !== 'compare' && <>
      <form className="db-filters" onSubmit={e => { e.preventDefault(); if (newId && Number(newId) > 0) changeSelection(newId) }}>
        <label>{isTH ? 'Global ID ที่ต้องการเก็บ' : 'Global ID to save'}<input type="number" min="1" step="1" value={newId} onChange={e => setNewId(e.target.value)} placeholder={isTH ? 'เช่น 3' : 'e.g., 3'} /></label>
        <button type="submit" disabled={!newId || Number(newId) < 1}>{isTH ? 'เริ่มเก็บ ID นี้' : 'Start Saving ID'}</button>
      </form>
      <p>{isTH ? 'รอบันทึก: ' : 'Waiting to save: '}{selected.length ? selected.map(gid => <span key={gid} style={{ marginRight: 12 }}>ID {gid}</span>) : (isTH ? 'ยังไม่ได้เลือก ID' : 'No ID selected')}{isTH ? ' · เมื่อบันทึกได้แล้วระบบจะหยุดเก็บ ID นั้นอัตโนมัติ' : ' · Saving stops automatically after the ID is stored.'}</p>
    </>}
    {!compact && <>
    <form className="db-filters" onSubmit={search}>
      <label>Global ID<input type="number" min="1" step="1" value={id} onChange={e => setId(e.target.value)} placeholder={isTH ? 'ทั้งหมด' : 'All'} /></label>
      <label>{isTH ? 'วันที่' : 'Date'}<input type="date" value={date} onChange={e => setDate(e.target.value)} /></label>
      <label>{isTH ? 'แหล่งที่มา' : 'Source'}<select value={sourceType} onChange={e => { setSourceType(e.target.value); setPage(1) }}><option value="">{isTH ? 'ทั้งหมด' : 'All'}</option><option value="live">Realtime</option><option value="video">Video</option></select></label>
      <button type="submit">{isTH ? 'ค้นหา' : 'Search'}</button><button type="button" onClick={clear}>{isTH ? 'ล้างตัวกรอง' : 'Clear Filters'}</button>
      <button type="button" onClick={() => setRefreshKey(key => key + 1)}>{isTH ? 'รีเฟรชรายการ' : 'Refresh'}</button>
    </form>
    <p role="status">{error || (loading ? (isTH ? 'กำลังโหลด...' : 'Loading...') : (isTH ? `พบ ${result.total} รายการ` : `${result.total} records`))}</p>
    <div className="db-scroll"><table><thead><tr><th>Session</th><th>Global ID</th><th>{isTH ? 'วันที่' : 'Date'}</th><th>{isTH ? 'เวลา' : 'Time'}</th><th>{isTH ? 'กล้อง' : 'Camera'}</th><th>{isTH ? 'แหล่งที่มา' : 'Source'}</th><th>{isTH ? 'ขนาดเวกเตอร์' : 'Vector Size'}</th><th>{isTH ? 'รูป' : 'Images'}</th><th>{isTH ? 'ข้อมูล' : 'Vector'}</th><th>{isTH ? 'ลบ' : 'Delete'}</th></tr></thead>
      <tbody>{result.items.map(row => <FragmentRow key={row.id} row={row} expanded={openVector === row.id}
        toggle={() => toggleVector(row.id)} remove={() => deleteRecord(row.id)} vector={vector} vectorLoading={vectorLoading} vectorError={vectorError} language={language} />)}</tbody></table></div>
    <div className="db-pager"><button type="button" disabled={page <= 1 || loading} onClick={() => setPage(p => p - 1)}>{isTH ? 'ก่อนหน้า' : 'Previous'}</button>
      <span>{isTH ? 'หน้า' : 'Page'} {page} / {Math.max(1, Math.ceil(result.total / 25))}</span>
      <button type="button" disabled={page * 25 >= result.total || loading} onClick={() => setPage(p => p + 1)}>{isTH ? 'ถัดไป' : 'Next'}</button></div>
    </>}
  </Root>
}

export function AmbiguousCandidates({ result, language = 'en' }) {
  const isTH = language === 'th';
  if (result?.status !== 'AMBIGUOUS') return null
  return <div className="db-scroll search-results">
    <p role="status">{isTH ? 'พบ identity ที่ใกล้เคียงกัน ยังไม่สามารถยืนยันว่าเป็นบุคคลใด' : 'Similar identities found. The person cannot be confirmed yet.'}</p>
    <table><thead><tr><th>{isTH ? 'อันดับ' : 'Rank'}</th><th>{isTH ? 'รูป' : 'Image'}</th><th>Session</th><th>Global ID</th><th>Similarity</th><th>{isTH ? 'วันที่' : 'Date'}</th></tr></thead>
      <tbody>{result.candidates.map((candidate, index) =>
        <tr key={`${candidate.identity_session_id}:${candidate.global_id}`}>
          <td>{index + 1}</td><td>{candidate.id ? <img style={{width:70,height:95,objectFit:'contain'}} src={`${API_BASE}/api/embeddings/${candidate.id}/crops/1/image`} alt={`Global ID ${candidate.global_id}`} /> : '—'}</td><td>{candidate.session_display_name}</td>
          <td>{candidate.global_id}</td><td className="score">{candidate.similarity.toFixed(4)}</td><td>{candidate.captured_date || '—'}</td>
        </tr>)}</tbody>
    </table>
  </div>
}

function FragmentRow({ row, expanded, toggle, remove, vector, vectorLoading, vectorError, language = 'en' }) {
  const isTH = language === 'th';
  const [showCrops, setShowCrops] = useState(false)
  const [crops, setCrops] = useState([])
  const [cropError, setCropError] = useState('')

  async function toggleCrops() {
    if (showCrops) { setShowCrops(false); return }
    if (!crops.length) {
      try {
        const response = await fetch(`${API_BASE}/api/embeddings/${row.id}/crops`)
        if (!response.ok) throw new Error(`API ${response.status}`)
        setCrops((await response.json()).crops || [])
        setCropError('')
      } catch (e) { setCropError(`${isTH ? 'อ่านรูปไม่สำเร็จ' : 'Failed to load images'}: ${e.message}`) }
    }
    setShowCrops(true)
  }

  return <>
    <tr><td>{row.session_display_name}</td><td>{row.global_id}</td><td>{row.captured_date}</td><td>{row.captured_time}</td>
      <td>{row.camera_name || '—'}</td><td>{row.source_type === 'live' ? 'Realtime' : row.source_type === 'video' ? 'Video' : 'Unknown'}</td><td>{row.embedding_dim}</td>
      <td><button type="button" onClick={toggleCrops}>{showCrops ? (isTH ? 'ซ่อนรูป' : 'Hide') : (isTH ? 'ดูรูป' : 'View')}</button></td>
      <td><button type="button" onClick={toggle}>{expanded ? (isTH ? 'ซ่อน embedding' : 'Hide') : (isTH ? 'ดู embedding' : 'View')}</button></td>
      <td><button type="button" onClick={remove}>{isTH ? 'ลบ' : 'Delete'}</button></td></tr>
    {showCrops && <tr><td colSpan="10"><strong>{isTH ? 'Crop ก่อนเข้า OSNet' : 'Crops before OSNet'} — Global ID {row.global_id}</strong>
      {cropError ? <p role="alert">{cropError}</p> : crops.length ? <div className="crop-thumbs">
        {crops.map(crop => <figure className="crop-card" key={crop.crop_index}>
          <img src={`${API_BASE}${crop.image_url}`} alt={`Global ID ${row.global_id} crop ${crop.crop_index}`} loading="lazy" />
          <figcaption>{isTH ? 'รูป' : 'Image'} {crop.crop_index} · Frame {crop.frame_index}</figcaption>
        </figure>)}
      </div> : <p>{isTH ? 'ไม่มีรูป crop สำหรับรายการนี้' : 'No crop images for this record'}</p>}
    </td></tr>}
    {expanded && <tr><td colSpan="10"><strong>{isTH ? 'Embedding ของ' : 'Embedding for'} {row.session_display_name} / Global ID {row.global_id}</strong>
      {vectorLoading ? <p>{isTH ? 'กำลังโหลด...' : 'Loading...'}</p> : vectorError ? <p role="alert">{vectorError}</p>
        : vector && <pre className="db-vector">{vector.map((value, index) => `${index}: ${value}`).join('\n')}</pre>}
    </td></tr>}
  </>
}
