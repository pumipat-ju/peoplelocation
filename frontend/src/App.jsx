import { useState, useEffect } from 'react';
import { Camera, Map, Upload, Video, Trash2, AlertCircle, CheckCircle2, Crosshair, Play, Pause } from 'lucide-react';
import './index.css';
import CalibrationModal from './CalibrationModal';
import VideoUploader from './VideoUploader';
import EmbeddingDatabase from './EmbeddingDatabase';

const API_URL = 'http://localhost:8899/api';
const HOST_URL = 'http://localhost:8899';


function CollapsiblePanel({ storageKey, title, children, className = "glass-panel" }) {
  const [open, setOpen] = useState(() => sessionStorage.getItem(storageKey) === '1');
  const handleToggle = (event) => {
    const next = event.currentTarget.open;
    setOpen(next);
    sessionStorage.setItem(storageKey, next ? '1' : '0');
  };
  return (
    <details className={className} open={open} onToggle={handleToggle}>
      <summary className="section-title" style={{cursor:'pointer', userSelect:'none', listStyle:'none', display:'flex', alignItems:'center', justifyContent:'space-between', gap:'0.75rem', margin:0, fontWeight:700}}>
        <span style={{display:'inline-flex', alignItems:'center', gap:'0.5rem'}}>{title}</span>
        <span aria-hidden="true" style={{display:'inline-flex', alignItems:'center', justifyContent:'center', width:'34px', height:'34px', flex:'0 0 34px', borderRadius:'9px', background:'var(--primary-soft, #eaf3ff)', color:'var(--primary, #3478dc)', fontSize:'1.65rem', fontWeight:700, lineHeight:1}}>{open ? '−' : '+'}</span>
      </summary>
      <div style={{marginTop:'1rem'}}>{children}</div>
    </details>
  );
}

const mapLabel = (ref) => {
  const parts = String(ref || '').split('|');
  return parts.length === 3 ? `${parts[1]} / ${parts[2]} (${parts[0]})` : String(ref || '');
};

export default function App() {
  const [status, setStatus] = useState({ cameras: {}, floorplan_exists: false });
  const [alert, setAlert] = useState(null);
  const [loading, setLoading] = useState(true);
  const [calibratingCamera, setCalibratingCamera] = useState(null);
  const [selectedVideos, setSelectedVideos] = useState([]);
  const [playbackLoading, setPlaybackLoading] = useState(false);
  const [selectedFloorplans, setSelectedFloorplans] = useState([]);
  const [mapStreamVersions, setMapStreamVersions] = useState({});
  const [activePage, setActivePage] = useState(() => {
    const savedPage = sessionStorage.getItem('ui:active-page');
    return ['realtime', 'video', 'database'].includes(savedPage) ? savedPage : 'realtime';
  });
  const [language, setLanguage] = useState(() => sessionStorage.getItem('ui:language') || 'en');
  const [openLocations, setOpenLocations] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem('map:open-locations') || '{}'); }
    catch { return {}; }
  });
  const [mapLocationPickerOpen, setMapLocationPickerOpen] = useState(() => ({
    realtime: sessionStorage.getItem('panel:realtime-map-locations') === '1',
    video: sessionStorage.getItem('panel:video-map-locations') === '1',
  }));


  const tr = (en, th) => language === 'th' ? th : en;
  const changeLanguage = (next) => {
    setLanguage(next);
    sessionStorage.setItem('ui:language', next);
  };
  const toggleLocation = (location) => {
    setOpenLocations(current => {
      const next = { ...current, [location]: !current[location] };
      sessionStorage.setItem('map:open-locations', JSON.stringify(next));
      return next;
    });
  };


  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/status`);
      const data = await res.json();
      setStatus(data);
      setSelectedVideos((current) => current.filter(
        (name) => data.cameras?.[name]?.source_type === 'video'
      ));
    } catch (err) {
      console.error("Failed to fetch status:", err);
      showAlert("Cannot connect to backend server", "error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const names = status.floorplans || [];
    setSelectedFloorplans((current) => {
      const available = current.filter(name => names.includes(name));
      return available.length > 0 ? available : names.slice(0, 1);
    });
  }, [JSON.stringify(status.floorplans || [])]);

  const toggleFloorplan = (name) => {
    setSelectedFloorplans((current) => {
      if (current.includes(name)) {
        return current.filter(item => item !== name);
      }

      // Force React/browser to create a completely new MJPEG connection
      // whenever a hidden floorplan is shown again.
      setMapStreamVersions((versions) => ({
        ...versions,
        [name]: (versions[name] || 0) + 1,
      }));
      return [...current, name];
    });
  };

  const showAlert = (message, type = "error") => {
    setAlert({ message, type });
    setTimeout(() => setAlert(null), 5000);
  };

  const handleUploadMap = async (e) => {
    e.preventDefault();
    const form = e.currentTarget;
    const formData = new FormData(form);
    try {
      const res = await fetch(`${API_URL}/upload_floorplan`, { method: 'POST', body: formData });
      const data = await res.json();
      showAlert(data.message, data.success ? "success" : "error");
      if (data.success) {
        form.reset();
        fetchStatus();
      }
    } catch (err) {
      showAlert("Upload failed", "error");
    }
  };

  const handleDeleteFloorplan = async (name) => {
    if (!confirm(`ต้องการลบ Floorplan "${name}" หรือไม่? การลบไม่สามารถย้อนกลับได้`)) return;
    try {
      const res = await fetch(`${API_URL}/floorplans/${encodeURIComponent(name)}`, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok || !data.success) {
        const cameraText = data.cameras?.length ? ` (ใช้งานโดย: ${data.cameras.join(', ')})` : '';
        showAlert(`${data.message || 'ลบ Floorplan ไม่สำเร็จ'}${cameraText}`, 'error');
        return;
      }
      setSelectedFloorplans(current => current.filter(item => item !== name));
      showAlert(data.message || 'ลบ Floorplan สำเร็จ', 'success');
      fetchStatus();
    } catch (err) {
      showAlert('ลบ Floorplan ไม่สำเร็จ', 'error');
    }
  };

  const handleAddCamera = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    try {
      const res = await fetch(`${API_URL}/add_camera`, { method: 'POST', body: formData });
      const data = await res.json();
      showAlert(data.message, data.success ? "success" : "error");
      if (data.success) {
        fetchStatus();
        e.target.reset();
      }
    } catch (err) {
      showAlert("Failed to add camera", "error");
    }
  };

  const handleDeleteCamera = async (name, displayName = name) => {
    if (!confirm(`Are you sure you want to delete ${displayName}?`)) return;
    try {
      const res = await fetch(`${API_URL}/delete_camera/${name}`, { method: 'DELETE' });
      const data = await res.json();
      showAlert(data.message, data.success ? "success" : "error");
      if (data.success) fetchStatus();
    } catch (err) {
      showAlert("Failed to delete", "error");
    }
  };

  const videoNames = Object.entries(status.cameras)
    .filter(([, cam]) => cam.source_type === 'video')
    .map(([name]) => name);

  const allVideosSelected = videoNames.length > 0
    && videoNames.every((name) => selectedVideos.includes(name));

  const toggleVideoSelection = (name) => {
    setSelectedVideos((current) => current.includes(name)
      ? current.filter((item) => item !== name)
      : [...current, name]);
  };

  const toggleAllVideos = () => {
    setSelectedVideos(allVideosSelected ? [] : videoNames);
  };

  const handlePlayback = async (action, cameraNames = null) => {
    if (cameraNames && cameraNames.length === 0) {
      showAlert('Select at least one video clip', 'error');
      return;
    }

    setPlaybackLoading(true);

    try {
      const body = { action };
      if (cameraNames) body.camera_names = cameraNames;

      const res = await fetch(`${API_URL}/video_playback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();

      showAlert(data.message, data.success ? 'success' : 'error');

      if (data.success) {
        await fetchStatus();
      }
    } catch (err) {
      console.error('Failed to control video playback:', err);
      showAlert('Failed to control video playback', 'error');
    } finally {
      setPlaybackLoading(false);
    }
  };

  const realtimeCameras = Object.entries(status.cameras)
    .filter(([, cam]) => cam.source_type !== 'video');
  const videoCameras = Object.entries(status.cameras)
    .filter(([, cam]) => cam.source_type === 'video');

  const renderMapPanel = (title, sourceType) => {
    const groups = {};
    for (const name of (status.floorplans || [])) {
      const parts = String(name).split('|');
      const location = parts.length === 3 ? parts[1] : tr('Other', 'อื่น ๆ');
      const room = parts.length === 3 ? parts[2] : name;
      (groups[location] ||= []).push({ name, room });
    }

    return (
      <div className="glass-panel">
        <div className="section-title" style={{display:'flex', alignItems:'center', justifyContent:'flex-start', gap:'0.5rem', margin:0}}>
          <h2 style={{margin:0, font:'inherit', color:'inherit'}}>{title}</h2>
          <button type="button" onClick={() => {
              setMapLocationPickerOpen(current => {
                const nextOpen = !current[sourceType];
                sessionStorage.setItem(`panel:${sourceType}-map-locations`, nextOpen ? '1' : '0');
                return { ...current, [sourceType]: nextOpen };
              });
            }}
            aria-label={mapLocationPickerOpen[sourceType] ? tr('Hide locations', 'ซ่อนสถานที่') : tr('Show locations', 'แสดงสถานที่')}
            style={{border:0, cursor:'pointer', display:'inline-flex', alignItems:'center', justifyContent:'center', width:'34px', height:'34px', flex:'0 0 34px', borderRadius:'9px', background:'var(--primary-soft, #eaf3ff)', color:'var(--primary, #3478dc)', fontSize:'1.65rem', fontWeight:700, lineHeight:1}}>
            {mapLocationPickerOpen[sourceType] ? '−' : '+'}
          </button>
        </div>
        {(status.floorplans || []).length > 0 ? <>
          {mapLocationPickerOpen[sourceType] && (
          <div className="floorplan-picker">
            {Object.entries(groups)
              .sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }))
              .map(([location, rooms]) => (
                <div key={location} style={{marginBottom:'0.45rem'}}>
                  <div style={{display:'flex', alignItems:'center', justifyContent:'flex-start', gap:'0.5rem'}}>
                    <span style={{fontWeight:700, color:'inherit'}}>{location}</span>
                    <button type="button" onClick={() => toggleLocation(location)}
                      aria-label={openLocations[location] ? tr('Collapse location', 'พับสถานที่') : tr('Expand location', 'กางสถานที่')}
                      style={{border:0, cursor:'pointer', display:'inline-flex', alignItems:'center', justifyContent:'center', width:'34px', height:'34px', flex:'0 0 34px', borderRadius:'9px', background:'var(--primary-soft, #eaf3ff)', color:'var(--primary, #3478dc)', fontSize:'1.65rem', fontWeight:700, lineHeight:1}}>
                      {openLocations[location] ? '−' : '+'}
                    </button>
                  </div>
                  {openLocations[location] && (
                    <div style={{paddingLeft:'1.25rem', display:'grid', gridTemplateColumns:'repeat(10, max-content)', gap:'0.45rem 0.75rem', alignItems:'center'}}>
                      {rooms.sort((a,b) => a.room.localeCompare(b.room, undefined, {numeric:true, sensitivity:'base'})).map(({name, room}) => (
                        <div key={name} className="floorplan-option">
                          <label>
                            <input type="checkbox" checked={selectedFloorplans.includes(name)} onChange={() => toggleFloorplan(name)} />
                            {room}
                          </label>
                          <button type="button" className="btn-icon" title={`${tr('Delete', 'ลบ')} ${room}`}
                            onClick={() => handleDeleteFloorplan(name)} style={{color:'var(--danger)', border:'none', cursor:'pointer'}}>
                            <Trash2 size={16} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
          </div>
          )}
          {selectedFloorplans.length === 0 && <p style={{color:'var(--text-muted)'}}>{tr('Select a map to display', 'เลือก Map ที่ต้องการแสดง')}</p>}
          <div className="tracking-map-grid">
            {[...selectedFloorplans].sort((a,b) => a.localeCompare(b, undefined, {numeric:true, sensitivity:'base'})).map(name => {
              const streamVersion = mapStreamVersions[name] || 0;
              return <div key={`${name}-${streamVersion}`}>
                <h3 className="map-name">{mapLabel(name)}</h3>
                <div className="map-container">
                  <img key={`${name}-stream-${streamVersion}`}
                    src={`${API_URL}/global_map_feed?name=${encodeURIComponent(name)}&source_type=${encodeURIComponent(sourceType)}&v=${streamVersion}`}
                    alt={`Global Map ${name}`} />
                </div>
              </div>;
            })}
          </div>
        </> : <p style={{color:'var(--text-muted)', marginTop:'1rem'}}>{tr('No Floorplan Uploaded', 'ยังไม่มี Floorplan')}</p>}
      </div>
    );
  };

  const renderCameraGrid = (entries, emptyText) => (
    <div className="cameras-grid">
      {entries.map(([name, cam]) => (
        <div key={name} className="glass-panel camera-card animate-in">
          <div className="camera-header">
            <div className="camera-title">
              {cam.source_type === 'video' ? <Video size={18} /> : <Camera size={18} />}
              {cam.display_name || name}
            </div>
            <div style={{display: 'flex', gap: '0.5rem'}}>
              {cam.source_type === 'video' && (
                <>
                  <label className="video-card-select" title={`Select ${cam.display_name || name}`}>
                    <input
                      type="checkbox"
                      checked={selectedVideos.includes(name)}
                      onChange={() => toggleVideoSelection(name)}
                    />
                  </label>
                  <span className={`badge ${cam.is_playing ? 'active' : 'paused'}`}>
                    {cam.is_playing ? tr('Playing', 'กำลังเล่น') : tr('Paused', 'หยุดชั่วคราว')}
                  </span>
                </>
              )}
              {cam.has_processor && <span className="badge active">{tr('Calibrated', 'คาลิเบรตแล้ว')}</span>}
              <button
                onClick={() => setCalibratingCamera(name)}
                className="btn-icon"
                style={{color: 'var(--accent)', border: 'none', cursor: 'pointer'}}
                title="Calibrate"
              >
                <Crosshair size={18} />
              </button>
              <button
                onClick={() => handleDeleteCamera(name, cam.display_name || name)}
                className="btn-icon"
                style={{color: 'var(--danger)', border: 'none', cursor: 'pointer'}}
                title="Delete"
              >
                <Trash2 size={18} />
              </button>
            </div>
          </div>
          <div className="camera-stream">
            <img key={`${name}-${cam.source_instance_id || 'source'}`} src={`${API_URL}/video_feed/${encodeURIComponent(name)}?v=${encodeURIComponent(cam.source_instance_id || 'source')}`} alt={cam.display_name || name} />
          </div>
        </div>
      ))}
      {entries.length === 0 && !loading && (
        <div className="empty-state">{emptyText}</div>
      )}
    </div>
  );

  const mapUploadPanel = (
    <CollapsiblePanel storageKey="panel:global-map-upload" title={<><Map size={20} /> {tr('Global Map', 'แผนที่')}</>}>
      <form onSubmit={handleUploadMap}>
        <div className="form-group">
          <label>{tr('Location', 'สถานที่')}</label>
          <input type="text" name="location" className="form-control" required maxLength="120" placeholder={tr('e.g. Building A', 'เช่น อาคาร A')} />
        </div>
        <div className="form-group">
          <label>{tr('Room', 'ห้อง')}</label>
          <input type="text" name="room" className="form-control" required maxLength="120" placeholder={tr('e.g. Room 101', 'เช่น ห้อง 101')} />
        </div>
        <div className="form-group">
          <input type="file" name="file" accept="image/*" className="form-control" required />
        </div>
        <button type="submit" className="btn">
          <Upload size={18} /> {tr('Upload Floorplan', 'อัปโหลดแผนที่')}
        </button>
      </form>
    </CollapsiblePanel>
  );

  const realtimeCameraPanel = (
    <CollapsiblePanel storageKey="panel:add-camera" title={<><Camera size={20} /> {tr('Add Camera Stream', 'เพิ่มกล้อง')}</>}>
      <form onSubmit={handleAddCamera}>
        <div className="form-group">
          <label>{tr('Camera Name', 'ชื่อกล้อง')}</label>
          <input type="text" name="name" className="form-control" required placeholder="e.g., Cam1" />
        </div>
        <div className="form-group">
          <label>{tr('RTSP / HTTP URL', 'ที่อยู่ RTSP / HTTP')}</label>
          <input type="text" name="url" className="form-control" required placeholder="rtsp://..." />
        </div>
        <button type="submit" className="btn">{tr('Add Stream', 'เพิ่มกล้อง')}</button>
      </form>
    </CollapsiblePanel>
  );

  const playbackPanel = videoNames.length > 0 && (
    <div className="glass-panel playback-panel">
      <div className="playback-selection">
        <label className="video-select-label">
          <input type="checkbox" checked={allVideosSelected} onChange={toggleAllVideos} />
          {tr('Select all videos', 'เลือกวิดีโอทั้งหมด')}
        </label>
        <span className="selection-count">
          {selectedVideos.length} {tr('of', 'จาก')} {videoNames.length} {tr('selected', 'ที่เลือก')}
        </span>
      </div>
      <div className="playback-actions">
        <button type="button" className="btn playback-button"
          onClick={() => handlePlayback('play', selectedVideos)}
          disabled={playbackLoading || selectedVideos.length === 0}>
          <Play size={17} /> {tr('Play Selected', 'เล่นที่เลือก')}
        </button>
        <button type="button" className="btn playback-button secondary"
          onClick={() => handlePlayback('pause', selectedVideos)}
          disabled={playbackLoading || selectedVideos.length === 0}>
          <Pause size={17} /> {tr('Pause Selected', 'หยุดที่เลือก')}
        </button>
        <button type="button" className="btn playback-button"
          onClick={() => handlePlayback('play')} disabled={playbackLoading}>
          <Play size={17} /> {tr('Play All', 'เล่นทั้งหมด')}
        </button>
        <button type="button" className="btn playback-button secondary"
          onClick={() => handlePlayback('pause')} disabled={playbackLoading}>
          <Pause size={17} /> {tr('Pause All', 'หยุดทั้งหมด')}
        </button>
      </div>
    </div>
  );

  return (
    <div className="app-shell animate-in">
      <aside className="main-nav">
        <div className="nav-brand">
          <div style={{display:'flex', alignItems:'flex-start', justifyContent:'space-between', gap:'0.5rem'}}>
            <div><h1>People Location</h1><h1>Tracker</h1></div>
            <div style={{display:'flex', gap:'0.25rem'}}>
              <button type="button" onClick={() => changeLanguage('en')} aria-pressed={language === 'en'}
                style={{padding:'0.25rem 0.4rem', borderRadius:6, border:'1px solid rgba(255,255,255,.55)', cursor:'pointer', background:language === 'en' ? '#fff' : 'transparent', color:language === 'en' ? '#3478dc' : '#fff'}}>EN</button>
              <button type="button" onClick={() => changeLanguage('th')} aria-pressed={language === 'th'}
                style={{padding:'0.25rem 0.4rem', borderRadius:6, border:'1px solid rgba(255,255,255,.55)', cursor:'pointer', background:language === 'th' ? '#fff' : 'transparent', color:language === 'th' ? '#3478dc' : '#fff'}}>ไทย</button>
            </div>
          </div>
        </div>

        <nav className="nav-menu" aria-label="Main navigation">
          <button
            type="button"
            className={`nav-item ${activePage === 'realtime' ? 'active' : ''}`}
            onClick={() => { setActivePage('realtime'); sessionStorage.setItem('ui:active-page', 'realtime'); }}
          >
            <Camera size={20} />
            <span>{tr('Realtime', 'เรียลไทม์')}</span>
          </button>
          <button
            type="button"
            className={`nav-item ${activePage === 'video' ? 'active' : ''}`}
            onClick={() => { setActivePage('video'); sessionStorage.setItem('ui:active-page', 'video'); }}
          >
            <Video size={20} />
            <span>{tr('Video', 'วิดีโอ')}</span>
          </button>
          <button
            type="button"
            className={`nav-item ${activePage === 'database' ? 'active' : ''}`}
            onClick={() => { setActivePage('database'); sessionStorage.setItem('ui:active-page', 'database'); }}
          >
            <span className="nav-db-icon">DB</span>
            <span>{tr('Database', 'ฐานข้อมูล')}</span>
          </button>
        </nav>

        <div className="nav-status">
          <span className="status-dot" />
          <span>{tr('API Connected', 'เชื่อมต่อ API แล้ว')}</span>
        </div>
      </aside>

      <main className="page-content">
        {alert && (
          <div className={`alert ${alert.type === 'success' ? 'alert-success' : ''} animate-in`}>
            <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
              {alert.type === 'success' ? <CheckCircle2 size={20} /> : <AlertCircle size={20} />}
              <span>{alert.message}</span>
            </div>
          </div>
        )}

        {activePage === 'realtime' && (
          <section className="page-view">
            <div className="page-heading">
              <div>
                <h2>{tr('Realtime Tracking', 'ติดตามแบบเรียลไทม์')}</h2>
                <p>{tr('Track positions from realtime cameras', 'แสดงตำแหน่งจากกล้องแบบเรียลไทม์')}</p>
              </div>
            </div>
            <div className="workspace-grid">
              <aside className="control-column">
                {mapUploadPanel}
                {realtimeCameraPanel}
                <EmbeddingDatabase compact sourceType="realtime" language={language} />
              </aside>
              <div className="workspace-main">
                {renderMapPanel(tr('Realtime Tracking Map', 'แผนที่ติดตามแบบเรียลไทม์'), 'realtime')}
                {renderCameraGrid(realtimeCameras, tr('No realtime cameras added yet.', 'ยังไม่มีกล้องเรียลไทม์'))}
              </div>
            </div>
          </section>
        )}

        {activePage === 'video' && (
          <section className="page-view">
            <div className="page-heading">
              <div>
                <h2>{tr('Video Tracking', 'ติดตามจากวิดีโอ')}</h2>
                <p>{tr('Track positions from uploaded video files', 'ติดตามตำแหน่งจากไฟล์วิดีโอที่อัปโหลด')}</p>
              </div>
            </div>
            <div className="workspace-grid">
              <aside className="control-column">
                {mapUploadPanel}
                <VideoUploader
                  API_URL={API_URL}
                  language={language}
                  onSuccess={(msg) => { showAlert(msg, "success"); fetchStatus(); }}
                />
                <EmbeddingDatabase compact sourceType="video" language={language} />
              </aside>
              <div className="workspace-main">
                {renderMapPanel(tr('Video Tracking Map', 'แผนที่ติดตามจากวิดีโอ'), 'video')}
                {playbackPanel}
                {renderCameraGrid(videoCameras, tr('No video files added yet.', 'ยังไม่มีไฟล์วิดีโอ'))}
              </div>
            </div>
          </section>
        )}

        {activePage === 'database' && (
          <section className="page-view database-page">
            <div className="page-heading">
              <div>
                <h2>{tr('Database', 'ฐานข้อมูล')}</h2>
                <p>{tr('Search, compare, and manage embedding data', 'ค้นหา เปรียบเทียบ และจัดการข้อมูล Embedding')}</p>
              </div>
            </div>
            <EmbeddingDatabase language={language} />
          </section>
        )}
      </main>

      {calibratingCamera && (
        <CalibrationModal 
          camName={calibratingCamera}
          camDisplayName={status.cameras?.[calibratingCamera]?.display_name || calibratingCamera}
          sourceType={status.cameras?.[calibratingCamera]?.source_type || 'realtime'}
          language={language} 
          API_URL={API_URL} 
          onClose={() => setCalibratingCamera(null)} 
          onSuccess={(msg) => { showAlert(msg, "success"); fetchStatus(); }} 
        />
      )}
    </div>
  );
}
