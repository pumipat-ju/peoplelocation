import { useState, useEffect } from 'react';
import { Camera, Map, Upload, Video, Trash2, AlertCircle, CheckCircle2, Crosshair, Play, Pause } from 'lucide-react';
import './index.css';
import CalibrationModal from './CalibrationModal';
import VideoUploader from './VideoUploader';
import EmbeddingDatabase from './EmbeddingDatabase';

const API_URL = 'http://localhost:8899/api';
const HOST_URL = 'http://localhost:8899';

const mapLabel = (ref) => {
  const parts = String(ref || '').split('|');
  return parts.length === 3 ? `${parts[1]} / ${parts[2]} (${parts[0]})` : String(ref || '');
};

const mapParts = (ref) => {
  const parts = String(ref || '').split('|');
  return parts.length === 3
    ? { date: parts[0], location: parts[1], room: parts[2] }
    : { date: '', location: 'Other', room: String(ref || '') };
};

function PersistentDetails({ storageKey, className, children }) {
  const [open, setOpen] = useState(() => sessionStorage.getItem(`panel:${storageKey}`) === '1');

  const handleToggle = (event) => {
    const nextOpen = event.currentTarget.open;
    setOpen(nextOpen);
    sessionStorage.setItem(`panel:${storageKey}`, nextOpen ? '1' : '0');
  };

  return (
    <details className={className} open={open} onToggle={handleToggle}>
      {children}
    </details>
  );
}

export default function App() {
  const [status, setStatus] = useState({ cameras: {}, floorplan_exists: false });
  const [alert, setAlert] = useState(null);
  const [loading, setLoading] = useState(true);
  const [calibratingCamera, setCalibratingCamera] = useState(null);
  const [selectedVideos, setSelectedVideos] = useState([]);
  const [playbackLoading, setPlaybackLoading] = useState(false);
  const [selectedFloorplans, setSelectedFloorplans] = useState([]);
  const [mapStreamVersions, setMapStreamVersions] = useState({});
  const [activePage, setActivePage] = useState('realtime');
  const [language, setLanguage] = useState(() => localStorage.getItem('ui-language') || 'en');
  const isTH = language === 'th';
  const setUILanguage = (next) => {
    setLanguage(next);
    localStorage.setItem('ui-language', next);
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
    if (!confirm(isTH ? `ต้องการลบ Map "${name}" หรือไม่? การลบไม่สามารถย้อนกลับได้` : `Delete map "${name}"? This cannot be undone.`)) return;
    try {
      const res = await fetch(`${API_URL}/floorplans/${encodeURIComponent(name)}`, { method: 'DELETE' });
      const data = await res.json();
      if (!res.ok || !data.success) {
        const cameraText = data.cameras?.length ? (isTH ? ` (ใช้งานโดย: ${data.cameras.join(', ')})` : ` (used by: ${data.cameras.join(', ')})`) : '';
        showAlert(`${data.message || (isTH ? 'ลบ Map ไม่สำเร็จ' : 'Failed to delete map')}${cameraText}`, 'error');
        return;
      }
      setSelectedFloorplans(current => current.filter(item => item !== name));
      showAlert(data.message || (isTH ? 'ลบ Map สำเร็จ' : 'Map deleted'), 'success');
      fetchStatus();
    } catch (err) {
      showAlert(isTH ? 'ลบ Map ไม่สำเร็จ' : 'Failed to delete map', 'error');
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

  const handleDeleteCamera = async (name) => {
    if (!confirm(`Are you sure you want to delete ${name}?`)) return;
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

  const renderMapPanel = (title) => (
    <div className="glass-panel">
      <h2 className="section-title">{title}</h2>
      {(status.floorplans || []).length > 0 ? <>
        <PersistentDetails storageKey="floorplan-list" className="floorplan-collapse">
          <summary className="collapsible-summary floorplan-summary">
            Map
          </summary>
          <div className="floorplan-picker floorplan-groups">
            {Object.entries(
              status.floorplans.reduce((groups, name) => {
                const info = mapParts(name);
                if (!groups[info.location]) groups[info.location] = [];
                groups[info.location].push({ name, ...info });
                return groups;
              }, {})
            )
              .sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }))
              .map(([location, maps]) => (
                <PersistentDetails key={location} storageKey={`floorplan-location:${location}`} className="floorplan-location">
                  <summary className="collapsible-summary floorplan-location-summary">
                    <strong>{location}</strong>
                  </summary>
                  <div className="floorplan-rooms">
                    {maps
                      .sort((a, b) => a.room.localeCompare(b.room, undefined, { numeric: true, sensitivity: 'base' }))
                      .map(({ name, room, date }) => (
                        <div key={name} className="floorplan-option floorplan-room">
                          <label>
                            <input
                              type="checkbox"
                              checked={selectedFloorplans.includes(name)}
                              onChange={() => toggleFloorplan(name)}
                            />
                            <span>{room}{date ? ` (${date})` : ''}</span>
                          </label>
                          <button
                            type="button"
                            className="btn-icon"
                            title={`${isTH ? 'ลบ' : 'Delete'} ${mapLabel(name)}`}
                            onClick={() => handleDeleteFloorplan(name)}
                            style={{color: 'var(--danger)', border: 'none', cursor: 'pointer'}}
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      ))}
                  </div>
                </PersistentDetails>
              ))}
          </div>
        </PersistentDetails>
        {selectedFloorplans.length === 0 && (
          <p style={{color: 'var(--text-muted)'}}>{isTH ? 'เลือก Map ที่ต้องการแสดง' : 'Select a map to display'}</p>
        )}
        <div className="tracking-map-grid">
          {[...selectedFloorplans]
            .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }))
            .map(name => {
              const streamVersion = mapStreamVersions[name] || 0;
              return (
                <div key={`${name}-${streamVersion}`}>
                  <h3 className="map-name">{mapLabel(name)}</h3>
                  <div className="map-container">
                    <img
                      key={`${name}-stream-${streamVersion}`}
                      src={`${API_URL}/global_map_feed?name=${encodeURIComponent(name)}&v=${streamVersion}`}
                      alt={`Global Map ${name}`}
                    />
                  </div>
                </div>
              );
            })}
        </div>
      </> : (
        <p style={{color: 'var(--text-muted)'}}>{isTH ? 'ยังไม่มี Map' : 'No maps uploaded'}</p>
      )}
    </div>
  );

  const renderCameraGrid = (entries, emptyText) => (
    <div className="cameras-grid">
      {entries.map(([name, cam]) => (
        <div key={name} className="glass-panel camera-card animate-in">
          <div className="camera-header">
            <div className="camera-title">
              {cam.source_type === 'video' ? <Video size={18} /> : <Camera size={18} />}
              {name}
            </div>
            <div style={{display: 'flex', gap: '0.5rem'}}>
              {cam.source_type === 'video' && (
                <>
                  <label className="video-card-select" title={`Select ${name}`}>
                    <input
                      type="checkbox"
                      checked={selectedVideos.includes(name)}
                      onChange={() => toggleVideoSelection(name)}
                    />
                  </label>
                  <span className={`badge ${cam.is_playing ? 'active' : 'paused'}`}>
                    {cam.is_playing ? 'Playing' : 'Paused'}
                  </span>
                </>
              )}
              {cam.has_processor && <span className="badge active">Calibrated</span>}
              <button
                onClick={() => setCalibratingCamera(name)}
                className="btn-icon"
                style={{color: 'var(--accent)', border: 'none', cursor: 'pointer'}}
                title="Calibrate"
              >
                <Crosshair size={18} />
              </button>
              <button
                onClick={() => handleDeleteCamera(name)}
                className="btn-icon"
                style={{color: 'var(--danger)', border: 'none', cursor: 'pointer'}}
                title="Delete"
              >
                <Trash2 size={18} />
              </button>
            </div>
          </div>
          <div className="camera-stream">
            <img src={`${API_URL}/video_feed/${name}`} alt={name} />
          </div>
        </div>
      ))}
      {entries.length === 0 && !loading && (
        <div className="empty-state">{emptyText}</div>
      )}
    </div>
  );

  const mapUploadPanel = (
    <PersistentDetails storageKey="global-map-upload" className="glass-panel collapsible-panel">
      <summary className="section-title collapsible-summary"><Map size={20} /> {isTH ? 'เพิ่ม Map' : 'Add Map'}</summary>
      <form onSubmit={handleUploadMap}>
        <div className="form-group">
          <label>{isTH ? 'สถานที่' : 'Location'}</label>
          <input type="text" name="location" className="form-control" required maxLength="120" placeholder={isTH ? 'เช่น อาคาร A' : 'e.g., Building A'} />
        </div>
        <div className="form-group">
          <label>{isTH ? 'ห้อง' : 'Room'}</label>
          <input type="text" name="room" className="form-control" required maxLength="120" placeholder={isTH ? 'เช่น ห้อง 101' : 'e.g., Room 101'} />
        </div>
        <div className="form-group">
          <input type="file" name="file" accept="image/*" className="form-control" required />
        </div>
        <button type="submit" className="btn">
          <Upload size={18} /> {isTH ? 'อัปโหลด Map' : 'Upload Map'}
        </button>
      </form>
    </PersistentDetails>
  );

  const realtimeCameraPanel = (
    <PersistentDetails storageKey="add-camera-stream" className="glass-panel collapsible-panel">
      <summary className="section-title collapsible-summary"><Camera size={20} /> {isTH ? 'เพิ่มกล้อง' : 'Add Camera'}</summary>
      <form onSubmit={handleAddCamera}>
        <div className="form-group">
          <label>{isTH ? 'ชื่อกล้อง' : 'Camera Name'}</label>
          <input type="text" name="name" className="form-control" required placeholder={isTH ? 'เช่น Cam1' : 'e.g., Cam1'} />
        </div>
        <div className="form-group">
          <label>RTSP / HTTP URL</label>
          <input type="text" name="url" className="form-control" required placeholder="rtsp://..." />
        </div>
        <button type="submit" className="btn">{isTH ? 'เพิ่มกล้อง' : 'Add Camera'}</button>
      </form>
    </PersistentDetails>
  );

  const playbackPanel = videoNames.length > 0 && (
    <div className="glass-panel playback-panel">
      <div className="playback-selection">
        <label className="video-select-label">
          <input type="checkbox" checked={allVideosSelected} onChange={toggleAllVideos} />
          {isTH ? 'เลือกวิดีโอทั้งหมด' : 'Select all videos'}
        </label>
        <span className="selection-count">
          {isTH ? `เลือก ${selectedVideos.length} จาก ${videoNames.length}` : `${selectedVideos.length} of ${videoNames.length} selected`}
        </span>
      </div>
      <div className="playback-actions">
        <button type="button" className="btn playback-button"
          onClick={() => handlePlayback('play', selectedVideos)}
          disabled={playbackLoading || selectedVideos.length === 0}>
          <Play size={17} /> {isTH ? 'เล่นที่เลือก' : 'Play Selected'}
        </button>
        <button type="button" className="btn playback-button secondary"
          onClick={() => handlePlayback('pause', selectedVideos)}
          disabled={playbackLoading || selectedVideos.length === 0}>
          <Pause size={17} /> {isTH ? 'หยุดที่เลือก' : 'Pause Selected'}
        </button>
        <button type="button" className="btn playback-button"
          onClick={() => handlePlayback('play')} disabled={playbackLoading}>
          <Play size={17} /> {isTH ? 'เล่นทั้งหมด' : 'Play All'}
        </button>
        <button type="button" className="btn playback-button secondary"
          onClick={() => handlePlayback('pause')} disabled={playbackLoading}>
          <Pause size={17} /> {isTH ? 'หยุดทั้งหมด' : 'Pause All'}
        </button>
      </div>
    </div>
  );

  return (
    <div className="app-shell animate-in">
      <aside className="main-nav">
        <div className="nav-brand" style={{display:'flex', alignItems:'flex-start', justifyContent:'space-between', gap:8}}>
          <div>
            <h1>People Location</h1>
            <h1>Tracker</h1>
          </div>
          <div className="language-switch" style={{display:'flex', gap:4, flexShrink:0}}>
            <button type="button" onClick={() => setUILanguage('en')} title="English" style={{padding:'5px 7px', minWidth:34, borderRadius:7, cursor:'pointer', opacity: language === 'en' ? 1 : .6}}>EN</button>
            <button type="button" onClick={() => setUILanguage('th')} title="ภาษาไทย" style={{padding:'5px 7px', minWidth:34, borderRadius:7, cursor:'pointer', opacity: language === 'th' ? 1 : .6}}>ไทย</button>
          </div>
        </div>

        <nav className="nav-menu" aria-label="Main navigation">
          <button
            type="button"
            className={`nav-item ${activePage === 'realtime' ? 'active' : ''}`}
            onClick={() => setActivePage('realtime')}
          >
            <Camera size={20} />
            <span>{isTH ? 'เรียลไทม์' : 'Realtime'}</span>
          </button>
          <button
            type="button"
            className={`nav-item ${activePage === 'video' ? 'active' : ''}`}
            onClick={() => setActivePage('video')}
          >
            <Video size={20} />
            <span>{isTH ? 'วิดีโอ' : 'Video'}</span>
          </button>
          <button
            type="button"
            className={`nav-item ${activePage === 'database' ? 'active' : ''}`}
            onClick={() => setActivePage('database')}
          >
            <span className="nav-db-icon">DB</span>
            <span>{isTH ? 'ฐานข้อมูล' : 'Database'}</span>
          </button>
        </nav>

        <div className="nav-status">
          <span className="status-dot" />
          <span>{isTH ? 'เชื่อมต่อ API แล้ว' : 'API Connected'}</span>
        </div>
      </aside>


      <style>{`
        .collapsible-panel > .collapsible-summary,
        .floorplan-collapse > .collapsible-summary {
          position: relative;
          cursor: pointer;
          list-style: none;
          padding-right: 44px;
          min-height: 34px;
        }
        .collapsible-panel > .collapsible-summary::-webkit-details-marker,
        .floorplan-collapse > .collapsible-summary::-webkit-details-marker {
          display: none;
        }
        .collapsible-panel > .collapsible-summary::after,
        .floorplan-collapse > .collapsible-summary::after {
          content: '+';
          position: absolute;
          right: 2px;
          top: 50%;
          transform: translateY(-50%);
          width: 34px;
          height: 34px;
          display: grid;
          place-items: center;
          border-radius: 9px;
          border: 1px solid var(--border);
          background: rgba(15, 23, 42, 0.7);
          color: var(--text);
          font-size: 27px;
          font-weight: 700;
          line-height: 1;
        }
        .collapsible-panel[open] > .collapsible-summary::after,
        .floorplan-collapse[open] > .collapsible-summary::after {
          content: '−';
        }

        .floorplan-collapse > .floorplan-summary {
          display: inline-flex;
          width: auto;
          align-items: center;
          gap: 10px;
          padding-right: 0;
        }
        .floorplan-collapse > .floorplan-summary::after {
          position: static;
          transform: none;
          width: 30px;
          height: 30px;
          flex: 0 0 30px;
        }
        .floorplan-groups {
          display: grid;
          gap: 8px;
        }
        .floorplan-location {
          border: 1px solid var(--border);
          border-radius: 10px;
          background: rgba(15, 23, 42, 0.28);
          overflow: hidden;
        }
        .floorplan-location > .floorplan-location-summary {
          display: inline-flex;
          width: auto;
          align-items: center;
          gap: 10px;
          min-height: 42px;
          padding: 8px 12px;
          cursor: pointer;
          list-style: none;
        }
        .floorplan-location > .floorplan-location-summary::-webkit-details-marker {
          display: none;
        }
        .floorplan-location > .floorplan-location-summary::after {
          content: '+';
          position: static;
          transform: none;
          width: 30px;
          height: 30px;
          flex: 0 0 30px;
          display: grid;
          place-items: center;
          border-radius: 8px;
          border: 1px solid var(--border);
          background: rgba(15, 23, 42, 0.7);
          color: var(--text);
          font-size: 23px;
          font-weight: 700;
          line-height: 1;
        }
        .floorplan-location[open] > .floorplan-location-summary::after {
          content: '−';
        }
        .floorplan-rooms {
          display: grid;
          gap: 4px;
          padding: 0 8px 8px 20px;
        }
        .floorplan-room {
          min-height: 36px;
        }
      `}</style>

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
                <h2>{isTH ? 'ติดตามแบบเรียลไทม์' : 'Realtime Tracking'}</h2>
                <p>{isTH ? 'แสดงตำแหน่งจากกล้องแบบเรียลไทม์' : 'Track people from live cameras'}</p>
              </div>
            </div>
            <div className="workspace-grid">
              <aside className="control-column">
                {mapUploadPanel}
                {realtimeCameraPanel}
                <EmbeddingDatabase compact compactMode="save" language={language} />
                <EmbeddingDatabase compact compactMode="compare" language={language} />
              </aside>
              <div className="workspace-main">
                {renderMapPanel(isTH ? 'แผนที่ติดตามแบบเรียลไทม์' : 'Realtime Tracking Map')}
                {renderCameraGrid(realtimeCameras, 'No realtime cameras added yet.')}
              </div>
            </div>
          </section>
        )}

        {activePage === 'video' && (
          <section className="page-view">
            <div className="page-heading">
              <div>
                <h2>{isTH ? 'ติดตามจากวิดีโอ' : 'Video Tracking'}</h2>
                <p>{isTH ? 'ติดตามตำแหน่งจากไฟล์วิดีโอที่อัปโหลด' : 'Track people from uploaded video files'}</p>
              </div>
            </div>
            <div className="workspace-grid">
              <aside className="control-column">
                {mapUploadPanel}
                <VideoUploader
                  API_URL={API_URL}
                  onSuccess={(msg) => { showAlert(msg, "success"); fetchStatus(); }}
                  language={language}
                />
                <EmbeddingDatabase compact compactMode="save" language={language} />
                <EmbeddingDatabase compact compactMode="compare" language={language} />
              </aside>
              <div className="workspace-main">
                {renderMapPanel(isTH ? 'แผนที่ติดตามจากวิดีโอ' : 'Video Tracking Map')}
                {playbackPanel}
                {renderCameraGrid(videoCameras, 'No video files added yet.')}
              </div>
            </div>
          </section>
        )}

        {activePage === 'database' && (
          <section className="page-view database-page">
            <div className="page-heading">
              <div>
                <h2>{isTH ? 'ฐานข้อมูล' : 'Database'}</h2>
                <p>{isTH ? 'ค้นหา เปรียบเทียบ และจัดการข้อมูล Embedding' : 'Search, compare, and manage saved embeddings'}</p>
              </div>
            </div>
            <EmbeddingDatabase language={language} />
          </section>
        )}
      </main>

      {calibratingCamera && (
        <CalibrationModal 
          camName={calibratingCamera} 
          API_URL={API_URL} 
          onClose={() => setCalibratingCamera(null)} 
          onSuccess={(msg) => { showAlert(msg, "success"); fetchStatus(); }} 
        />
      )}
    </div>
  );
}
