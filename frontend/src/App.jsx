import { useState, useEffect } from 'react';
import { Camera, Map, Upload, Video, Trash2, AlertCircle, CheckCircle2, Crosshair, Play, Pause } from 'lucide-react';
import './index.css';
import CalibrationModal from './CalibrationModal';
import VideoUploader from './VideoUploader';
import EmbeddingDatabase from './EmbeddingDatabase';

const API_URL = 'http://localhost:8899/api';
const HOST_URL = 'http://localhost:8899';

export default function App() {
  const [status, setStatus] = useState({ cameras: {}, floorplan_exists: false });
  const [alert, setAlert] = useState(null);
  const [loading, setLoading] = useState(true);
  const [calibratingCamera, setCalibratingCamera] = useState(null);
  const [selectedVideos, setSelectedVideos] = useState([]);
  const [playbackLoading, setPlaybackLoading] = useState(false);
  const [selectedFloorplans, setSelectedFloorplans] = useState([]);

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
    setSelectedFloorplans((current) => current.includes(name)
      ? current.filter(item => item !== name)
      : [...current, name]);
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

  return (
    <div className="app-container animate-in">
      <header>
        <h1>People Location Tracker</h1>
        <div className="flex gap-2">
          <span className="badge active">API Connected</span>
        </div>
      </header>

      {alert && (
        <div className={`alert ${alert.type === 'success' ? 'alert-success' : ''} animate-in`}>
          <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
            {alert.type === 'success' ? <CheckCircle2 size={20} /> : <AlertCircle size={20} />}
            <span>{alert.message}</span>
          </div>
        </div>
      )}

      <div className="dashboard-grid">
        <aside className="sidebar">
          {/* Map Upload */}
          <div className="glass-panel">
            <h2 className="section-title"><Map size={20} /> Global Map</h2>
            <form onSubmit={handleUploadMap}>
              <div className="form-group">
                <label>Floorplan Name</label>
                <input type="text" name="floorplan_name" className="form-control"
                  required maxLength="80" placeholder="เช่น ชั้น 1 หรือ Office Zone A" />
              </div>
              <div className="form-group">
                <input type="file" name="file" accept="image/*" className="form-control" required />
              </div>
              <button type="submit" className="btn">
                <Upload size={18} /> Upload Floorplan
              </button>
            </form>
          </div>

          {/* Add RTSP Camera */}
          <div className="glass-panel">
            <h2 className="section-title"><Camera size={20} /> Add Camera Stream</h2>
            <form onSubmit={handleAddCamera}>
              <div className="form-group">
                <label>Camera Name</label>
                <input type="text" name="name" className="form-control" required placeholder="e.g., Cam1" />
              </div>
              <div className="form-group">
                <label>RTSP / HTTP URL</label>
                <input type="text" name="url" className="form-control" required placeholder="rtsp://..." />
              </div>
              <button type="submit" className="btn">Add Stream</button>
            </form>
          </div>

          {/* Upload Video File */}
          <VideoUploader
            API_URL={API_URL}
            onSuccess={(msg) => { showAlert(msg, "success"); fetchStatus(); }}
          />
        </aside>

        <main className="cameras-section">
          {/* Global Map Display */}
          <div className="glass-panel">
            <h2 className="section-title">Live Tracking Map</h2>
            {(status.floorplans || []).length > 0 ? <>
              <div style={{display: 'flex', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '1rem'}}>
                {status.floorplans.map(name => (
                  <div key={name} style={{display: 'flex', alignItems: 'center', gap: '0.35rem'}}>
                    <label style={{display: 'flex', alignItems: 'center', gap: '0.4rem'}}>
                      <input type="checkbox" checked={selectedFloorplans.includes(name)} onChange={() => toggleFloorplan(name)} />
                      {name}
                    </label>
                    <button type="button" className="btn-icon" title={`ลบ ${name}`}
                      onClick={() => handleDeleteFloorplan(name)}
                      style={{color: 'var(--danger)', border: 'none', cursor: 'pointer'}}>
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
              </div>
              {selectedFloorplans.length === 0 && <p style={{color: 'var(--text-muted)'}}>เลือก Floorplan ที่ต้องการแสดง</p>}
              <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1rem'}}>
                {selectedFloorplans.map(name => (
                  <div key={name}>
                    <h3 style={{marginBottom: '0.5rem'}}>{name}</h3>
                    <div className="map-container">
                      <img src={`${API_URL}/global_map_feed?name=${encodeURIComponent(name)}&t=${Date.now()}`} alt={`Global Map ${name}`} />
                    </div>
                  </div>
                ))}
              </div>
            </> : (
              <p style={{color: 'var(--text-muted)'}}>No Floorplan Uploaded</p>
            )}
          </div>

          {/* Video Playback Controls */}
          {videoNames.length > 0 && (
            <div className="glass-panel playback-panel">
              <div className="playback-selection">
                <label className="video-select-label">
                  <input
                    type="checkbox"
                    checked={allVideosSelected}
                    onChange={toggleAllVideos}
                  />
                  Select all videos
                </label>
                <span className="selection-count">
                  {selectedVideos.length} of {videoNames.length} selected
                </span>
              </div>
              <div className="playback-actions">
                <button
                  type="button"
                  className="btn playback-button"
                  onClick={() => handlePlayback('play', selectedVideos)}
                  disabled={playbackLoading || selectedVideos.length === 0}
                >
                  <Play size={17} /> Play Selected
                </button>
                <button
                  type="button"
                  className="btn playback-button secondary"
                  onClick={() => handlePlayback('pause', selectedVideos)}
                  disabled={playbackLoading || selectedVideos.length === 0}
                >
                  <Pause size={17} /> Pause Selected
                </button>
                <button
                  type="button"
                  className="btn playback-button"
                  onClick={() => handlePlayback('play')}
                  disabled={playbackLoading}
                >
                  <Play size={17} /> Play All
                </button>
                <button
                  type="button"
                  className="btn playback-button secondary"
                  onClick={() => handlePlayback('pause')}
                  disabled={playbackLoading}
                >
                  <Pause size={17} /> Pause All
                </button>
              </div>
            </div> 
          )}

          {/* Cameras Grid */}
          <div className="cameras-grid">
            {Object.entries(status.cameras).map(([name, cam]) => (
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
                      className="btn-icon" style={{color: 'var(--accent)', border: 'none', cursor: 'pointer'}} title="Calibrate">
                      <Crosshair size={18} />
                    </button>
                    <button 
                      onClick={() => handleDeleteCamera(name)} 
                      className="btn-icon" style={{color: 'var(--danger)', border: 'none', cursor: 'pointer'}} title="Delete">
                      <Trash2 size={18} />
                    </button>
                  </div>
                </div>
                <div className="camera-stream">
                  <img src={`${API_URL}/video_feed/${name}`} alt={name} />
                </div>
              </div>
            ))}
            {Object.keys(status.cameras).length === 0 && !loading && (
              <div style={{gridColumn: '1 / -1', textAlign: 'center', padding: '3rem', color: 'var(--text-muted)'}}>
                No cameras or videos added yet.
              </div>
            )}
          </div>
          <EmbeddingDatabase />
        </main>
      </div>

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
