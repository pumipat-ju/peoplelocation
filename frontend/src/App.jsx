import { useState, useEffect } from 'react';
import { Camera, Map, Upload, Video, Trash2, AlertCircle, CheckCircle2, Crosshair } from 'lucide-react';
import './index.css';
import CalibrationModal from './CalibrationModal';

const API_URL = 'http://localhost:8899/api';
const HOST_URL = 'http://localhost:8899';

export default function App() {
  const [status, setStatus] = useState({ cameras: {}, floorplan_exists: false });
  const [alert, setAlert] = useState(null);
  const [loading, setLoading] = useState(true);
  const [calibratingCamera, setCalibratingCamera] = useState(null);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/status`);
      const data = await res.json();
      setStatus(data);
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

  const showAlert = (message, type = "error") => {
    setAlert({ message, type });
    setTimeout(() => setAlert(null), 5000);
  };

  const handleUploadMap = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    try {
      const res = await fetch(`${API_URL}/upload_floorplan`, { method: 'POST', body: formData });
      const data = await res.json();
      showAlert(data.message, data.success ? "success" : "error");
      if (data.success) fetchStatus();
    } catch (err) {
      showAlert("Upload failed", "error");
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

  const handleUploadVideo = async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    formData.append('loop_video', 'true');
    try {
      const res = await fetch(`${API_URL}/upload_video`, { method: 'POST', body: formData });
      const data = await res.json();
      showAlert(data.message, data.success ? "success" : "error");
      if (data.success) {
        fetchStatus();
        e.target.reset();
      }
    } catch (err) {
      showAlert("Failed to upload video", "error");
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
          <div className="glass-panel">
            <h2 className="section-title"><Video size={20} /> Upload Video Source</h2>
            <form onSubmit={handleUploadVideo}>
              <div className="form-group">
                <label>Camera Name</label>
                <input type="text" name="name" className="form-control" required placeholder="e.g., Cam2" />
              </div>
              <div className="form-group">
                <label>Video File (MP4/AVI)</label>
                <input type="file" name="file" accept="video/*" className="form-control" required />
              </div>
              <button type="submit" className="btn">Upload Video</button>
            </form>
          </div>
        </aside>

        <main className="cameras-section">
          {/* Global Map Display */}
          <div className="glass-panel">
            <h2 className="section-title">Live Tracking Map</h2>
            <div className="map-container">
              {status.floorplan_exists ? (
                <img src={`${API_URL}/global_map_feed?t=${Date.now()}`} alt="Global Map" />
              ) : (
                <p style={{color: 'var(--text-muted)'}}>No Floorplan Uploaded</p>
              )}
            </div>
          </div>

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
