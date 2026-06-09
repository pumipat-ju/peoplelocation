import { useState, useRef, useEffect } from 'react';
import { X, Save, RotateCcw } from 'lucide-react';
import './CalibrationModal.css';

export default function CalibrationModal({ camName, API_URL, onClose, onSuccess }) {
  const [ptsSrc, setPtsSrc] = useState([]);
  const [ptsDst, setPtsDst] = useState([]);
  const [frameUrl, setFrameUrl] = useState(null);
  const [mapUrl, setMapUrl] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchImages = async () => {
      try {
        const frameRes = await fetch(`${API_URL}/capture_frame/${camName}?t=${Date.now()}`);
        const frameData = await frameRes.json();
        if (frameData.image_base64 || frameData.data?.image_base64) {
          const b64 = frameData.image_base64 || frameData.data.image_base64;
          setFrameUrl(`data:image/jpeg;base64,${b64}`);
        } else {
          console.error("Failed to capture frame");
        }

        const mapRes = await fetch(`${API_URL}/get_floorplan?t=${Date.now()}`);
        const mapData = await mapRes.json();
        if (mapData.image_base64 || mapData.data?.image_base64) {
          const b64 = mapData.image_base64 || mapData.data.image_base64;
          setMapUrl(`data:image/jpeg;base64,${b64}`);
        } else {
          console.error("Failed to get floorplan");
        }
      } catch (err) {
        console.error("Error fetching calibration images:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchImages();
  }, [camName, API_URL]);

  const handleImageClick = (e, isSrc) => {
    const rect = e.target.getBoundingClientRect();
    // Get coordinates relative to the original image dimensions
    // To do this accurately, we need the natural dimensions of the image vs rendered dimensions.
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Calculate scaling factor
    const scaleX = e.target.naturalWidth / rect.width;
    const scaleY = e.target.naturalHeight / rect.height;

    const actualX = Math.round(x * scaleX);
    const actualY = Math.round(y * scaleY);

    if (isSrc) {
      if (ptsSrc.length < 4) setPtsSrc([...ptsSrc, [actualX, actualY]]);
    } else {
      if (ptsDst.length < 4) setPtsDst([...ptsDst, [actualX, actualY]]);
    }
  };

  const handleSave = async () => {
    if (ptsSrc.length !== 4 || ptsDst.length !== 4) return;

    try {
      const formData = new FormData();
      formData.append("src_pts", JSON.stringify(ptsSrc));
      formData.append("dst_pts", JSON.stringify(ptsDst));

      const res = await fetch(`${API_URL}/save_calibration/${camName}`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.success || res.ok) {
        onSuccess(data.message || "Calibration saved!");
        onClose();
      } else {
        alert("Calibration failed: " + (data.message || JSON.stringify(data.detail) || "Unknown error"));
      }
    } catch (err) {
      alert("Error saving calibration.");
    }
  };

  const resetPts = () => {
    setPtsSrc([]);
    setPtsDst([]);
  };

  return (
    <div className="modal-overlay">
      <div className="modal-content animate-in">
        <div className="modal-header">
          <h2>Calibrate Camera: {camName}</h2>
          <button className="btn-icon" onClick={onClose}><X size={20} /></button>
        </div>

        <div className="modal-body">
          <p style={{marginBottom: '1rem', color: 'var(--text-muted)'}}>
            Click 4 points on the camera view (left) and the corresponding 4 points on the floorplan (right) in the exact same order.
          </p>

          <div className="calibration-grid">
            <div className="calib-col">
              <h3>Camera View ({ptsSrc.length}/4)</h3>
              <div className="calib-image-container">
                {frameUrl && (
                  <img 
                    src={frameUrl} 
                    alt="Camera Frame" 
                    onClick={(e) => handleImageClick(e, true)}
                    onLoad={(e) => setLoading(false)}
                  />
                )}
                {/* Visual dots */}
                {ptsSrc.map((pt, i) => (
                  <div key={i} className="calib-dot" style={{
                    left: `calc(${(pt[0] / (document.querySelector('.calib-col img')?.naturalWidth || 1)) * 100}% - 8px)`,
                    top: `calc(${(pt[1] / (document.querySelector('.calib-col img')?.naturalHeight || 1)) * 100}% - 8px)`
                  }}>
                    {i + 1}
                  </div>
                ))}
              </div>
            </div>

            <div className="calib-col">
              <h3>Floorplan ({ptsDst.length}/4)</h3>
              <div className="calib-image-container">
                {mapUrl && (
                  <img 
                    src={mapUrl} 
                    alt="Floorplan" 
                    onClick={(e) => handleImageClick(e, false)} 
                  />
                )}
                 {ptsDst.map((pt, i) => (
                  <div key={i} className="calib-dot" style={{
                    left: `calc(${(pt[0] / (document.querySelectorAll('.calib-col img')[1]?.naturalWidth || 1)) * 100}% - 8px)`,
                    top: `calc(${(pt[1] / (document.querySelectorAll('.calib-col img')[1]?.naturalHeight || 1)) * 100}% - 8px)`
                  }}>
                    {i + 1}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={resetPts} style={{width: 'auto'}}>
            <RotateCcw size={18} /> Reset Points
          </button>
          <button 
            className="btn" 
            onClick={handleSave} 
            disabled={ptsSrc.length !== 4 || ptsDst.length !== 4}
            style={{width: 'auto', opacity: (ptsSrc.length === 4 && ptsDst.length === 4) ? 1 : 0.5}}
          >
            <Save size={18} /> Save Calibration
          </button>
        </div>
      </div>
    </div>
  );
}
