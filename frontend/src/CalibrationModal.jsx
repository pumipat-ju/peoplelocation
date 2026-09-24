import { useState, useEffect } from 'react';
import { X, Save, RotateCcw } from 'lucide-react';
import './CalibrationModal.css';

function CalibrationImage({ src, alt, points, onAddPoint, referencePolygons = [] }) {
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 });
  const [cursorPoint, setCursorPoint] = useState(null);

  const pointFromEvent = (e) => {
    if (!naturalSize.width || !naturalSize.height) return null;

    const rect = e.currentTarget.getBoundingClientRect();
    return [
      Math.round((e.clientX - rect.left) * naturalSize.width / rect.width),
      Math.round((e.clientY - rect.top) * naturalSize.height / rect.height)
    ];
  };

  const handleLoad = (e) => {
    setNaturalSize({
      width: e.currentTarget.naturalWidth,
      height: e.currentTarget.naturalHeight
    });
  };

  const handleClick = (e) => {
    if (points.length >= 4) return;
    const point = pointFromEvent(e);
    if (point) onAddPoint(point);
  };

  const linePoints = points.map(point => point.join(',')).join(' ');
  const lastPoint = points.at(-1);

  return (
    <div
      className="calib-image-container"
      onClick={handleClick}
      onMouseMove={(e) => setCursorPoint(pointFromEvent(e))}
      onMouseLeave={() => setCursorPoint(null)}
    >
      <img src={src} alt={alt} onLoad={handleLoad} draggable="false" />

      {naturalSize.width > 0 && (
        <svg
          className="calib-line-overlay"
          viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          {referencePolygons.map((region, regionIndex) => {
            const polygonPoints = region.points.map(point => point.join(',')).join(' ');
            const centerX = region.points.reduce((sum, point) => sum + point[0], 0) / region.points.length;
            const centerY = region.points.reduce((sum, point) => sum + point[1], 0) / region.points.length;
            return <g key={`${region.camera_name}-${regionIndex}`}>
              <polygon points={polygonPoints} fill="rgba(255,165,0,0.18)" stroke="#ff9f1c" strokeWidth="4" strokeDasharray="10 6" />
              <text x={centerX} y={centerY} textAnchor="middle" dominantBaseline="middle"
                fill="#fff" stroke="#111827" strokeWidth="4" paintOrder="stroke" fontSize="20" fontWeight="700">
                {region.camera_name}
              </text>
            </g>;
          })}
          {points.length >= 2 && points.length < 4 && (
            <polyline className="calib-shape" points={linePoints} />
          )}
          {points.length === 4 && (
            <polygon className="calib-shape calib-shape-complete" points={linePoints} />
          )}
          {lastPoint && cursorPoint && points.length < 4 && (
            <line
              className="calib-preview-line"
              x1={lastPoint[0]}
              y1={lastPoint[1]}
              x2={cursorPoint[0]}
              y2={cursorPoint[1]}
            />
          )}
          {cursorPoint && points.length < 4 && (
            <>
              <line className="calib-guide-line" x1="0" y1={cursorPoint[1]} x2={naturalSize.width} y2={cursorPoint[1]} />
              <line className="calib-guide-line" x1={cursorPoint[0]} y1="0" x2={cursorPoint[0]} y2={naturalSize.height} />
            </>
          )}
        </svg>
      )}

      {naturalSize.width > 0 && points.map((point, index) => (
        <div
          key={index}
          className="calib-dot"
          style={{
            left: `${point[0] / naturalSize.width * 100}%`,
            top: `${point[1] / naturalSize.height * 100}%`
          }}
        >
          {index + 1}
        </div>
      ))}
    </div>
  );
}

export default function CalibrationModal({ camName, API_URL, onClose, onSuccess }) {
  const [ptsSrc, setPtsSrc] = useState([]);
  const [ptsDst, setPtsDst] = useState([]);
  const [frameUrl, setFrameUrl] = useState(null);
  const [mapUrl, setMapUrl] = useState(null);
  const [floorplans, setFloorplans] = useState([]);
  const [selectedFloorplan, setSelectedFloorplan] = useState('');
  const [loadingFloorplans, setLoadingFloorplans] = useState(true);
  const [existingCalibrations, setExistingCalibrations] = useState([]);

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

        const floorplanRes = await fetch(`${API_URL}/floorplans?t=${Date.now()}`);
        const floorplanData = await floorplanRes.json();
        setFloorplans(floorplanData.floorplans || []);
      } catch (err) {
        console.error("Error fetching calibration images:", err);
      } finally {
        setLoadingFloorplans(false);
      }
    };

    fetchImages();
  }, [camName, API_URL]);

  useEffect(() => {
    if (!selectedFloorplan) {
      setMapUrl(null);
      setPtsDst([]);
      setExistingCalibrations([]);
      return;
    }
    let cancelled = false;
    const loadSelectedFloorplan = async () => {
      try {
        const [res, regionsRes] = await Promise.all([
          fetch(`${API_URL}/get_floorplan?name=${encodeURIComponent(selectedFloorplan)}&t=${Date.now()}`),
          fetch(`${API_URL}/floorplans/${encodeURIComponent(selectedFloorplan)}/calibrations?exclude_camera=${encodeURIComponent(camName)}&t=${Date.now()}`)
        ]);
        const data = await res.json();
        const regionsData = await regionsRes.json();
        if (!res.ok) throw new Error(data.error || 'Failed to load floorplan');
        if (!regionsRes.ok) throw new Error(regionsData.detail || 'Failed to load calibration regions');
        const b64 = data.image_base64 || data.data?.image_base64;
        if (!cancelled && b64) {
          setMapUrl(`data:image/jpeg;base64,${b64}`);
          setPtsDst([]);
          setExistingCalibrations(regionsData.calibrations || []);
        }
      } catch (err) {
        if (!cancelled) {
          setMapUrl(null);
          setExistingCalibrations([]);
          alert(`โหลด Floorplan ไม่สำเร็จ: ${err.message}`);
        }
      }
    };
    loadSelectedFloorplan();
    return () => { cancelled = true; };
  }, [selectedFloorplan, API_URL, camName]);

  const handleSave = async () => {
    if (!selectedFloorplan || ptsSrc.length !== 4 || ptsDst.length !== 4) return;

    try {
      const formData = new FormData();
      formData.append("src_pts", JSON.stringify(ptsSrc));
      formData.append("dst_pts", JSON.stringify(ptsDst));
      formData.append("floorplan_name", selectedFloorplan);

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
          <div className="form-group" style={{marginBottom: '1rem'}}>
            <label htmlFor="calibration-floorplan">เลือก Floorplan ที่ต้องการ Calibration</label>
            <select
              id="calibration-floorplan"
              className="form-control"
              value={selectedFloorplan}
              onChange={(e) => setSelectedFloorplan(e.target.value)}
              disabled={loadingFloorplans}
            >
              <option value="">{loadingFloorplans ? 'กำลังโหลด...' : 'กรุณาเลือก Floorplan'}</option>
              {floorplans.map(item => <option key={item.name} value={item.name}>{item.name}</option>)}
            </select>
            {!loadingFloorplans && floorplans.length === 0 && (
              <p style={{color: 'var(--danger)', marginTop: '0.5rem'}}>ยังไม่มี Floorplan กรุณาอัปโหลดก่อน Calibration</p>
            )}
          </div>
          <p style={{marginBottom: '1rem', color: 'var(--text-muted)'}}>
            Click 4 points on the camera view (left) and the corresponding 4 points on the floorplan (right) in the exact same order.
          </p>

          <div className="calibration-grid">
            <div className="calib-col">
              <h3>Camera View ({ptsSrc.length}/4)</h3>
              {frameUrl && (
                <CalibrationImage
                  src={frameUrl}
                  alt="Camera Frame"
                  points={ptsSrc}
                  onAddPoint={(point) => setPtsSrc(prev => [...prev, point])}
                />
              )}
            </div>

            <div className="calib-col">
              <h3>Floorplan: {selectedFloorplan || 'ยังไม่ได้เลือก'} ({ptsDst.length}/4)</h3>
              {mapUrl && (
                <CalibrationImage
                  src={mapUrl}
                  alt="Floorplan"
                  points={ptsDst}
                  onAddPoint={(point) => setPtsDst(prev => [...prev, point])}
                  referencePolygons={existingCalibrations}
                />
              )}
              {existingCalibrations.length > 0 && (
                <p style={{color: 'var(--text-muted)', marginTop: '0.5rem'}}>
                  ขอบเส้นประสีส้มคือพื้นที่ Calibration ของกล้องอื่นบน Floorplan นี้
                </p>
              )}
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
            disabled={!selectedFloorplan || ptsSrc.length !== 4 || ptsDst.length !== 4}
            style={{width: 'auto', opacity: (selectedFloorplan && ptsSrc.length === 4 && ptsDst.length === 4) ? 1 : 0.5}}
          >
            <Save size={18} /> Save Calibration
          </button>
        </div>
      </div>
    </div>
  );
}
