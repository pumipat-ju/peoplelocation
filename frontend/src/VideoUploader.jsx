import { useState, useRef } from 'react';
import { Video, UploadCloud, Trash2, CheckCircle2, AlertCircle } from 'lucide-react';

export default function VideoUploader({ API_URL, onSuccess, language = 'en' }) {
  const isTH = language === 'th';
  const [files, setFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [cameraPrefix, setCameraPrefix] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);
  const [panelOpen, setPanelOpen] = useState(() => sessionStorage.getItem('panel:upload-video') === '1');

  const handlePanelToggle = (event) => {
    const nextOpen = event.currentTarget.open;
    setPanelOpen(nextOpen);
    sessionStorage.setItem('panel:upload-video', nextOpen ? '1' : '0');
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const addFiles = (newFiles) => {
    const validFiles = Array.from(newFiles).filter(file => file.type.startsWith('video/') || file.name.match(/\.(mp4|avi|mov|mkv|webm)$/i));
    const mappedFiles = validFiles.map((file) => ({
      id: Math.random().toString(36).substring(2, 9),
      file,
      progress: 0,
      offset: 0,
      status: 'pending', // pending, uploading, success, error
      errorMessage: ''
    }));
    setFiles(prev => [...prev, ...mappedFiles]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(e.target.files);
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removeFile = (id) => {
    setFiles(prev => prev.filter(f => f.id !== id));
  };

  const getResponseError = (xhr) => {
    try {
      const data = JSON.parse(xhr.responseText);
      if (data.message) return data.message;
      if (Array.isArray(data.detail)) {
        return data.detail.map(item => item.msg).filter(Boolean).join(', ');
      }
      if (typeof data.detail === 'string') return data.detail;
    } catch {
      // The backend may return a plain-text error (for example from a proxy).
    }
    return `Upload failed (HTTP ${xhr.status})`;
  };

  const uploadFile = (fileObj, cameraName) => {
    return new Promise((resolve) => {
      const formData = new FormData();
      formData.append('name', cameraName);
      formData.append('file', fileObj.file);
      formData.append('loop_video', 'false');
      formData.append('time_offset_sec', String(fileObj.offset || 0));

      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${API_URL}/upload_video`, true);

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) {
          const percentComplete = Math.round((event.loaded / event.total) * 100);
          setFiles(prev => prev.map(f => f.id === fileObj.id ? { ...f, progress: percentComplete, status: 'uploading' } : f));
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            if (data.success) {
              setFiles(prev => prev.map(f => f.id === fileObj.id ? { ...f, progress: 100, status: 'success', errorMessage: '' } : f));
              resolve({ success: true });
            } else {
              setFiles(prev => prev.map(f => f.id === fileObj.id ? { ...f, status: 'error', errorMessage: data.message || 'Upload failed' } : f));
              resolve({ success: false });
            }
          } catch(e) {
            setFiles(prev => prev.map(f => f.id === fileObj.id ? { ...f, status: 'error', errorMessage: 'Invalid response from server' } : f));
            resolve({ success: false });
          }
        } else {
          const errorMessage = getResponseError(xhr);
          setFiles(prev => prev.map(f => f.id === fileObj.id ? { ...f, status: 'error', errorMessage } : f));
          resolve({ success: false });
        }
      };

      xhr.onerror = () => {
        setFiles(prev => prev.map(f => f.id === fileObj.id ? { ...f, status: 'error', errorMessage: 'Network error' } : f));
        resolve({ success: false });
      };

      xhr.send(formData);
    });
  };

  const handleUploadAll = async () => {
    if (files.length === 0) return;
    if (!cameraPrefix) {
      alert("Please enter a Camera Name (Prefix)");
      return;
    }
    setIsUploading(true);

    const pendingFiles = files.filter(f => f.status === 'pending' || f.status === 'error');

    const results = await Promise.all(pendingFiles.map((fileObj) => {
      const fileIndex = files.findIndex(item => item.id === fileObj.id);
      const cameraName = files.length === 1
        ? cameraPrefix
        : `${cameraPrefix}_${fileIndex + 1}`;
      return uploadFile(fileObj, cameraName);
    }));
    const anySuccess = results.some(result => result.success);
    const successfulIds = new Set(
      pendingFiles
        .filter((_, index) => results[index].success)
        .map(fileObj => fileObj.id)
    );

    setIsUploading(false);
    setFiles(prev => prev.filter(fileObj => !successfulIds.has(fileObj.id)));

    if (anySuccess && onSuccess) {
      onSuccess("Videos uploaded successfully!");
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <details className="glass-panel collapsible-panel" open={panelOpen} onToggle={handlePanelToggle}>

      <style>{`
        .collapsible-panel > .collapsible-summary {
          cursor: pointer;
          user-select: none;
          list-style: none;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
          margin: 0;
          font-weight: 700;
        }
        .collapsible-panel > .collapsible-summary::-webkit-details-marker { display: none; }
      `}</style>

      <summary className="section-title collapsible-summary">
        <span style={{display:'inline-flex', alignItems:'center', gap:'0.5rem'}}>
          <Video size={20} /> {isTH ? 'เพิ่มวิดีโอ' : 'Add Video'}
        </span>
        <span aria-hidden="true" style={{display:'inline-flex', alignItems:'center', justifyContent:'center', width:'34px', height:'34px', flex:'0 0 34px', borderRadius:'9px', background:'var(--primary-soft, #eaf3ff)', color:'var(--primary, #3478dc)', fontSize:'1.65rem', fontWeight:700, lineHeight:1}}>
          {panelOpen ? '−' : '+'}
        </span>
      </summary>

      <div className="form-group">
        <label>{isTH ? 'ชื่อกล้อง' : 'Camera Name'}</label>
        <input
          type="text"
          value={cameraPrefix}
          onChange={(e) => setCameraPrefix(e.target.value)}
          className="form-control"
          placeholder="e.g., Cam2"
          disabled={isUploading}
        />
      </div>

      <div
        className={`drop-zone ${isDragging ? 'active' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current && fileInputRef.current.click()}
      >
        <UploadCloud size={32} style={{marginBottom: '0.5rem'}} />
        <p>{isTH ? 'ลากไฟล์วิดีโอมาวาง หรือคลิกเพื่อเลือกไฟล์' : 'Drag & drop videos here, or click to choose files'}</p>
        <input
          type="file"
          ref={fileInputRef}
          accept="video/*"
          multiple
          onChange={handleFileSelect}
        />
      </div>

      {files.length > 0 && (
        <div className="file-list">
          {files.map((fileObj) => (
            <div key={fileObj.id} className="file-list-item">
              <div className="file-list-header">
                <span className="file-name" title={fileObj.file.name}>
                  {fileObj.file.name} <span style={{color: 'var(--text-muted)', fontSize: '0.8rem'}}>({formatBytes(fileObj.file.size)})</span>
                </span>
                <label style={{display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.75rem'}}>
                  {isTH ? 'เวลาเริ่ม (วินาที)' : 'Start offset (s)'}
                  <input
                    type="number"
                    step="0.01"
                    value={fileObj.offset}
                    disabled={isUploading || fileObj.status === 'success'}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => setFiles(prev => prev.map(f => f.id === fileObj.id ? {...f, offset: e.target.value} : f))}
                    style={{width: '5rem'}}
                  />
                </label>

                <div className="file-actions">
                  {fileObj.status === 'success' && <CheckCircle2 size={16} color="var(--success)" />}
                  {fileObj.status === 'error' && <AlertCircle size={16} color="var(--danger)" title={fileObj.errorMessage} />}

                  {fileObj.status !== 'success' && fileObj.status !== 'uploading' && (
                    <button
                      type="button"
                      className="btn-icon"
                      style={{color: 'var(--danger)', border: 'none', cursor: 'pointer', padding: '0.2rem'}}
                      onClick={(e) => { e.stopPropagation(); removeFile(fileObj.id); }}
                    >
                      <Trash2 size={16} />
                    </button>
                  )}
                </div>
              </div>

              {(fileObj.status === 'uploading' || fileObj.progress > 0) && (
                <div className="progress-container">
                  <div
                    className={`progress-fill ${fileObj.status}`}
                    style={{width: `${fileObj.progress}%`}}
                  ></div>
                </div>
              )}
              {fileObj.status === 'error' && <div style={{color: 'var(--danger)', fontSize: '0.75rem', marginTop: '0.2rem'}}>{fileObj.errorMessage}</div>}
            </div>
          ))}
        </div>
      )}

      {files.length > 0 && (
        <button
          type="button"
          className="btn"
          onClick={handleUploadAll}
          disabled={isUploading || files.every(f => f.status === 'success')}
        >
          {isUploading ? (isTH ? 'กำลังอัปโหลด...' : 'Uploading...') : (isTH ? 'อัปโหลดวิดีโอทั้งหมด' : 'Upload All Videos')}
        </button>
      )}
    </details>
  );
}
