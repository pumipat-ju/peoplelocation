import cv2
import os
import threading
import time
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# โหลดโมเดล YOLO (แนะนำ yolov8n.pt สำหรับ CPU ทั่วไป)
model = YOLO('yolov8s.pt')

# เก็บสถานะการทำงานและรายชื่อกล้อง
app.is_running = True
camera_sources = {}

def generate_frames(camera_url):
    """ฟังก์ชันดึงภาพจากกล้องและส่งเข้า Model AI"""
    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) # ลด Delay
    
    while app.is_running:
        success, frame = cap.read()
        if not success:
            break
        
        # ประมวลผล YOLOv8 (Tracking คน)
        results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)
        
        # วาด Box และ ID ลงบนภาพ
        annotated_frame = results[0].plot()

        # แปลงภาพเป็น JPG เพื่อส่งผ่านเว็บ
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()
    print(f"Closed connection to: {camera_url}")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "cameras": camera_sources})

@app.post("/add_camera")
async def add_camera(name: str = Form(...), url: str = Form(...)):
    # ถ้าใส่เลข 0 หรือ 1 ให้แปลงเป็น int สำหรับ Webcam
    final_url = int(url) if url.isdigit() else url
    camera_sources[name] = final_url
    return HTMLResponse("<script>alert('เพิ่มกล้องสำเร็จ'); window.location.href='/';</script>")

@app.get("/video_feed/{cam_name}")
async def video_feed(cam_name: str):
    url = camera_sources.get(cam_name)
    if url is not None:
        return StreamingResponse(generate_frames(url), 
                                 media_type="multipart/x-mixed-replace; boundary=frame")
    return {"error": "Camera not found"}

@app.post("/shutdown")
async def shutdown_system():
    """API สำหรับปิดระบบแบบสะอาด"""
    app.is_running = False
    
    def kill_server():
        time.sleep(1) # รอให้ระบบเครือข่ายส่ง Response จบก่อน
        print("System shut down safely.")
        os._exit(0)

    threading.Thread(target=kill_server).start()
    return {"status": "shutting down"}

if __name__ == "__main__":
    import uvicorn
    # host 0.0.0.0 เพื่อให้ iPad/มือถือ เข้าดูได้ผ่าน IP เครื่อง
    uvicorn.run(app, host="0.0.0.0", port=8000)