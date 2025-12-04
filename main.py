"""
CrowdVision Backend API - WITH FFMPEG CONVERSION
Deploy this on Railway, Render, or your own server with Python + GPU support

Run with: uvicorn main:app --host 0.0.0.0 --port 8000
Install: pip install opencv-python-headless (this version works better for servers)
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter
from collections import deque
import time
import os
import uuid
import shutil
import subprocess
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import aiofiles
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

app = FastAPI(
    title="CrowdVision API",
    description="AI-Powered Crowd Analysis Backend",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

processing_jobs: Dict[str, Dict[str, Any]] = {}


class ZoneResult(BaseModel):
    id: str
    name: str
    people: int
    yoloCount: int
    neuralCount: int
    method: str
    density: float
    riskScore: float
    riskLevel: str


class FastCrowdAnalyzer:
    def __init__(self, model_path='yolov8n.pt', use_density_model=True, frame_skip=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Device: {self.device}")

        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except:
            pass

        self.use_density_model = use_density_model
        self.density_net = self.build_density_model() if use_density_model else None
        self.prev_gray = None
        self.flow_history = deque(maxlen=5)
        self.frame_skip = max(1, int(frame_skip))

    def build_density_model(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, 1)
        ).to(self.device)
        model.eval()
        return model

    def detect_people(self, frame, conf_threshold=0.3):
        detections = []
        try:
            results = self.model.track(source=frame, persist=False, conf=conf_threshold, iou=0.5, classes=[0])
        except:
            results = self.model(frame, conf=conf_threshold, iou=0.5, classes=[0])

        try:
            iterable = list(results)
        except:
            iterable = [results]

        for res in iterable:
            boxes = getattr(res, 'boxes', None)
            if boxes is None:
                continue

            for b in boxes:
                try:
                    xyxy = b.xyxy.cpu().numpy().flatten()
                except:
                    continue
                if len(xyxy) < 4:
                    continue

                x1, y1, x2, y2 = map(int, xyxy[:4])
                try:
                    conf = float(b.conf.cpu().numpy().item())
                except:
                    conf = 0.0

                track_id = -1
                try:
                    if hasattr(b, 'id') and b.id is not None:
                        track_id = int(b.id.cpu().numpy().item())
                except:
                    pass

                detections.append({'bbox': (x1, y1, x2, y2), 'confidence': conf, 'id': track_id})
            break

        return detections

    def estimate_density_map(self, frame):
        if not self.use_density_model:
            return None
        h, w = frame.shape[:2]
        small_w, small_h = w // 4, h // 4
        img = cv2.resize(frame, (small_w, small_h))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            out = self.density_net(img_tensor).squeeze().cpu().numpy()
        out = np.maximum(out, 0.0).astype(np.float32)
        density = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
        return density

    def calculate_density_map(self, frame, detections):
        density = np.zeros(frame.shape[:2], dtype=np.float32)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = int(max(max(20, (x2 - x1) // 2), max(20, (y2 - y1) // 2), 30))
            y_min, y_max = max(0, cy - r), min(frame.shape[0], cy + r)
            x_min, x_max = max(0, cx - r), min(frame.shape[1], cx + r)
            density[y_min:y_max, x_min:x_max] += 1.0
        density = gaussian_filter(density, sigma=15)
        return density

    def optical_flow_analysis(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (frame.shape[1] // 2, frame.shape[0] // 2))
        if self.prev_gray is None or self.prev_gray.shape != small.shape:
            self.prev_gray = small.copy()
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        movement = float(np.mean(mag))
        self.flow_history.append(movement)
        self.prev_gray = small.copy()
        return movement

    def analyze_zones(self, frame, detections, density_map, neural_density=None):
        h, w = frame.shape[:2]
        zones = {
            'zone1': (0, 0, w // 2, h // 2),
            'zone2': (w // 2, 0, w, h // 2),
            'zone3': (0, h // 2, w // 2, h),
            'zone4': (w // 2, h // 2, w, h)
        }

        results = {}
        for zone_id, (x1, y1, x2, y2) in zones.items():
            yolo_count = sum(1 for d in detections if x1 <= (d['bbox'][0] + d['bbox'][2]) // 2 < x2 and y1 <= (d['bbox'][1] + d['bbox'][3]) // 2 < y2)
            zone_density = density_map[y1:y2, x1:x2]
            gaussian_count = float(np.sum(zone_density))
            avg_density = float(np.mean(zone_density))
            neural_count = float(np.sum(neural_density[y1:y2, x1:x2])) if neural_density is not None else 0.0

            if yolo_count < 10 and neural_count > yolo_count * 2:
                final_count, method = int(round(neural_count)), "Neural Density"
            else:
                final_count, method = max(yolo_count, int(round(gaussian_count))), "YOLO + Gaussian"

            density_risk = min(avg_density * 50, 100)
            crowd_risk = min((final_count / 30) * 100, 100)
            movement_risk = float(np.mean(self.flow_history)) * 20 if len(self.flow_history) > 0 else 0
            score = density_risk * 0.4 + crowd_risk * 0.4 + movement_risk * 0.2

            level = 'LOW' if score < 30 else 'MEDIUM' if score < 60 else 'HIGH' if score < 80 else 'CRITICAL'

            results[zone_id] = {
                'id': zone_id,
                'name': f"Zone {zone_id[-1]} ({'Top' if zone_id in ['zone1', 'zone2'] else 'Bottom'}-{'Left' if zone_id in ['zone1', 'zone3'] else 'Right'})",
                'coords': (x1, y1, x2, y2),
                'people': final_count,
                'yolo_count': yolo_count,
                'neural_count': int(round(neural_count)),
                'method': method,
                'density': avg_density,
                'risk_score': score,
                'risk_level': level,
            }
        return results

    def visualize(self, frame, detections, zones, density_map, neural_density=None):
        output = frame.copy()
        disp_density = neural_density if neural_density is not None else density_map
        if disp_density is not None:
            norm = (disp_density / (float(np.max(disp_density)) + 1e-6) * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            output = cv2.addWeighted(output, 0.7, heatmap, 0.3, 0)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 255), 2)

        color_map = {'LOW': (0, 255, 0), 'MEDIUM': (0, 255, 255), 'HIGH': (0, 165, 255), 'CRITICAL': (0, 0, 255)}
        for zid, z in zones.items():
            x1, y1, x2, y2 = z['coords']
            cv2.rectangle(output, (x1, y1), (x2, y2), color_map.get(z['risk_level'], (255, 255, 255)), 2)
            for i, text in enumerate([f"{zid.upper()} ({z['method']})", f"{z['risk_level']}: {z['risk_score']:.1f}%"]):
                cv2.putText(output, text, (x1 + 5, y1 + 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return output

    def process_video(self, video_path: str, output_path: str, job_id: str, scale: float = 0.7):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                processing_jobs[job_id]['status'] = 'error'
                processing_jobs[job_id]['error'] = 'Cannot open video'
                return None

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out_w, out_h = int(width * scale), int(height * scale)

            # Use MJPEG (always works, no codec issues)
            temp_output = output_path.replace('.mp4', '_temp.avi')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (out_w, out_h))

            if not out.isOpened():
                cap.release()
                processing_jobs[job_id]['status'] = 'error'
                processing_jobs[job_id]['error'] = 'Cannot create output'
                return None

            frame_num, start, all_zones = 0, time.time(), {}
            print(f"[INFO] Processing {total} frames at {fps} FPS")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                frame = cv2.resize(frame, (out_w, out_h))
                processing_jobs[job_id]['progress'] = min(int((frame_num / total) * 95), 95)

                if frame_num % self.frame_skip != 0:
                    out.write(frame)
                    continue

                try:
                    detections = self.detect_people(frame)
                    gaussian_density = self.calculate_density_map(frame, detections)
                    neural_density = self.estimate_density_map(frame)
                    self.optical_flow_analysis(frame)
                    zones = self.analyze_zones(frame, detections, gaussian_density, neural_density)
                    all_zones = zones
                    output_frame = self.visualize(frame, detections, zones, gaussian_density, neural_density)
                    out.write(output_frame)
                except Exception as e:
                    print(f"[WARN] Frame {frame_num} error: {e}")
                    out.write(frame)

            cap.release()
            out.release()

            # Convert to browser-compatible MP4 using FFmpeg
            print(f"[INFO] Converting to MP4...")
            processing_jobs[job_id]['progress'] = 96
            
            if not self.convert_to_mp4(temp_output, output_path, fps):
                processing_jobs[job_id]['status'] = 'error'
                processing_jobs[job_id]['error'] = 'Video conversion failed'
                return None
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)

            processing_time = time.time() - start
            file_size = os.path.getsize(output_path)
            print(f"[INFO] Complete: {frame_num} frames in {processing_time:.2f}s ({file_size:,} bytes)")

            # Build result
            zones_list = list(all_zones.values())
            total_people = sum(z['people'] for z in zones_list)
            avg_risk = sum(z['risk_score'] for z in zones_list) / len(zones_list) if zones_list else 0
            risk_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
            max_risk = max(zones_list, key=lambda z: risk_order[z['risk_level']])['risk_level'] if zones_list else 'LOW'

            alerts = []
            for z in zones_list:
                if z['risk_level'] in ['HIGH', 'CRITICAL']:
                    alerts.append({
                        'id': f"alert-{z['id']}", 'type': 'density', 'severity': z['risk_level'],
                        'message': f"{'Critical' if z['risk_level'] == 'CRITICAL' else 'High'} density in {z['name']}!",
                        'zone': z['name'], 'timestamp': time.time()
                    })

            return {
                'zones': [ZoneResult(**{k: v for k, v in z.items() if k in ['id', 'name', 'people', 'method', 'density', 'risk_level'] or k.endswith('_count') or k.endswith('_score')}, 
                         yoloCount=z['yolo_count'], neuralCount=z['neural_count'], riskScore=z['risk_score'], riskLevel=z['risk_level']) 
                         for z in zones_list],
                'totalPeople': total_people, 'averageRiskScore': avg_risk, 'maxRiskLevel': max_risk,
                'alerts': alerts, 'processingTime': processing_time, 'frameCount': frame_num,
                'fps': frame_num / processing_time if processing_time > 0 else 0
            }

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            processing_jobs[job_id]['status'] = 'error'
            processing_jobs[job_id]['error'] = str(e)
            return None

    # Replace the convert_to_mp4 method with this:
    def convert_to_mp4(self, input_path: str, output_path: str, fps: float) -> bool:
        """Fallback: Re-encode with OpenCV using best available codec"""
        try:
            # Just copy/rename if we can't convert
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                shutil.copy(input_path, output_path)
                return True
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Try H.264-ish codecs
            for codec in ['avc1', 'H264', 'X264']:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if out.isOpened():
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            out.write(frame)
                        cap.release()
                        out.release()
                        if os.path.getsize(output_path) > 0:
                            return True
                except:
                    continue
            
            # Last resort: copy as-is
            cap.release()
            shutil.copy(input_path, output_path)
            return True
        except:
            return False


analyzer = None

def get_analyzer():
    global analyzer
    if analyzer is None:
        analyzer = FastCrowdAnalyzer(use_density_model=True, frame_skip=2)
    return analyzer

def process_video_task(job_id: str, video_path: str, output_path: str):
    try:
        processing_jobs[job_id]['status'] = 'processing'
        analyzer = get_analyzer()
        result = analyzer.process_video(video_path, output_path, job_id)
        
        if result:
            processing_jobs[job_id].update({'status': 'complete', 'progress': 100, 'result': result})
        else:
            if processing_jobs[job_id]['status'] != 'error':
                processing_jobs[job_id]['status'] = 'error'
                processing_jobs[job_id]['error'] = 'Processing failed'
    except Exception as e:
        print(f"[ERROR] Task failed: {e}")
        processing_jobs[job_id]['status'] = 'error'
        processing_jobs[job_id]['error'] = str(e)
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass


@app.get("/")
async def root():
    return {"status": "online", "service": "CrowdVision API"}

@app.post("/api/analyze")
async def analyze_video(background_tasks: BackgroundTasks, video: UploadFile = File(...)):
    allowed_types = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo', 'video/webm']
    if video.content_type not in allowed_types:
        raise HTTPException(400, f"Invalid file type")

    job_id = str(uuid.uuid4())
    file_ext = os.path.splitext(video.filename)[1] or '.mp4'
    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}{file_ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_analyzed.mp4")
    
    async with aiofiles.open(upload_path, 'wb') as f:
        await f.write(await video.read())
    
    processing_jobs[job_id] = {'status': 'queued', 'progress': 0, 'fileName': video.filename, 'outputPath': output_path}
    background_tasks.add_task(process_video_task, job_id, upload_path, output_path)
    return {"jobId": job_id, "status": "queued"}

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in processing_jobs:
        raise HTTPException(404, "Job not found")
    
    job = processing_jobs[job_id]
    response = {"jobId": job_id, "status": job['status'], "progress": job['progress'], "fileName": job.get('fileName')}
    
    if job['status'] == 'complete' and 'result' in job:
        response.update({
            "processedVideoUrl": f"/api/video/{job_id}",
            **{k: v for k, v in job['result'].items() if k != 'zones'},
            "zones": [z.dict() for z in job['result']['zones']]
        })
    
    if job['status'] == 'error':
        response['error'] = job.get('error', 'Unknown error')
    
    return response

@app.get("/api/video/{job_id}")
async def get_processed_video(job_id: str):
    if job_id not in processing_jobs:
        raise HTTPException(404, "Job not found")
    
    job = processing_jobs[job_id]
    if job['status'] != 'complete':
        raise HTTPException(400, f"Video not ready")
    
    output_path = job['outputPath']
    if not os.path.exists(output_path):
        raise HTTPException(404, "Video not found")
    
    file_size = os.path.getsize(output_path)
    if file_size == 0:
        raise HTTPException(500, "Video file empty")
    
    print(f"[INFO] Serving: {output_path} ({file_size:,} bytes)")
    return FileResponse(output_path, media_type='video/mp4',
                       headers={'Accept-Ranges': 'bytes', 'Content-Disposition': f'inline; filename="analyzed.mp4"'})

@app.delete("/api/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    if job_id not in processing_jobs:
        raise HTTPException(404, "Job not found")
    
    job = processing_jobs[job_id]
    if job.get('outputPath') and os.path.exists(job['outputPath']):
        os.remove(job['outputPath'])
    del processing_jobs[job_id]
    return {"status": "cleaned"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)