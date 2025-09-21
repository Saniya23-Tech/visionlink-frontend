# ----------------------------- 
# app.py for VisionLink Backend (With Face Detection Data)
# -----------------------------

import os
import sys
import contextlib
import atexit
import threading
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client
import face_recognition
import cv2
from dotenv import load_dotenv
import google.generativeai as genai
import pyttsx3
import asyncio
import websockets
import logging
import warnings

# -----------------------------
# Minimal noise for third-party libs
# -----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = os.getenv('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = os.getenv('TF_ENABLE_ONEDNN_OPTS', '0')

with contextlib.redirect_stdout(open(os.devnull, 'w')):
    with contextlib.redirect_stderr(open(os.devnull, 'w')):
        try:
            import mediapipe as mp
        except Exception:
            mp = None

with contextlib.redirect_stderr(open(os.devnull, 'w')):
    try:
        import face_recognition_models
    except Exception:
        pass

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)5s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("visionlink_improved.log")]
)
logger = logging.getLogger("visionlink")

# -----------------------------
# Env / Config
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
FLASK_PORT = int(os.getenv("FLASK_PORT", 8000))
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8001))

# Tunables with safe defaults
FACE_DISTANCE_THRESHOLD = float(os.getenv("FACE_DISTANCE_THRESHOLD", 0.55))
EMERGENCY_CONF_THRESHOLD = float(os.getenv("EMERGENCY_CONF_THRESHOLD", 0.90))
DEBOUNCE_WINDOW = int(os.getenv("DEBOUNCE_WINDOW", 3))
DEBOUNCE_REQUIRED = int(os.getenv("DEBOUNCE_REQUIRED", 2))
CACHE_DURATION = int(os.getenv("CACHE_DURATION", 30))
MAX_GEMINI_CONCURRENCY = int(os.getenv("MAX_GEMINI_CONCURRENCY", 2))
GEMINI_RETRY = int(os.getenv("GEMINI_RETRY", 1))

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("Missing SUPABASE_URL or SUPABASE_KEY in environment")
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment")

# -----------------------------
# Flask app + WebSocket
# -----------------------------
app = Flask(__name__)
CORS(app)

genai.configure(api_key=GEMINI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

mp_pose = mp.solutions.pose if mp else None
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) if mp_pose else None

# TTS engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 160)
tts_engine.setProperty("volume", 1.0)

def announce_emergency(room, emergency_type):
    def _speak():
        try:
            message = f"Emergency detected in room {room}. Type: {emergency_type}."
            logger.warning("[ALERT] " + message)
            tts_engine.say(message)
            tts_engine.runAndWait()
        except Exception as e:
            logger.exception("TTS error: %s", e)
    threading.Thread(target=_speak, daemon=True).start()

# -----------------------------
# Globals
# -----------------------------
connected_clients = set()
connected_clients_lock = threading.Lock()
stop_flag = False
patients = []
patient_encodings = {}
patient_info_map = {}  # New mapping for quick patient info lookup

# Per-patient recent results for debounce and smoothing:
recent_results = defaultdict(lambda: deque(maxlen=DEBOUNCE_WINDOW))
smoothed_metrics = defaultdict(lambda: {"stress": deque(maxlen=5), "heart": deque(maxlen=5), "stroke": deque(maxlen=5)})

analysis_cache = {}  # simple in-memory cache {cache_key: (timestamp, result)}

# Executor for gemini analysis with limited concurrency
executor = ThreadPoolExecutor(max_workers=MAX_GEMINI_CONCURRENCY)

# Graceful cleanup
def cleanup():
    global stop_flag
    stop_flag = True
    logger.info("Cleanup requested: stopping workers and closing websockets")
    for client in list(connected_clients):
        try:
            asyncio.get_event_loop().run_until_complete(client.close())
        except Exception:
            pass
    executor.shutdown(wait=False)
atexit.register(cleanup)

# -----------------------------
# Helper functions
# -----------------------------
def load_patients():
    try:
        response = supabase.table("patients").select("*").execute()
        patients_data = response.data or []
        encs = {}
        info_map = {}
        
        for p in patients_data:
            fe = p.get("face_encoding")
            if fe:
                try:
                    encs[p['id']] = np.array(json.loads(fe))
                    # Store complete patient info for quick lookup
                    info_map[p['id']] = {
                        'name': p.get('name', 'Unknown'),
                        'room': p.get('room', 'Unknown'),
                        'id': p['id']
                    }
                except Exception as e:
                    logger.debug("Invalid face encoding for patient id %s: %s", p.get('id'), e)
        
        logger.info("Loaded %d patients with face encodings", len(encs))
        return patients_data, encs, info_map
    except Exception as e:
        logger.exception("Failed to load patients: %s", e)
        return [], {}, {}

def get_cache(cache_key):
    item = analysis_cache.get(cache_key)
    if not item:
        return None
    ts, res = item
    if time.time() - ts < CACHE_DURATION:
        return res
    # stale
    analysis_cache.pop(cache_key, None)
    return None

def set_cache(cache_key, result):
    analysis_cache[cache_key] = (time.time(), result)

def merge_smooth(patient_id, result):
    """Store and compute simple moving average for numeric metrics."""
    sm = smoothed_metrics[patient_id]
    sm["stress"].append(result.get("stress_level", 0.0))
    sm["heart"].append(result.get("heart_attack_risk", 0.0))
    sm["stroke"].append(result.get("stroke_risk", 0.0))
    # compute averages
    avg_stress = float(sum(sm["stress"]) / len(sm["stress"])) if sm["stress"] else 0.0
    avg_heart = float(sum(sm["heart"]) / len(sm["heart"])) if sm["heart"] else 0.0
    avg_stroke = float(sum(sm["stroke"]) / len(sm["stroke"])) if sm["stroke"] else 0.0
    # return new result with smoothed values
    new = dict(result)
    new["stress_level"] = avg_stress
    new["heart_attack_risk"] = avg_heart
    new["stroke_risk"] = avg_stroke
    return new

async def broadcast_to_clients(message):
    with connected_clients_lock:
        clients = list(connected_clients)
    if not clients:
        return
    message_str = json.dumps(message)
    coros = [client.send(message_str) for client in clients]
    await asyncio.gather(*coros, return_exceptions=True)

# -----------------------------
# AI Analysis (wrapped for executor)
# -----------------------------
def call_gemini_analyze(image_bytes, patient_info, fall_detected=False):
    """Synchronous wrapper to call Gemini. Includes small retry and robust parsing."""
    prompt = f"""
    Analyze patient {patient_info.get('name')} in Room {patient_info.get('room')} for heart attack, stroke, stress, emotion, fall.
    Return JSON object: emergency_type, confidence, stress_level, heart_attack_risk, stroke_risk, emotion, fall_detected, timestamp
    """
    last_exc = None
    for attempt in range(GEMINI_RETRY + 1):
        try:
            response = genai.GenerativeModel("gemini-pro-vision").generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])
            text = getattr(response, "text", str(response))
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result.setdefault("timestamp", datetime.utcnow().isoformat())
                # Ensure numeric fields exist and are floats
                for k in ["confidence", "stress_level", "heart_attack_risk", "stroke_risk"]:
                    if k in result:
                        try:
                            result[k] = float(result[k])
                        except Exception:
                            result[k] = 0.0
                # Guarantee fall_detected presence
                result["fall_detected"] = bool(result.get("fall_detected", fall_detected))
                return result
            else:
                logger.debug("Gemini returned non-JSON: %s", str(response)[:200])
        except Exception as e:
            last_exc = e
            logger.warning("Gemini attempt %d failed: %s", attempt + 1, e)
            time.sleep(0.5 * (attempt + 1))
    logger.error("Gemini analysis failed after retries: %s", last_exc)
    # Return fallback safe result
    return {
        "emergency_type": "none",
        "confidence": 0.0,
        "stress_level": 0.0,
        "heart_attack_risk": 0.0,
        "stroke_risk": 0.0,
        "emotion": "neutral",
        "fall_detected": bool(fall_detected),
        "timestamp": datetime.utcnow().isoformat(),
    }

# -----------------------------
# Frame processing and orchestration
# -----------------------------
def analyze_frame(frame, patient_info, fall_detected=False):
    """
    Same behaviour as before but uses caching, smoothing and returns the result.
    Non-blocking heavy work is done through ThreadPoolExecutor in callers.
    """
    try:
        # downscale for quick detection; keep original for gemini
        success, buf_small = cv2.imencode(".jpg", cv2.resize(frame, (640, 480)))
        # full image bytes for gemini (optional: could further compress)
        success_full, buf_full = cv2.imencode(".jpg", frame)
        image_bytes = buf_full.tobytes()

        cache_key = f"{patient_info.get('id')}_{fall_detected}"
        cached = get_cache(cache_key)
        if cached:
            return cached

        # Call Gemini (synchronous wrapper)
        result = call_gemini_analyze(image_bytes, patient_info, fall_detected)
        # Smoothing numeric values
        result = merge_smooth(patient_info.get('id'), result)
        set_cache(cache_key, result)
        return result
    except Exception as e:
        logger.exception("analyze_frame error: %s", e)
        return {
            "emergency_type": "none",
            "confidence": 0.0,
            "stress_level": 0.0,
            "heart_attack_risk": 0.0,
            "stroke_risk": 0.0,
            "emotion": "neutral",
            "fall_detected": fall_detected,
            "timestamp": datetime.utcnow().isoformat(),
        }

def should_emit_and_store(patient_id, candidate_result):
    """
    Debounce logic:
    - Append to recent_results[patient_id]
    - Determine if majority of the last N samples indicate an emergency with required confidence
    - Only return True if the debounce rules are met.
    """
    recent_results[patient_id].append(candidate_result)
    window = recent_results[patient_id]
    # Count entries in window that are emergency and high enough confidence
    positives = 0
    for r in window:
        if r.get("emergency_type", "none") != "none" and float(r.get("confidence", 0.0)) >= EMERGENCY_CONF_THRESHOLD:
            positives += 1
    if len(window) < 1:
        return False
    return positives >= DEBOUNCE_REQUIRED

# -----------------------------
# Face recognition + processing
# -----------------------------
def process_frame(frame):
    """
    Enhanced face recognition with better patient identification
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)

    faces = face_recognition.face_locations(small)
    encodings = face_recognition.face_encodings(small, faces)
    results = []

    tasks = []
    futures = []

    # Send face detection data to frontend for visualization
    face_detections = []
    for i, (face, encoding) in enumerate(zip(faces, encodings)):
        best_id, best_dist = None, FACE_DISTANCE_THRESHOLD
        patient_name = "Unknown"
        room = "Unknown"
        
        for pid, known in patient_encodings.items():
            try:
                dist = float(face_recognition.face_distance([known], encoding)[0])
            except Exception:
                continue
            if dist < best_dist:
                best_dist = dist
                best_id = pid
                if pid in patient_info_map:
                    patient_name = patient_info_map[pid].get('name', 'Unknown')
                    room = patient_info_map[pid].get('room', 'Unknown')

        # Add to face detection results for visualization
        top, right, bottom, left = face
        top *= 2; right *= 2; bottom *= 2; left *= 2
        face_detections.append({
            'patient_id': best_id,
            'patient_name': patient_name,
            'room': room,
            'bbox': [int(top), int(right), int(bottom), int(left)],
            'status': 'recognized' if best_id else 'unrecognized'
        })

        if best_id and best_id in patient_info_map:
            patient_info = patient_info_map[best_id]
            results.append({"patient_info": patient_info, "bbox": (top, right, bottom, left)})

            # Quick fall detection using pose if available
            fall_detected = False
            if pose:
                pose_landmarks = pose.process(rgb_frame)
                if pose_landmarks and pose_landmarks.pose_landmarks:
                    lm = pose_landmarks.pose_landmarks.landmark
                    # safe indexes check
                    if len(lm) > 24:
                        shoulder_avg = (lm[11].y + lm[12].y)/2
                        hip_avg = (lm[23].y + lm[24].y)/2
                        fall_detected = abs(shoulder_avg - hip_avg) < 0.1

            # Submit gemini call to executor (limit concurrency)
            # Use a closure to capture frame copy and patient_info
            frame_copy = frame.copy()
            future = executor.submit(analyze_frame, frame_copy, patient_info, fall_detected)
            futures.append((patient_info, future))
        else:
            logger.debug("Unrecognized face or patient not in database")

    # Send face detection data to frontend
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    face_detection_payload = {
        "type": "face_detection",
        "faces": face_detections,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if loop.is_running():
        asyncio.create_task(broadcast_to_clients(face_detection_payload))
    else:
        try:
            loop.run_until_complete(broadcast_to_clients(face_detection_payload))
        except Exception as e:
            logger.exception("Face detection broadcast failed: %s", e)

    # Gather results (wait for futures to complete but don't block indefinitely)
    for patient_info, fut in futures:
        try:
            result = fut.result(timeout=20)  # wait up to 20s for a gemini result
        except Exception as e:
            logger.warning("Gemini task failed or timed out for patient %s: %s", patient_info.get('id'), e)
            result = {
                "emergency_type": "none",
                "confidence": 0.0,
                "stress_level": 0.0,
                "heart_attack_risk": 0.0,
                "stroke_risk": 0.0,
                "emotion": "neutral",
                "fall_detected": False,
                "timestamp": datetime.utcnow().isoformat()
            }

        # Debounce & persist logic
        pid = str(patient_info.get('id'))
        if should_emit_and_store(pid, result):
            # Persist to supabase
            try:
                supabase.table("emergencies").insert({
                    "patient_id": patient_info['id'],
                    **result
                }).execute()
            except Exception as e:
                logger.exception("Supabase insert failed for patient %s: %s", pid, e)

            # Broadcast
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            payload = {
                "patient_id": patient_info['id'],
                "patient_name": patient_info.get("name", "Unknown"),
                "room": patient_info.get("room", "Unknown"),
                **result
            }
            
            # Log the payload for debugging
            logger.info("Broadcasting patient data: %s", payload)
            
            if loop.is_running():
                asyncio.create_task(broadcast_to_clients(payload))
            else:
                try:
                    loop.run_until_complete(broadcast_to_clients(payload))
                except Exception as e:
                    logger.exception("Broadcast failed: %s", e)

            # Announce if critical
            try:
                if result["emergency_type"] != "none" and result.get("confidence", 0) >= EMERGENCY_CONF_THRESHOLD:
                    announce_emergency(patient_info.get("room", "Unknown"), result["emergency_type"])
            except Exception:
                logger.exception("Announce error")

    return results

# -----------------------------
# WebSocket handlers
# -----------------------------
async def register_client(websocket):
    with connected_clients_lock:
        connected_clients.add(websocket)
    try:
        await websocket.send(json.dumps({"type": "connected", "message": "Connected to VisionLink emergency monitoring"}))
        await websocket.wait_closed()
    except Exception as e:
        logger.debug("WebSocket client error: %s", e)
    finally:
        with connected_clients_lock:
            if websocket in connected_clients:
                connected_clients.remove(websocket)

async def websocket_server():
    async with websockets.serve(register_client, "0.0.0.0", WEBSOCKET_PORT):
        logger.info("WebSocket server started on port %s", WEBSOCKET_PORT)
        await asyncio.Future()  # run forever

def start_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(websocket_server())
    except Exception as e:
        logger.exception("WebSocket server failed: %s", e)

# -----------------------------
# Camera monitor & Flask endpoints
# -----------------------------
def monitor_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Camera %s not found", camera_index)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    logger.info("Started monitoring camera %s", camera_index)
    frame_count = 0
    process_every_n_frames = int(os.getenv("PROCESS_EVERY_N_FRAMES", 3))

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera %s", camera_index)
            time.sleep(1)
            continue
        frame_count += 1
        if frame_count % process_every_n_frames == 0:
            try:
                process_frame(frame)
            except Exception as e:
                logger.exception("Error processing frame: %s", e)
        time.sleep(0.03)

    cap.release()
    logger.info("Stopped monitoring camera %s", camera_index)

@app.route("/api/start-monitoring", methods=["POST"])
def start_monitoring():
    camera_index = request.json.get("camera_index", 0)
    threading.Thread(target=monitor_camera, args=(camera_index,), daemon=True).start()
    return jsonify({"status": "success", "message": f"Monitoring started on camera {camera_index}"})

@app.route("/api/stop-monitoring", methods=["POST"])
def stop_monitoring():
    global stop_flag
    stop_flag = True
    return jsonify({"status": "success", "message": "Monitoring stopped"})

@app.route("/api/emergencies", methods=["GET"])
def get_emergencies():
    try:
        response = supabase.table("emergencies").select("*").order("timestamp", desc=True).execute()
        return jsonify(response.data)
    except Exception as e:
        logger.exception("Error fetching emergencies: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/patients", methods=["GET"])
def get_patients():
    try:
        response = supabase.table("patients").select("*").execute()
        return jsonify(response.data)
    except Exception as e:
        logger.exception("Error fetching patients: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/refresh-patients", methods=["POST"])
def refresh_patients():
    global patients, patient_encodings, patient_info_map
    patients, patient_encodings, patient_info_map = load_patients()
    return jsonify({"status": "success", "message": f"Loaded {len(patients)} patients"})

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Backend running",
        "patients_loaded": len(patients),
        "patients_with_faces": len(patient_encodings),
        "connected_clients": len(connected_clients),
    })

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    patients, patient_encodings, patient_info_map = load_patients()
    logger.info("Loaded %d patients on startup (%d with face encodings)", len(patients), len(patient_encodings))

    # start websocket server thread
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()

    # start flask app (debug=False for production)
    logger.info("Starting Flask server on port %s", FLASK_PORT)
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=True, use_reloader=False)