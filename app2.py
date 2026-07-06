import os
import asyncio
import io
import traceback
import cv2
import pyaudio
import PIL.Image
from PIL import ImageGrab 
import threading
import http.server
import socketserver
import time
import json
import urllib.parse
import socket

try:
    import psutil
except ImportError:
    print("[ERROR] Please install psutil: pip install psutil")
    exit(1)

try:
    import webview
except ImportError:
    print("[ERROR] Please install pywebview for the desktop app: pip install pywebview")
    exit(1)

try:
    import numpy as np
except ImportError:
    print("[ERROR] Please install numpy for voice interruption: pip install numpy")
    exit(1)

from google import genai
from google.genai import types

# =====================================================================
# GLOBAL VARIABLES FOR SYSTEM STATE & COMMLINK
# =====================================================================
LATEST_JPEG = b""  
TARGET_CAM_SOURCE = "off"
TEXT_PROMPT_QUEUE = []  
AI_CHAT_QUEUE = []      
CURRENT_LATENCY = 0
CAMERA_RUNNING = True
MIC_MUTED = False # Only controls the microphone now

# =====================================================================
# AUDIO & AI SETTINGS
# =====================================================================
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MODEL = "models/gemini-3.1-flash-live-preview"

# --- API CLIENT ---
client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key="AIzaSyDyEBB-QcsNMkwFWbz-_7PncNTpz6gVb0A", 
)

# ---------------------------------------------------------------------
# NEXUS SYSTEM OVERRIDE PROMPT
# ---------------------------------------------------------------------
CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"], 
    system_instruction=types.Content(
        parts=[
            types.Part.from_text(
                text=(
                    "You are NEXUS, an advanced AI assistant designed for Shashank Gowda NB. "
                    "You have real-time multimodal capabilities via a visual feed. "
                    "CRITICAL VISION RULES: "
                    "1. If the user asks what you see, or asks for suggestions based on the screen (like YouTube or WhatsApp), you MUST look at the current visual feed and accurately describe it. "
                    "2. Do not say 'I am a language model' or 'I cannot see'. "
                    "3. High priority rule: always answer typed keyboard messages out loud immediately."
                )
            )
        ]
    ),
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=104857,
        sliding_window=types.SlidingWindow(target_tokens=52428),
    ),
)

pya = pyaudio.PyAudio()

# =====================================================================
# DESKTOP WINDOW API (PyWebView)
# =====================================================================
webview_window = None

class WebviewAPI:
    def enter_mini_mode(self):
        global TARGET_CAM_SOURCE
        TARGET_CAM_SOURCE = "screen"
        if webview_window:
            webview_window.resize(380, 480)
            webview_window.on_top = True

    def exit_mini_mode(self):
        global TARGET_CAM_SOURCE, CAMERA_RUNNING
        TARGET_CAM_SOURCE = "0" if CAMERA_RUNNING else "off"
        if webview_window:
            webview_window.resize(1280, 800)
            webview_window.on_top = False

    def toggle_mute(self):
        global MIC_MUTED
        MIC_MUTED = not MIC_MUTED

    def send_message(self, text):
        if text.strip():
            TEXT_PROMPT_QUEUE.append(text.strip())

    def set_camera(self, src):
        global TARGET_CAM_SOURCE
        TARGET_CAM_SOURCE = src

    def terminate(self):
        print("\n[SYSTEM] Termination command executed. Graceful exit.")
        os._exit(0)


# =====================================================================
# HARDWARE, NETWORK & DEDICATED CAMERA/SCREEN THREAD
# =====================================================================
def measure_latency():
    global CURRENT_LATENCY
    while True:
        try:
            start = time.time()
            socket.create_connection(('8.8.8.8', 53), timeout=2)
            CURRENT_LATENCY = int((time.time() - start) * 1000)
        except Exception:
            CURRENT_LATENCY = -1
        time.sleep(2)

def camera_worker():
    global LATEST_JPEG, TARGET_CAM_SOURCE, CAMERA_RUNNING
    cap = None
    current_src = "off"
    
    while CAMERA_RUNNING:
        try:
            if current_src != TARGET_CAM_SOURCE:
                if cap is not None:
                    cap.release()
                    cap = None
                
                current_src = TARGET_CAM_SOURCE
                LATEST_JPEG = b""
                
                if current_src not in ["off", "screen"]:
                    src_val = int(current_src) if str(current_src).isdigit() else current_src
                    if isinstance(src_val, int):
                        cap = cv2.VideoCapture(src_val, cv2.CAP_DSHOW)
                        if not cap.isOpened():
                            cap = cv2.VideoCapture(src_val)
                    else:
                        cap = cv2.VideoCapture(src_val)
                    time.sleep(1.0) 
            
            if current_src == "off":
                LATEST_JPEG = b""
                time.sleep(0.1)
                continue
                
            if current_src == "screen":
                # Screen Capture Logic for Mini-Mode
                img = ImageGrab.grab()
                img.thumbnail([800, 600]) 
                bio = io.BytesIO()
                img.save(bio, format="jpeg", quality=75)
                LATEST_JPEG = bio.getvalue()
                time.sleep(0.5) 
                continue

            if cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = PIL.Image.fromarray(frame_rgb)
                    img.thumbnail([640, 480])
                    bio = io.BytesIO()
                    img.save(bio, format="jpeg")
                    LATEST_JPEG = bio.getvalue()
                else:
                    time.sleep(0.05)
            else:
                time.sleep(0.1)
                
        except Exception as e:
            time.sleep(0.5)

    if cap is not None:
        cap.release()

def get_hardware_stats():
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    batt = psutil.sensors_battery()
    batt_str = f"{batt.percent}%" if batt else "AC"
    
    net_status = "OFFLINE"
    if CURRENT_LATENCY > 0 and CURRENT_LATENCY < 80:
        net_status = "EXCELLENT"
    elif CURRENT_LATENCY >= 80 and CURRENT_LATENCY < 200:
        net_status = "GOOD"
    elif CURRENT_LATENCY >= 200:
        net_status = "POOR"

    return json.dumps({
        "cpu": cpu,
        "ram_used": round(ram.used / (1024**3), 2),
        "ram_total": round(ram.total / (1024**3), 2),
        "battery": batt_str,
        "ping": f"{CURRENT_LATENCY}ms" if CURRENT_LATENCY != -1 else "ERR",
        "net_status": net_status,
        "audio_muted": MIC_MUTED
    })

# =====================================================================
# EMBEDDED HTML DASHBOARD (EDGE-TO-EDGE CINEMATIC + NATIVE MINI MODE)
# =====================================================================
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>NEXUS COMMAND CENTER</title>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@200;300;400;500;600&family=Inter:wght@300;400;500&family=Space+Mono&display=swap" rel="stylesheet">
    
    <style>
        :root { 
            --bg-base: #020202; 
            --bg-panel: rgba(10, 10, 10, 0.85);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --gold-primary: #C5A059; 
            --gold-light: #F3E5AB;
            --text-primary: #FFFFFF; 
            --text-secondary: #7A7A7A;
            --danger: #E53935;
            --success: #00E676;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body { 
            background-color: var(--bg-base); color: var(--text-primary); 
            font-family: 'Inter', sans-serif; height: 100vh; overflow: hidden; 
            background-image: radial-gradient(circle at center, #0a0a0c 0%, #000 100%);
            -webkit-font-smoothing: antialiased;
        }

        h1, h2, h3, .brand-font { font-family: 'Montserrat', sans-serif; }
        .tech-font { font-family: 'Space Mono', monospace; }
        
        .overlay-screen {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            z-index: 1000; transition: opacity 1s ease; background: #020202;
        }
        
        .auth-card {
            background: rgba(5, 5, 5, 0.8); border: 1px solid rgba(197, 160, 89, 0.2);
            padding: 60px 50px; border-radius: 8px; text-align: center; width: 90%; max-width: 420px; 
            box-shadow: 0 0 50px rgba(0,0,0,1), inset 0 0 30px rgba(197,160,89,0.05);
            backdrop-filter: blur(20px); z-index: 1;
        }
        
        .auth-title { font-size: 2rem; font-weight: 200; letter-spacing: 12px; margin-bottom: 40px; color: var(--gold-light); }
        
        .luxury-input {
            width: 100%; background: transparent; border: none; border-bottom: 1px solid rgba(255,255,255,0.2);
            color: var(--text-primary); font-size: 0.9rem; padding: 12px 0; font-family: 'Space Mono', monospace;
            text-align: center; margin-bottom: 40px; outline: none; transition: border-color 0.3s; letter-spacing: 3px;
        }
        .luxury-input:focus { border-bottom-color: var(--gold-primary); }
        
        .btn-auth {
            width: 100%; background: transparent; color: var(--gold-primary); border: 1px solid var(--gold-primary);
            padding: 16px; font-size: 0.8rem; font-weight: 500; letter-spacing: 4px;
            cursor: pointer; border-radius: 4px; font-family: 'Montserrat', sans-serif; transition: all 0.3s;
        }
        .btn-auth:hover { background: var(--gold-primary); color: #000; box-shadow: 0 0 25px rgba(197, 160, 89, 0.4); }
        
        #main-dashboard { display: none; height: 100vh; width: 100vw; opacity: 0; transition: opacity 1s ease; display: flex; }
        
        .sidebar {
            width: 320px; background: var(--bg-panel); border-right: 1px solid var(--border-subtle);
            display: flex; flex-direction: column; padding: 30px; backdrop-filter: blur(10px);
        }
        
        .nav-brand { font-size: 1.5rem; font-weight: 300; letter-spacing: 8px; color: var(--gold-primary); margin-bottom: 40px; text-align: center;}
        
        .panel-title { font-size: 0.75rem; color: var(--gold-primary); letter-spacing: 4px; border-bottom: 1px solid var(--border-subtle); padding-bottom: 10px; margin-bottom: 15px; margin-top: 20px;}
        .stat-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 0.85rem;}
        .stat-label { color: var(--text-secondary); letter-spacing: 2px; font-size: 0.75rem;}
        .stat-value { font-weight: 500; font-size: 0.9rem;}
        
        .btn-terminate {
            background: transparent; border: 1px solid var(--danger); color: var(--danger);
            padding: 12px; width: 100%; font-size: 0.75rem; letter-spacing: 3px; cursor: pointer;
            font-family: 'Montserrat', sans-serif; border-radius: 4px; transition: all 0.3s; margin-top: auto;
        }
        .btn-terminate:hover { background: var(--danger); color: #fff; }

        .main-content { flex: 1; display: flex; flex-direction: column; position: relative; }

        .top-bar {
            height: 70px; border-bottom: 1px solid var(--border-subtle); display: flex;
            align-items: center; justify-content: flex-end; padding: 0 30px; gap: 20px;
        }

        .btn-action {
            background: transparent; border: 1px solid var(--gold-primary); color: var(--gold-primary);
            padding: 10px 20px; border-radius: 4px; font-family: 'Montserrat', sans-serif;
            font-size: 0.75rem; letter-spacing: 2px; font-weight: 500; cursor: pointer; transition: all 0.3s;
        }
        .btn-action:hover { background: rgba(197, 160, 89, 0.1); box-shadow: 0 0 15px rgba(197, 160, 89, 0.3); }

        .btn-mini-mode { border-color: var(--text-secondary); color: var(--text-primary); }
        .btn-mini-mode:hover { border-color: #fff; background: rgba(255,255,255,0.1); box-shadow: 0 0 15px rgba(255,255,255,0.2); }

        .visual-stage { flex: 1; display: flex; justify-content: center; align-items: center; position: relative; overflow: hidden; }

        .holo-scene { width: 400px; height: 400px; perspective: 1000px; display: flex; justify-content: center; align-items: center; }
        .holo-sphere { position: relative; width: 100%; height: 100%; transform-style: preserve-3d; animation: rotateSphere 20s linear infinite; display: flex; justify-content: center; align-items: center; }
        .orbit { position: absolute; border-radius: 50%; transform-style: preserve-3d; }
        .orbit-1 { width: 350px; height: 350px; border: 1px solid rgba(197, 160, 89, 0.2); border-top: 2px solid var(--gold-primary); animation: spinX 8s linear infinite; }
        .orbit-2 { width: 280px; height: 280px; border: 1px dashed rgba(255, 255, 255, 0.3); animation: spinY 12s linear infinite; }
        .orbit-3 { width: 200px; height: 200px; border: 2px dotted var(--gold-light); animation: spinZ 15s linear infinite; opacity: 0.7; }
        
        @keyframes rotateSphere { 0% { transform: rotateX(60deg) rotateZ(0); } 100% { transform: rotateX(60deg) rotateZ(360deg); } }
        @keyframes spinX { 0% { transform: rotateX(0deg); } 100% { transform: rotateX(360deg); } }
        @keyframes spinY { 0% { transform: rotateY(0deg); } 100% { transform: rotateY(360deg); } }
        @keyframes spinZ { 0% { transform: rotateZ(0deg); } 100% { transform: rotateZ(360deg); } }

        #liveFeed { display: none; width: 100%; height: 100%; object-fit: contain; position: absolute; top:0; left:0; z-index: 5; background: #000; }
        .subtitle-overlay { position: absolute; bottom: 30px; text-align: center; width: 80%; font-family: 'Montserrat', sans-serif; font-size: 1.2rem; font-weight: 300; color: #fff; text-shadow: 0 2px 10px rgba(0,0,0,1); z-index: 10; }

        .terminal-bar {
            height: 200px; border-top: 1px solid var(--border-subtle); display: flex;
            background: var(--bg-panel); backdrop-filter: blur(10px);
        }
        
        .chat-log { flex: 1; overflow-y: auto; padding: 20px; font-size: 0.85rem; font-weight: 300; display: flex; flex-direction: column; gap: 10px; }
        .chat-log::-webkit-scrollbar { width: 4px; }
        .chat-log::-webkit-scrollbar-thumb { background: rgba(197, 160, 89, 0.3); }
        .ai-msg { color: var(--gold-light); border-left: 2px solid var(--gold-primary); padding-left: 10px;}
        .user-msg { color: #fff; border-left: 2px solid var(--border-subtle); padding-left: 10px;}
        
        .cmd-section { width: 400px; border-left: 1px solid var(--border-subtle); padding: 20px; display: flex; flex-direction: column; gap: 15px; justify-content: center;}
        .cmd-input { width: 100%; background: rgba(0,0,0,0.5); border: 1px solid var(--border-subtle); padding: 15px; border-radius: 4px; color: var(--text-primary); font-family: 'Space Mono', monospace; font-size: 0.8rem; }
        .cmd-input:focus { outline: none; border-color: var(--gold-primary); }

        /* NATIVE APP MINI MODE */
        #mini-dashboard { display: none; height: 100vh; width: 100vw; background: #020202; align-items: center; justify-content: center; }
        
        .mini-card {
            background: rgba(10, 10, 10, 0.95); border: 1px solid var(--gold-primary);
            border-radius: 12px; width: 340px; padding: 30px; box-shadow: 0 20px 50px rgba(0,0,0,0.8);
            display: flex; flex-direction: column; align-items: center; gap: 20px; position: relative;
        }

        .mini-title { font-family: 'Montserrat', sans-serif; font-size: 0.8rem; letter-spacing: 4px; color: var(--gold-primary); text-align: center; }

        .wave-container { display: flex; justify-content: center; align-items: center; gap: 5px; height: 60px; }
        .wave-bar { width: 6px; background: var(--gold-light); border-radius: 3px; animation: wave 1.2s ease-in-out infinite; }
        .wave-bar:nth-child(1) { height: 20px; animation-delay: 0.0s; }
        .wave-bar:nth-child(2) { height: 40px; animation-delay: 0.1s; }
        .wave-bar:nth-child(3) { height: 60px; animation-delay: 0.2s; }
        .wave-bar:nth-child(4) { height: 40px; animation-delay: 0.3s; }
        .wave-bar:nth-child(5) { height: 20px; animation-delay: 0.4s; }
        
        .wave-container.muted .wave-bar { animation: none; height: 10px; background: var(--danger); box-shadow: none; }
        @keyframes wave { 0%, 100% { height: 15px; box-shadow: 0 0 5px transparent; } 50% { height: 50px; box-shadow: 0 0 15px var(--gold-light); } }

        .mini-transcript { font-size: 0.85rem; color: #fff; text-align: center; height: 60px; overflow: hidden; display: flex; align-items: center; justify-content: center; font-weight: 300;}

        .mini-controls { display: flex; gap: 15px; width: 100%; }
        .btn-mini-mute { flex: 1; background: transparent; border: 1px solid var(--success); color: var(--success); padding: 12px; border-radius: 6px; cursor: pointer; transition: all 0.3s; font-family: 'Montserrat', sans-serif; font-size: 0.75rem; letter-spacing: 2px; }
        .btn-mini-mute.muted { border-color: var(--danger); color: var(--danger); }
        
        .btn-mini-back { flex: 1; background: transparent; border: 1px solid var(--border-subtle); color: #fff; padding: 12px; border-radius: 6px; cursor: pointer; transition: all 0.3s; font-family: 'Montserrat', sans-serif; font-size: 0.75rem; letter-spacing: 2px; }
        .btn-mini-back:hover { background: rgba(255,255,255,0.1); }
    </style>
</head>
<body>

    <div id="auth-screen" class="overlay-screen">
        <div class="auth-card">
            <h1 class="brand-font auth-title">NEXUS</h1>
            <input type="text" id="usernameInput" class="luxury-input" placeholder="ENTER CREDENTIALS" autocomplete="off" onkeypress="if(event.key === 'Enter') initSystem()">
            <button class="btn-auth" onclick="initSystem()">INITIALIZE PROTOCOL</button>
        </div>
    </div>

    <div id="main-dashboard">
        <aside class="sidebar">
            <div class="nav-brand">NEXUS</div>
            
            <div class="panel-title">DIAGNOSTICS</div>
            <div class="stat-row"><span class="stat-label">COMPUTE</span><span class="stat-value tech-font" id="stat-cpu">--%</span></div>
            <div class="stat-row"><span class="stat-label">MEMORY</span><span class="stat-value tech-font" id="stat-ram">--GB</span></div>
            <div class="stat-row"><span class="stat-label">POWER</span><span class="stat-value tech-font" id="stat-batt">--</span></div>
            
            <div class="panel-title">UPLINK</div>
            <div class="stat-row"><span class="stat-label">LATENCY</span><span class="stat-value tech-font" id="stat-net-ping">--ms</span></div>
            <div class="stat-row"><span class="stat-label">NETWORK</span><span id="stat-net-status" class="stat-value" style="color: var(--gold-primary);">AWAITING</span></div>
            
            <div class="panel-title">HARDWARE</div>
            <div class="stat-row"><span class="stat-label">OPTICS</span><span class="stat-value tech-font" id="vision-status" style="color: var(--text-secondary);">OFFLINE</span></div>
            <div class="stat-row"><span class="stat-label">AUDIO</span><span class="stat-value tech-font" id="audio-status" style="color: var(--success);">ACTIVE</span></div>
            
            <button class="btn-terminate" onclick="terminateSystem()">TERMINATE SYSTEM</button>
        </aside>

        <div class="main-content">
            <div class="top-bar">
                <button class="btn-action" id="toggleCamBtn" onclick="toggleLocalCamera()">ACTIVATE WEBCAM</button>
                <button class="btn-action btn-mini-mode" onclick="enterMiniMode()">SCREEN SYNC (MINI MODE)</button>
            </div>
            
            <div class="visual-stage">
                <div class="holo-scene" id="hologramCore">
                    <div class="holo-sphere">
                        <div class="orbit orbit-1"></div>
                        <div class="orbit orbit-2"></div>
                        <div class="orbit orbit-3"></div>
                    </div>
                </div>
                <img id="liveFeed" src="" alt="Live Feed">
                <div id="subtitleOverlay" class="subtitle-overlay"></div>
            </div>

            <div class="terminal-bar">
                <div class="chat-log" id="chatBox">
                    <div style="color: var(--text-secondary); font-size: 0.75rem; letter-spacing: 1px;">SYSTEM: Secure connection established.</div>
                </div>
                <div class="cmd-section">
                    <input type="text" id="textInput" class="cmd-input" placeholder="Enter command..." autocomplete="off" onkeypress="if(event.key === 'Enter') handleCommand()">
                    <button class="btn-action" onclick="handleCommand()">SUBMIT COMMAND</button>
                </div>
            </div>
        </div>
    </div>

    <div id="mini-dashboard">
        <div class="mini-card">
            <div class="mini-title">SCREEN SYNC ACTIVE</div>
            
            <div class="wave-container" id="mic-wave">
                <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
            </div>
            
            <div class="mini-transcript" id="mini-transcript">System tracking screen content...</div>
            
            <div class="mini-controls">
                <button class="btn-mini-mute" id="btn-mute" onclick="toggleMute()">MUTE MIC</button>
                <button class="btn-mini-back" onclick="exitMiniMode()">FULLSCREEN</button>
            </div>
        </div>
    </div>

    <script>
        let isCameraRunning = false;
        let telemetryInterval = null;

        window.onload = () => {
            const savedUser = localStorage.getItem('nexusUser');
            if (savedUser) document.getElementById('usernameInput').value = savedUser;
        };

        function initSystem() {
            let user = document.getElementById('usernameInput').value.trim();
            if (!user) user = "Admin";
            localStorage.setItem('nexusUser', user);
            
            document.getElementById('auth-screen').style.opacity = '0';
            setTimeout(() => {
                document.getElementById('auth-screen').style.display = 'none';
                document.getElementById('main-dashboard').style.display = 'flex';
                setTimeout(() => { 
                    document.getElementById('main-dashboard').style.opacity = '1'; 
                    startTelemetry(); 
                }, 100);
            }, 1000);
        }

        // NATIVE WINDOW SIZING OVERRIDES
        function enterMiniMode() {
            document.getElementById('main-dashboard').style.display = 'none';
            document.getElementById('mini-dashboard').style.display = 'flex';
            if (window.pywebview) window.pywebview.api.enter_mini_mode();
            else fetch('/set_camera?src=screen');
        }

        function exitMiniMode() {
            document.getElementById('mini-dashboard').style.display = 'none';
            document.getElementById('main-dashboard').style.display = 'flex';
            if (window.pywebview) window.pywebview.api.exit_mini_mode();
            else {
                let target = isCameraRunning ? "0" : "off";
                fetch(`/set_camera?src=${target}`);
            }
        }

        function toggleMute() {
            if (window.pywebview) window.pywebview.api.toggle_mute();
            else fetch('/toggle_audio', { method: 'POST' });
        }

        function terminateSystem() {
            if (telemetryInterval) clearInterval(telemetryInterval);
            document.body.innerHTML = `
                <div style="display:flex; height:100vh; width:100vw; background:#020202; align-items:center; justify-content:center; flex-direction:column; font-family:'Montserrat', sans-serif;">
                    <div style="color:var(--danger); font-size:2rem; letter-spacing:10px; font-weight:300; text-transform:uppercase;">[ SYSTEM TERMINATED ]</div>
                </div>
            `;
            if (window.pywebview) window.pywebview.api.terminate();
            else fetch('/terminate', { method: 'POST' }).catch(() => {});
        }

        function startTelemetry() {
            telemetryInterval = setInterval(async () => {
                try {
                    let res = await fetch('/stats');
                    let data = await res.json();
                    document.getElementById('stat-cpu').innerText = data.cpu + '%';
                    document.getElementById('stat-ram').innerText = data.ram_used + 'GB';
                    document.getElementById('stat-batt').innerText = data.battery;
                    document.getElementById('stat-net-ping').innerText = data.ping;
                    
                    const netStatusElem = document.getElementById('stat-net-status');
                    netStatusElem.innerText = data.net_status;
                    netStatusElem.style.color = data.net_status === "EXCELLENT" ? "var(--success)" : (data.net_status === "GOOD" ? "var(--gold-primary)" : "var(--danger)");

                    const audioStatus = document.getElementById('audio-status');
                    const miniMuteBtn = document.getElementById('btn-mute');
                    const micWave = document.getElementById('mic-wave');
                    
                    if(data.audio_muted) {
                        audioStatus.innerText = "MUTED"; audioStatus.style.color = "var(--danger)";
                        miniMuteBtn.innerText = "UNMUTE MIC"; miniMuteBtn.classList.add('muted');
                        micWave.classList.add('muted');
                    } else {
                        audioStatus.innerText = "ACTIVE"; audioStatus.style.color = "var(--success)";
                        miniMuteBtn.innerText = "MUTE MIC"; miniMuteBtn.classList.remove('muted');
                        micWave.classList.remove('muted');
                    }
                } catch(e) { }

                try {
                    let chatRes = await fetch('/get_chat');
                    let chatData = await chatRes.json();
                    if(chatData.text) appendAiText(chatData.text);
                } catch(e) {}
            }, 1000); 
        }

        function handleCommand() {
            let inputField = document.getElementById('textInput');
            let val = inputField.value.trim();
            if(!val) return;
            logMsg(`<div class="user-msg">USER: ${val}</div>`);
            inputField.value = '';
            if(window.pywebview) window.pywebview.api.send_message(val);
            else fetch('/send_message', { method: 'POST', body: val });
        }

        let aiTextTimeout = null;
        let subtitleTimeout = null;
        function appendAiText(text) {
            const cb = document.getElementById('chatBox');
            let currentLine = cb.lastElementChild;
            if(!currentLine || !currentLine.classList.contains('active-ai-line')) {
                currentLine = document.createElement('div');
                currentLine.classList.add('active-ai-line', 'ai-msg');
                currentLine.innerHTML = `<strong>NEXUS:</strong> <span class="ai-text"></span>`;
                cb.appendChild(currentLine);
                document.getElementById('subtitleOverlay').innerText = ''; 
                document.getElementById('mini-transcript').innerText = ''; 
            }
            currentLine.querySelector('.ai-text').innerText += text;
            cb.scrollTop = cb.scrollHeight;
            
            document.getElementById('subtitleOverlay').innerText += text;
            document.getElementById('mini-transcript').innerText += text;
            
            clearTimeout(aiTextTimeout);
            clearTimeout(subtitleTimeout);
            aiTextTimeout = setTimeout(() => { if(cb.lastElementChild) cb.lastElementChild.classList.remove('active-ai-line'); }, 2500);
            subtitleTimeout = setTimeout(() => { 
                document.getElementById('subtitleOverlay').innerText = ''; 
                document.getElementById('mini-transcript').innerText = 'System tracking screen content...'; 
            }, 4500);
        }

        function logMsg(html) {
            const cb = document.getElementById('chatBox');
            cb.innerHTML += html;
            cb.scrollTop = cb.scrollHeight;
        }

        function toggleLocalCamera() {
            const btn = document.getElementById('toggleCamBtn');
            const feedImg = document.getElementById('liveFeed');
            const holoCore = document.getElementById('hologramCore');
            const statusLabel = document.getElementById('vision-status');

            isCameraRunning = !isCameraRunning;

            if(isCameraRunning) {
                btn.innerText = "DEACTIVATE WEBCAM";
                btn.style.color = "var(--danger)"; btn.style.borderColor = "var(--danger)";
                holoCore.style.display = 'none';
                
                if (window.pywebview) window.pywebview.api.set_camera("0");
                else fetch('/set_camera?src=0');
                
                setTimeout(() => {
                    feedImg.src = '/video_feed?t=' + new Date().getTime(); 
                    feedImg.style.display = 'block';
                    statusLabel.innerText = 'ACTIVE'; statusLabel.style.color = 'var(--success)';
                    logMsg(`<div class="sys-msg">SYSTEM: Hardware camera linked.</div>`);
                }, 1500);
            } else {
                btn.innerText = "ACTIVATE WEBCAM";
                btn.style.color = "var(--gold-primary)"; btn.style.borderColor = "var(--gold-primary)";
                feedImg.style.display = 'none'; feedImg.src = '';
                holoCore.style.display = 'flex';
                
                if (window.pywebview) window.pywebview.api.set_camera("off");
                else fetch('/set_camera?src=off');
                
                statusLabel.innerText = 'OFFLINE'; statusLabel.style.color = 'var(--text-secondary)';
                logMsg(`<div class="sys-msg">SYSTEM: Hardware camera unlinked.</div>`);
            }
        }
    </script>
</body>
</html>
"""

# =====================================================================
# LOCAL WEB SERVER & API HANDLER
# =====================================================================
class WebDashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode('utf-8'))
        elif parsed_path.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '-1')
            self.send_header('Connection', 'close')
            self.end_headers()
            try:
                while True:
                    global LATEST_JPEG
                    if LATEST_JPEG:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', str(len(LATEST_JPEG)))
                        self.end_headers()
                        self.wfile.write(LATEST_JPEG)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.033) 
            except Exception:
                pass 
        elif parsed_path.path == '/stats':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(get_hardware_stats().encode('utf-8'))
        elif parsed_path.path == '/get_chat':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            if AI_CHAT_QUEUE:
                msg = "".join(AI_CHAT_QUEUE)
                AI_CHAT_QUEUE.clear()
                self.wfile.write(json.dumps({"text": msg}).encode('utf-8'))
            else:
                self.wfile.write(json.dumps({"text": ""}).encode('utf-8'))
        elif parsed_path.path == '/set_camera':
            global TARGET_CAM_SOURCE
            qs = urllib.parse.parse_qs(parsed_path.query)
            if 'src' in qs:
                TARGET_CAM_SOURCE = qs['src'][0]
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/send_message':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            if post_data.strip():
                TEXT_PROMPT_QUEUE.append(post_data.strip())
            self.send_response(200)
            self.end_headers()
        elif parsed_path.path == '/toggle_audio':
            global MIC_MUTED
            MIC_MUTED = not MIC_MUTED
            self.send_response(200)
            self.end_headers()
        elif parsed_path.path == '/terminate':
            self.send_response(200)
            self.end_headers()
            print("\n[SYSTEM] Termination command executed. Graceful exit.")
            os._exit(0)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass 

def start_web_server():
    PORT = 8080
    class ReusableTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
    httpd = ReusableTCPServer(("127.0.0.1", PORT), WebDashboardHandler)
    httpd.serve_forever()

# =====================================================================
# CORE AI AUDIO/VIDEO/TEXT LOOP
# =====================================================================
class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None

    async def send_text_realtime(self):
        while True:
            if TEXT_PROMPT_QUEUE:
                msg = TEXT_PROMPT_QUEUE.pop(0)
                if self.session is not None and self.out_queue is not None:
                    await self.out_queue.put({"mime_type": "text/plain", "data": msg})
            await asyncio.sleep(0.1)

    async def get_frames(self):
        global LATEST_JPEG, TARGET_CAM_SOURCE
        last_ai_send_time = 0
        AI_SEND_INTERVAL = 1.0 

        while True:
            if TARGET_CAM_SOURCE != "off" and LATEST_JPEG:
                current_time = time.time()
                if current_time - last_ai_send_time >= AI_SEND_INTERVAL:
                    if self.out_queue is not None:
                        await self.out_queue.put({"mime_type": "image/jpeg", "data": LATEST_JPEG})
                    last_ai_send_time = current_time
            await asyncio.sleep(0.05)

    async def send_realtime(self):
        while True:
            if self.out_queue is not None:
                msg = await self.out_queue.get()
                if self.session is not None:
                    mime = msg.get("mime_type", "")
                    if mime == "text/plain":
                        text_data = msg.get("data")
                        forceful_prompt = f"The user just typed this message on their keyboard: '{text_data}'. Please reply to this message immediately out loud."
                        await self.session.send_realtime_input(text=forceful_prompt)
                    else:
                        if mime == "image/jpeg" and TARGET_CAM_SOURCE == "off":
                            continue
                        blob = types.Blob(mime_type=mime, data=msg.get("data"))
                        if "audio" in mime: await self.session.send_realtime_input(audio=blob)
                        elif "image" in mime: await self.session.send_realtime_input(video=blob)

    async def listen_audio(self):
        global MIC_MUTED
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
            input=True, input_device_index=mic_info["index"], frames_per_buffer=CHUNK_SIZE,
        )
        kwargs = {"exception_on_overflow": False} if __debug__ else {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            if not MIC_MUTED:
                # -------------------------------------------------------------
                # INSTANT INTERRUPTION LOGIC (Voice Activity Detection)
                # -------------------------------------------------------------
                try:
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    # Calculate volume (Root Mean Square) safely
                    rms = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
                    
                    if rms > 1000: # If you start speaking, volume spikes
                        # Flush the AI's playback queue immediately to cut it off
                        while not self.audio_in_queue.empty():
                            try:
                                self.audio_in_queue.get_nowait()
                            except:
                                break
                except Exception:
                    pass

                if self.out_queue is not None:
                    await self.out_queue.put({"data": data, "mime_type": f"audio/pcm;rate={SEND_SAMPLE_RATE}"})

    async def receive_audio(self):
        while True:
            if self.session is not None:
                turn = self.session.receive()
                async for response in turn:
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        print(text, end="", flush=True)
                        AI_CHAT_QUEUE.append(text)
                while not self.audio_in_queue.empty(): self.audio_in_queue.get_nowait()

    async def play_audio(self):
        # The AI now plays audio unconditionally, regardless of the user's mic state.
        stream = await asyncio.to_thread(pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE, output=True)
        while True:
            if self.audio_in_queue is not None:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                tg.create_task(self.send_text_realtime()) 
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.get_frames())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                while True: await asyncio.sleep(3600)
        except asyncio.CancelledError: pass
        except ExceptionGroup as EG:
            if self.audio_stream is not None: self.audio_stream.close()
            traceback.print_exception(EG)

# =====================================================================
# MAIN EXECUTION SEQUENCE
# =====================================================================
if __name__ == "__main__":
    print("[NEXUS] Booting Local Background Services...")
    
    threading.Thread(target=measure_latency, daemon=True).start()
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=start_web_server, daemon=True).start()
    
    def start_ai_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        main_loop = AudioLoop()
        loop.run_until_complete(main_loop.run())

    threading.Thread(target=start_ai_loop, daemon=True).start()
    
    print("[NEXUS] Launching Standalone Desktop Interface...")
    
    api = WebviewAPI()
    webview_window = webview.create_window(
        title='NEXUS COMMAND CENTER', 
        url='http://127.0.0.1:8080',
        js_api=api,
        width=1280, 
        height=800,
        background_color='#020202',
        confirm_close=True
    )
    
    # This completely overrides the default web browser and opens a native app window
    webview.start()