import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
import cv2
import tempfile
import numpy as np

# 1. Twilio Credentials Setup
try:
    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    twilio_number = st.secrets["TWILIO_PHONE_NUMBER"]
    my_number = st.secrets["MY_PHONE_NUMBER"]
    client = Client(account_sid, auth_token)
except Exception as e:
    st.warning("Twilio Secrets not fully configured.")

# 2. UI Design
st.set_page_config(page_title="NeuralVision AI", page_icon="🚨", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0A0E17; color: #E2E8F0; }
    [data-testid="stHeader"] { background-color: transparent; }
    h1 { color: #00D2FF !important; text-align: center; text-shadow: 0 0 15px rgba(0, 210, 255, 0.5); font-weight: 800; }
    .ai-subtitle { text-align: center; color: #00FF9D; font-size: 1.1rem; margin-bottom: 2rem; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>👁️‍🗨️ NeuralVision: Smart Crash Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='ai-subtitle'>[ SYSTEM ACTIVE ] // Motion-Gated AI Engine Running...</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  

model = load_model()

# --- IOU LOGIC ---
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    xi1 = max(x1, x3); yi1 = max(y1, y3)
    xi2 = min(x2, x4); yi2 = min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    if inter_area > 0:
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area
        
        # Checking for real overlap
        if 0.15 < iou < 0.85: 
            return True
    return False

# 4. Video Processing
uploaded_file = st.file_uploader("Upload Surveillance Feed (MP4/AVI)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() 
    
    # --- THE PRO FEATURE: Background Subtractor (Finds only MOVING things) ---
    backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
    
    accident_counter = 0
    REQUIRED_FRAMES = 3  
    accident_detected_final = False
    frame_count = 0

    st.info("Neural Engine syncing with Motion Physics...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  
            
        frame_count += 1
        
        # 1. Learn the background (Creates a black & white motion mask)
        fgMask = backSub.apply(frame)
        
        # Wait for 30 frames so the AI learns that the Van is "Parked" and ignores it
        if frame_count < 30:
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            continue

        # 2. YOLO Detection
        results = model.predict(frame, conf=0.40, verbose=False)
        active_vehicles = []
        vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # --- MOTION GATE LOGIC ---
                    # Check the motion mask strictly inside this vehicle's box
                    box_mask = fgMask[y1:y2, x1:x2]
                    moving_pixels = cv2.countNonZero(box_mask)
                    total_pixels = (x2 - x1) * (y2 - y1)
                    
                    # If the box has > 5% moving pixels, it's a moving vehicle!
                    # The parked van will have 0% moving pixels and will be DROPPED.
                    if total_pixels > 0 and (moving_pixels / total_pixels) > 0.05:
                        active_vehicles.append([x1, y1, x2, y2])

        # 3. Crash Detection (ONLY on actively moving vehicles)
        crash_detected_now = False
        crashing_vehicles = []
        
        for i in range(len(active_vehicles)):
            for j in range(i + 1, len(active_vehicles)):
                if calculate_iou(active_vehicles[i], active_vehicles[j]):
                    crash_detected_now = True
                    crashing_vehicles.append(active_vehicles[i])
                    crashing_vehicles.append(active_vehicles[j])

        # 4. Trigger Logic
        if crash_detected_now:
            accident_counter += 1
        else:
            accident_counter = 0

        annotated_frame = frame.copy()
        
        for v in active_vehicles:
            cv2.rectangle(annotated_frame, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), (0, 255, 0), 2)
            
        if crash_detected_now:
            for v in crashing_vehicles:
                cv2.rectangle(annotated_frame, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), (0, 0, 255), 4)
                cv2.putText(annotated_frame, "CRASH!", (int(v[0]), int(v[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        if accident_counter >= REQUIRED_FRAMES:
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            st.error("🚨 CRITICAL ACCIDENT DETECTED! Calling Emergency...")
            
            try:
                msg = '<Response><Say>Emergency alert! A collision has been detected.</Say></Response>'
                call = client.calls.create(twiml=msg, to=my_number, from_=twilio_number)
                st.success("Call successfully sent!")
                accident_detected_final = True
            except Exception as e:
                st.error("Call Failed.")
            break 
            
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)
    
    cap.release()
