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
    st.warning("Twilio Secrets not fully configured. Calling features are disabled.")

# 2. Futuristic Website UI Design 
st.set_page_config(page_title="NeuralVision AI", page_icon="🚨", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0A0E17; color: #E2E8F0; }
    [data-testid="stHeader"] { background-color: transparent; }
    h1 { color: #00D2FF !important; text-align: center; text-shadow: 0 0 15px rgba(0, 210, 255, 0.5); font-weight: 800; letter-spacing: 2px; }
    .ai-subtitle { text-align: center; color: #00FF9D; font-size: 1.1rem; margin-bottom: 2rem; font-family: monospace; letter-spacing: 1px; }
    [data-testid="stFileUploadDropzone"] { border: 2px dashed #00D2FF; background-color: rgba(0, 210, 255, 0.05); border-radius: 10px; padding: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>👁️‍🗨️ NeuralVision: Smart Accident AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='ai-subtitle'>[ SYSTEM ACTIVE ] // Analyzing traffic stream with Strict IoU...</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. Load Model (Using yolov8n.pt as it perfectly identifies vehicles)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  

model = load_model()

# --- THE STRICT MATHEMATICAL LOGIC (Fixing the "Gap" problem) ---
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # Calculate overlap boundaries
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    
    # Calculate overlap area
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # If they touch, calculate how severe the crash is
    if inter_area > 0:
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        
        # Calculate standard Intersection over Union (IoU)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area
        
        # STRICT RULE: They must overlap significantly (>35%) to be a real crash.
        # This ignores the "invisible corners" of bounding boxes.
        if iou > 0.35: 
            return True
            
    return False

# 4. Video Upload 
uploaded_file = st.file_uploader("Upload Surveillance Feed (MP4/AVI)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() 
    
    accident_counter = 0
    REQUIRED_FRAMES = 15 # Requires 0.5 seconds of sustained overlap to confirm
    call_triggered = False
    accident_detected_final = False

    st.info("Neural Engine tracking vehicle interactions...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  
        
        # Detect vehicles
        results = model.predict(frame, conf=0.50, verbose=False)
        vehicles = []
        vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in vehicle_classes:
                    vehicles.append(box.xyxy[0].cpu().numpy())

        crash_detected_now = False
        crashing_vehicles = []
        
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                if calculate_iou(vehicles[i], vehicles[j]):
                    crash_detected_now = True
                    crashing_vehicles.append(vehicles[i])
                    crashing_vehicles.append(vehicles[j])

        # Temporal filter
        if crash_detected_now:
            accident_counter += 1
        else:
            accident_counter = 0

        # Draw Frame
        annotated_frame = frame.copy()
        
        # Draw normal vehicles
        for v in vehicles:
            cv2.rectangle(annotated_frame, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), (0, 255, 0), 2)
            
        # Draw crashing vehicles
        if crash_detected_now:
            for v in crashing_vehicles:
                cv2.rectangle(annotated_frame, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), (0, 0, 255), 4)
                cv2.putText(annotated_frame, "CRASH!", (int(v[0]), int(v[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Trigger Event
        if accident_counter >= REQUIRED_FRAMES and not call_triggered:
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            st.error("🚨 CRITICAL ACCIDENT DETECTED! Calling Emergency...")
            
            try:
                msg = '<Response><Say>Emergency alert! A severe car accident has been detected.</Say></Response>'
                call = client.calls.create(twiml=msg, to=my_number, from_=twilio_number)
                st.success("Call successfully sent!")
                call_triggered = True
                accident_detected_final = True
            except Exception as e:
                st.error("Call Failed.")
                
            break 
            
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)
    
    cap.release()
    
    if not accident_detected_final:
        st.success("✅ Analysis Complete: No crashes detected. Road is safe.")
    else:
        st.warning("⚠️ System Log: Emergency services notified.")
