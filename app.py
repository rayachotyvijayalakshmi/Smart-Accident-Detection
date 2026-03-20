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

# 2. Website UI Design
st.set_page_config(page_title="AI Accident Alert", page_icon="🚨", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0A0E17; color: #E2E8F0; }
    [data-testid="stHeader"] { background-color: transparent; }
    h1 { color: #00D2FF !important; text-align: center; text-shadow: 0 0 15px rgba(0, 210, 255, 0.5); font-weight: 800; letter-spacing: 2px; }
    .ai-subtitle { text-align: center; color: #00FF9D; font-size: 1.1rem; margin-bottom: 2rem; font-family: monospace; letter-spacing: 1px; }
    [data-testid="stFileUploadDropzone"] { border: 2px dashed #00D2FF; background-color: rgba(0, 210, 255, 0.05); border-radius: 10px; padding: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>👁️‍🗨️ NeuralVision: Smart Crash Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='ai-subtitle'>[ SYSTEM ACTIVE ] // Using Spatio-Temporal Overlap Logic...</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. Load STANDARD Model (Throwing away best.pt!)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # This is extremely fast and accurate for vehicles

model = load_model()

# --- MATHEMATICAL LOGIC TO CHECK IF CARS CRASHED ---
def check_crash(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # Calculate overlap (Intersection)
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    if inter_area > 0:
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        min_area = min(box1_area, box2_area)
        
        # If they overlap by more than 20% of the smaller vehicle's size, it's a crash!
        if inter_area / min_area > 0.20:
            return True
    return False

# 4. Video Upload
uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() 
    
    accident_counter = 0
    REQUIRED_FRAMES = 2  # Very fast response
    accident_detected_final = False

    st.info("Neural Engine tracking vehicle physics...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  
        
        # Detect ALL vehicles perfectly using standard YOLO
        results = model.predict(frame, conf=0.40, verbose=False)
        
        vehicles = []
        # COCO labels: 2=car, 3=motorcycle, 5=bus, 7=truck
        vehicle_classes = [2, 3, 5, 7] 
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in vehicle_classes:
                    vehicles.append(box.xyxy[0].cpu().numpy())

        # Check if any two vehicles are crashing into each other
        crash_detected_now = False
        crash_boxes = []
        
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                if check_crash(vehicles[i], vehicles[j]):
                    crash_detected_now = True
                    crash_boxes.append(vehicles[i])
                    crash_boxes.append(vehicles[j])

        if crash_detected_now:
            accident_counter += 1
        else:
            accident_counter = 0

        # Draw the frame
        annotated_frame = frame.copy()
        
        # Draw all vehicles in normal green
        for v in vehicles:
            cv2.rectangle(annotated_frame, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), (0, 255, 0), 2)
            
        # Draw crashing vehicles in BRIGHT RED with text
        if crash_detected_now:
            for v in crash_boxes:
                cv2.rectangle(annotated_frame, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), (0, 0, 255), 4)
                cv2.putText(annotated_frame, "CRASH!", (int(v[0]), int(v[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        if accident_counter >= REQUIRED_FRAMES:
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            st.error("🚨 REAL CRASH DETECTED (Vehicles Collided)! Calling Emergency...")
            
            try:
                msg = '<Response><Say>Emergency! A collision has been detected.</Say></Response>'
                call = client.calls.create(twiml=msg, to=my_number, from_=twilio_number)
                st.success("Call Sent Successfully!")
                accident_detected_final = True
            except Exception as e:
                st.error("Call Failed.")
                
            break 
            
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)
    
    cap.release()
