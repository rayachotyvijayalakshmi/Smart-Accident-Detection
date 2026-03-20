import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
import cv2
import tempfile
import math

# 1. Twilio Setup
try:
    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    twilio_number = st.secrets["TWILIO_PHONE_NUMBER"]
    my_number = st.secrets["MY_PHONE_NUMBER"]
    client = Client(account_sid, auth_token)
except:
    st.warning("Twilio Secrets mismatch or missing.")

# 2. UI Styling (Your Awesome Design)
st.set_page_config(page_title="NeuralVision Pro", page_icon="🚨", layout="centered")
st.markdown("<h1 style='color: #00D2FF; text-align: center;'>👁️‍🗨️ NeuralVision: Smart Accident AI</h1>", unsafe_allow_html=True)
st.markdown("---")

# 3. Load YOLOv8 Standard (Best for vehicle tracking)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') 

model = load_model()

uploaded_file = st.file_uploader("Upload Video Feed", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() 
    
    frame_count = 0
    accident_counter = 0
    triggered = False

    st.info("Neural Engine Active. Analyzing...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break  
            
        frame_count += 1
        height, width, _ = frame.shape

        # Skip only the very first 20 frames to let the video load
        if frame_count < 20:
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            continue

        # RUN DETECTION
        results = model.predict(frame, conf=0.45, verbose=False)
        vehicles = []
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in [2, 3, 5, 7]: # Car, Motorcycle, Bus, Truck
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # SPATIAL FILTER: Ignore the parked Van (Left 40% of screen)
                    if cx < (width * 0.40):
                        continue
                    
                    vehicles.append({'box': [x1, y1, x2, y2], 'center': [cx, cy], 'w': (x2-x1)})

        # COLLISION LOGIC (Distance-Based)
        crash_now = False
        target_boxes = []
        
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                # Distance formula
                dist = math.sqrt((v1['center'][0]-v2['center'][0])**2 + (v1['center'][1]-v2['center'][1])**2)
                
                # If distance is very small relative to vehicle width, it's a crash
                if dist < (v1['w'] + v2['w']) / 2 * 0.90:
                    crash_now = True
                    target_boxes.extend([v1['box'], v2['box']])

        if crash_now:
            accident_counter += 1
        else:
            accident_counter = 0

        # DRAWING & TRIGGER
        annotated_frame = frame.copy()
        
        # Draw normal vehicles in green
        for v in vehicles:
            cv2.rectangle(annotated_frame, (int(v['box'][0]), int(v['box'][1])), (int(v['box'][2]), int(v['box'][3])), (0, 255, 0), 2)

        # Confirm accident after 5 frames of continuous collision
        if accident_counter >= 5 and not triggered:
            for b in target_boxes:
                cv2.rectangle(annotated_frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 4)
                cv2.putText(annotated_frame, "CRASH!", (int(b[0]), int(b[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            st.error("🚨 EMERGENCY: CRITICAL ACCIDENT DETECTED! INITIATING CALL...")
            
            try:
                msg = '<Response><Say>Emergency alert! A car accident has been detected.</Say></Response>'
                client.calls.create(twiml=msg, to=st.secrets["MY_PHONE_NUMBER"], from_=st.secrets["TWILIO_PHONE_NUMBER"])
                st.success("Twilio Call Sent Successfully!")
                triggered = True
            except:
                st.error("Call trigger failed. Check Twilio Secrets.")
            break # STOP VIDEO ON CRASH

        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    
    cap.release()
    if not triggered:
        st.success("✅ Analysis Complete: No accidents detected in the driving lanes.")
