import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
import cv2
import tempfile
import math

# 1. Twilio Credentials
try:
    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    twilio_number = st.secrets["TWILIO_PHONE_NUMBER"]
    my_number = st.secrets["MY_PHONE_NUMBER"]
    client = Client(account_sid, auth_token)
except Exception as e:
    st.warning("Twilio Secrets not configured.")

# 2. UI Design
st.set_page_config(page_title="NeuralVision AI", page_icon="🚨", layout="centered")
st.markdown("<h1 style='color: #00D2FF; text-align: center;'>👁️‍🗨️ NeuralVision: Pro Accident AI</h1>", unsafe_allow_html=True)
st.markdown("---")

# 3. Load Standard Model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') 

model = load_model()

# 4. Video Upload
uploaded_file = st.file_uploader("Upload Surveillance Feed", type=['mp4', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() 
    
    # --- PRO LOGIC VARIABLES ---
    frame_count = 0
    accident_counter = 0
    accident_detected_final = False

    st.info("System initializing... Waiting for stabilization (6 seconds)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break  
            
        frame_count += 1
        height, width, _ = frame.shape

        # --- STEP 1: SKIP STARTING JITTER (Crucial Fix) ---
        if frame_count < 150: # Skip first ~6 seconds
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            continue

        # --- STEP 2: RUN DETECTION ---
        results = model.predict(frame, conf=0.45, verbose=False)
        vehicles = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in [2, 3, 5, 7]: # Car, Motorcycle, Bus, Truck
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 # Center Point
                    
                    # --- STEP 3: SPATIAL FILTER (Ignore the Van Area) ---
                    if cx < (width * 0.40): # Ignore left 40% of the screen
                        continue
                    
                    vehicles.append({'box': [x1, y1, x2, y2], 'center': [cx, cy], 'width': (x2-x1)})

        # --- STEP 4: DISTANCE-BASED CRASH CHECK ---
        crash_now = False
        crash_pairs = []
        
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                
                # Calculate distance between centers
                dist = math.sqrt((v1['center'][0] - v2['center'][0])**2 + (v1['center'][1] - v2['center'][1])**2)
                
                # If distance is less than 85% of car's width, they have crashed!
                threshold = (v1['width'] + v2['width']) / 2 * 0.85
                
                if dist < threshold:
                    crash_now = True
                    crash_pairs.append(v1['box'])
                    crash_pairs.append(v2['box'])

        # --- STEP 5: TEMPORAL FILTER (Wait for 8 frames) ---
        if crash_now:
            accident_counter += 1
        else:
            accident_counter = 0

        # --- DRAWING ---
        annotated_frame = frame.copy()
        for v in vehicles:
            cv2.rectangle(annotated_frame, (int(v['box'][0]), int(v['box'][1])), (int(v['box'][2]), int(v['box'][3])), (0, 255, 0), 2)
            
        if accident_counter >= 8:
            for b in crash_pairs:
                cv2.rectangle(annotated_frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 4)
                cv2.putText(annotated_frame, "ACCIDENT!", (int(b[0]), int(b[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            st.error("🚨 EMERGENCY: ACCIDENT DETECTED AT CENTER LANES!")
            
            try:
                msg = '<Response><Say>Emergency! Accident detected.</Say></Response>'
                call = client.calls.create(twiml=msg, to=my_number, from_=twilio_number)
                st.success("Alert sent to authorities!")
                accident_detected_final = True
            except:
                st.error("Call trigger failed.")
            break 

        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    
    cap.release()
