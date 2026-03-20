import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
import cv2
import tempfile
import math

# 1. Twilio Credentials (Safely handling)
try:
    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    twilio_number = st.secrets["TWILIO_PHONE_NUMBER"]
    my_number = st.secrets["MY_PHONE_NUMBER"]
    client = Client(account_sid, auth_token)
except:
    st.warning("Twilio Secrets missing.")

# 2. UI Styling
st.set_page_config(page_title="NeuralVision Ultra", page_icon="🚨", layout="centered")
st.markdown("<h1 style='color: #00FF9D; text-align: center;'>👁️‍🗨️ NeuralVision: Deep Action Detection</h1>", unsafe_allow_html=True)

# 3. Load YOLOv8 Nano (Fastest for real-time)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') 

model = load_model()

uploaded_file = st.file_uploader("Upload Surveillance Feed", type=['mp4', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() 
    
    accident_counter = 0
    triggered = False

    st.info("AI Engine scanning for high-velocity interactions...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break  
            
        # --- THE FIX: Lower confidence to 0.20 to catch blurred accidents ---
        results = model.predict(frame, conf=0.20, verbose=False)
        vehicles = []
        
        for r in results:
            for box in r.boxes:
                # Car, Motorcycle, Bus, Truck, Person (even catch the victim)
                if int(box.cls[0]) in [0, 2, 3, 5, 7]: 
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf_score = float(box.conf[0])
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    vehicles.append({'box': [x1, y1, x2, y2], 'center': [cx, cy], 'w': (x2-x1), 'conf': conf_score})

        crash_now = False
        crash_boxes = []
        
        # Check every possible interaction
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                # Distance between vehicle centers
                dist = math.sqrt((v1['center'][0]-v2['center'][0])**2 + (v1['center'][1]-v2['center'][1])**2)
                
                # Loose distance threshold (If they are very close, it's a hit)
                threshold = (v1['w'] + v2['w']) / 2 * 0.95
                
                if dist < threshold:
                    crash_now = True
                    crash_boxes.extend([v1['box'], v2['box']])

        if crash_now:
            accident_counter += 1
        else:
            accident_counter = 0

        # Rendering
        annotated_frame = frame.copy()
        for v in vehicles:
            # Draw green boxes for all detected things
            cv2.rectangle(annotated_frame, (int(v['box'][0]), int(v['box'][1])), (int(v['box'][2]), int(v['box'][3])), (0, 255, 0), 1)

        if accident_counter >= 3 and not triggered:
            for b in crash_boxes:
                cv2.rectangle(annotated_frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 4)
                cv2.putText(annotated_frame, "CRASH!", (int(b[0]), int(b[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            st.error("🚨 ALERT: CRASH DETECTED BY MOTION PROXIMITY!")
            
            try:
                msg = '<Response><Say>Emergency! Crash detected.</Say></Response>'
                client.calls.create(twiml=msg, to=st.secrets["MY_PHONE_NUMBER"], from_=st.secrets["TWILIO_PHONE_NUMBER"])
                st.success("Authorities alerted successfully!")
                triggered = True
            except:
                st.error("Twilio Call failed.")
            break 

        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    
    cap.release()
