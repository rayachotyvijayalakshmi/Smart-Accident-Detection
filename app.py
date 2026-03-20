import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
import cv2
import tempfile

# 1. Twilio Setup (Bulletproof)
client = None
try:
    if all(k in st.secrets for k in ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "MY_PHONE_NUMBER"]):
        client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
    else:
        st.sidebar.warning("Twilio Secrets missing. Call feature disabled.")
except Exception:
    st.sidebar.error("Twilio Config Error.")

# 2. UI Styling
st.set_page_config(page_title="NeuralVision Pro", page_icon="🚨", layout="centered")
st.markdown("<h1 style='color: #00D2FF; text-align: center;'>👁️‍🗨️ NeuralVision: Smart Accident AI</h1>", unsafe_allow_html=True)
st.markdown("---")

# 3. Load Model (Using standard yolov8n - reliable)
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
    
    accident_counter = 0
    triggered = False

    st.info("Neural Engine active. Analyzing frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break  
        
        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # RUN DETECTION
        results = model.predict(frame, conf=0.40, verbose=False)
        boxes = results[0].boxes
        
        vehicles = []
        for box in boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]: # Car, Motorcycle, Bus, Truck
                vehicles.append(box.xyxy[0].cpu().numpy())

        # CRASH LOGIC: Overlap check
        crash_detected = False
        if len(vehicles) >= 2:
            for i in range(len(vehicles)):
                for j in range(i + 1, len(vehicles)):
                    b1, b2 = vehicles[i], vehicles[j]
                    
                    # Calculate Intersection
                    x_left = max(b1[0], b2[0])
                    y_top = max(b1[1], b2[1])
                    x_right = min(b1[2], b2[2])
                    y_bottom = min(b1[3], b2[3])

                    if x_right > x_left and y_bottom > y_top:
                        # They are overlapping!
                        crash_detected = True
                        break

        if crash_detected:
            accident_counter += 1
        else:
            accident_counter = 0

        # DRAWING
        annotated_frame = results[0].plot()
        
        if accident_counter >= 5 and not triggered:
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            st.error("🚨 ACCIDENT DETECTED! Video Frozen.")
            
            if client:
                try:
                    client.calls.create(
                        twiml='<Response><Say>Emergency alert! Accident detected.</Say></Response>',
                        to=st.secrets["MY_PHONE_NUMBER"],
                        from_=st.secrets["TWILIO_PHONE_NUMBER"]
                    )
                    st.success("Emergency Call Triggered!")
                except:
                    st.error("Twilio Call failed.")
            triggered = True
            break 

        stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    
    cap.release()
