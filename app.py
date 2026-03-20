import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
import cv2
import tempfile
import os

# 1. Twilio Credentials Setup
# Make sure these keys are added in your Streamlit Cloud Secrets!
try:
    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    twilio_number = st.secrets["TWILIO_PHONE_NUMBER"]
    my_number = st.secrets["MY_PHONE_NUMBER"]
    client = Client(account_sid, auth_token)
except Exception as e:
    st.warning("Twilio Secrets not fully configured. Call feature might not work.")

# 2. Website UI Design
st.set_page_config(page_title="AI Accident Alert", page_icon="🚨", layout="centered")

# --- HIGH-TECH AI DASHBOARD CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0A0E17; color: #E2E8F0; }
    [data-testid="stHeader"] { background-color: transparent; }
    h1 { color: #00D2FF !important; text-align: center; text-shadow: 0 0 15px rgba(0, 210, 255, 0.5); font-weight: 800; letter-spacing: 2px; }
    .ai-subtitle { text-align: center; color: #00FF9D; font-size: 1.1rem; margin-bottom: 2rem; font-family: monospace; letter-spacing: 1px; }
    [data-testid="stFileUploadDropzone"] { border: 2px dashed #00D2FF; background-color: rgba(0, 210, 255, 0.05); border-radius: 10px; padding: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>👁️‍🗨️ NeuralVision: Smart Accident_Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='ai-subtitle'>[ SYSTEM ACTIVE ] // Awaiting traffic surveillance feed...</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 4. Video Upload Functionality
uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() # Placeholder for video frames
    
    # --- LOGIC VARIABLES ---
    accident_counter = 0
    call_triggered = False
    REQUIRED_FRAMES = 5 # Requires ~1 second of continuous detection
    accident_detected_final = False

    st.info("Neural Engine analyzing video stream...")

    # --- THE MAIN DETECTION LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        # 1. AI PREDICTION (0.3 confidence so it catches fast cars easily)
        results = model.predict(frame, conf=0.6, verbose=False)
        accident_in_this_frame = False

        # 2. CHECK LABELS
        target_labels = ['Accident', 'severe', 'severe-accident', 'car-accident', 'car-crash']
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label in target_labels:
                    accident_in_this_frame = True
                    break

        # 3. FILTER (Wait for just 3 frames - super fast detection)
        if accident_in_this_frame:
            accident_counter += 1
        else:
            accident_counter = 0

        # 4. DRAW THE BOX (Prepare the frame with the box on it)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # 5. TRIGGER & FREEZE-FRAME LOGIC
        if accident_counter >= 3:
            # First, display the image with the box so it stays on screen
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            
            st.error("🚨 CRITICAL ACCIDENT DETECTED! Video Frozen. Initiating Call...")
            
            # Make the Twilio Call
            try:
                msg = '<Response><Say>Emergency alert! A severe car accident has been detected.</Say></Response>'
                call = client.calls.create(twiml=msg, to=my_number, from_=twilio_number)
                st.success(f"Call successfully sent! SID: {call.sid}")
                accident_final_status = True
            except Exception as e:
                st.error("Twilio Call Failed! Check keys.")
                
            # THE MAGIC WORD: Stop the video exactly here!
            break 
            
        # 6. If no accident, just keep playing the video normally
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)
    
    # 5. Final Status
    if not accident_detected_final:
        st.success("Analysis Complete: No accidents detected. Road is clear.")
    else:
        st.warning("System Alert: Emergency services have been notified.")
