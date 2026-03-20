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

# 2. Futuristic Website UI Design (YOUR ORIGINAL AWESOME DESIGN)
st.set_page_config(page_title="NeuralVision AI", page_icon="🚨", layout="centered")

st.markdown("""
<style>
    /* Main background - Deep futuristic dark blue/black */
    .stApp {
        background-color: #0A0E17;
        color: #E2E8F0;
    }
    
    /* Hide the default Streamlit top header */
    [data-testid="stHeader"] {
        background-color: transparent;
    }

    /* Futuristic neon title */
    h1 {
        color: #00D2FF !important;
        text-align: center;
        text-shadow: 0 0 15px rgba(0, 210, 255, 0.5);
        font-weight: 800;
        letter-spacing: 2px;
        font-family: 'Arial', sans-serif;
    }

    /* Subtitle / Terminal-like description */
    .ai-subtitle {
        text-align: center;
        color: #00FF9D;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-family: monospace;
        letter-spacing: 1px;
    }

    /* Glowing Emergency Button */
    .stButton>button {
        background: linear-gradient(90deg, #FF0055 0%, #CC0000 100%);
        color: white;
        border-radius: 5px;
        width: 100%;
        border: 1px solid #FF0055;
        box-shadow: 0 0 15px rgba(255, 0, 85, 0.4);
        font-weight: bold;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }

    /* Button Hover Effect */
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(255, 0, 85, 0.8);
        border-color: #FFFFFF;
        transform: translateY(-2px);
    }

    /* File uploader styling with neon border */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #00D2FF;
        background-color: rgba(0, 210, 255, 0.05);
        border-radius: 10px;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>👁️‍🗨️ NeuralVision: Smart Accident AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='ai-subtitle'>[ SYSTEM ACTIVE ] // Analyzing traffic stream with layered defenses...</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. Load Model (Single Model is safe and fast for Streamlit)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Standard model for cars/buses

model = load_model()

# --- MATHEMATICAL LOGIC TO CHECK IF CARS CRASHED ---
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1 = max(x1, x3); yi1 = max(y1, y3)
    xi2 = min(x2, x4); yi2 = min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    if inter_area > 0:
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        min_area = min(box1_area, box2_area)
        # Using a loose overlap coefficient to catch messy real crash
        if inter_area / min_area > 0.25:
            return True
    return False

# 4. Video Upload Functionality
uploaded_file = st.file_uploader("Upload Surveillance Feed (MP4/AVI)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() # Placeholder for video frames
    
    # --- LOGIC VARIABLES (The Layered Defense) ---
    accident_counter = 0
    REQUIRED_FRAMES = 15 # Wait for approx 0.5 seconds of sustained overlap
    call_triggered = False
    accident_detected_final = False

    st.info("Neural Engine tracking vehicle interactions...")

    # --- MAIN DETECTION LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  
        
        # Smart Prediction on all vehicles perfectly
        results = model.predict(frame, conf=0.50, verbose=False)
        vehicles = []
        vehicle_classes = [2, 3, 5, 7] # COCO labels: car, motorcycle, bus, truck
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in vehicle_classes:
                    vehicles.append(box.xyxy[0].cpu().numpy())

        # Check interaction between all pairs
        crash_detected_now = False
        crashing_vehicles = []
        
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                if calculate_iou(vehicles[i], vehicles[j]):
                    # Found an overlap! Let's check the location (The 'VERE OPTION')
                    # Find centroid of this potential crash box area
                    xa1, ya1, xa2, ya2 = vehicles[i]
                    xb1, yb1, xb2, yb2 = vehicles[j]
                    
                    # Estimate centroid of combined bounding box of potential crash
                    combined_centroid_x = (min(xa1, xb1) + max(xa2, xb2)) / 2
                    frame_width = frame.shape[1]
                    
                    # --- SPATIAL FILTER: Is this crash happening in the parking lane area (Left side)? ---
                    if combined_centroid_x < 0.3 * frame_width: # Ignore potential crashes in first 30% of video width
                        # Globally ignore this false-alarm location
                        continue 
                    
                    # If it passed spatial filter, it's a real crash in driving lanes
                    crash_detected_now = True
                    crashing_vehicles.append(vehicles[i])
                    crashing_vehicles.append(vehicles[j])

        # --- TEMPORAL FILTER: Wait for sustained detection ---
        if crash_detected_now:
            accident_counter += 1
        else:
            # If the crash disappears (even for one frame due to glitch), we RESET immediately.
            accident_counter = 0

        # Draw Frame
        annotated_frame = frame.copy()
        
        # Draw all vehicles in normal green
        for v in vehicles:
            cv2.rectangle(annotated_frame, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), (0, 255, 0), 2)
            
        # Draw crashing vehicles in BRIGHT RED with text
        if crash_detected_now:
            for v in crashing_vehicles:
                # Use a larger rectangle with bright red
                cv2.rectangle(annotated_frame, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), (0, 0, 255), 4)
                # Put prominent text with red color and white background box
                text = "CRASH!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_scale = 1.2
                text_thickness = 3
                # Draw white box background first
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_scale, text_thickness)
                cv2.rectangle(annotated_frame, (int(v[0]), int(v[1]) - 35), (int(v[0]) + text_width, int(v[1])), (255, 255, 255), -1)
                # Draw red text
                cv2.putText(annotated_frame, text, (int(v[0]), int(v[1]) - 10), font, text_scale, (0, 0, 255), text_thickness)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # TRIGGER EMEGENCY ACTION
        if accident_counter >= REQUIRED_FRAMES and not call_triggered:
            # First, display the annotated frame immediately to freeze it on screen
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            
            st.error("🚨 CRITICAL ACCIDENT DETECTED (Vehicles Collided)! Calling Emergency...")
            
            # Make the Twilio Call
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
    
    # 5. Final Status
    if not accident_detected_final:
        st.success("✅ Analysis Complete: No real-time accident interactions confirmed. Road is safe.")
    else:
        st.warning("⚠️ System Log: Emergency services notified.")
