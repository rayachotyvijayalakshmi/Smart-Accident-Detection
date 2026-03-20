import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
import cv2
import tempfile
import torch # We need this for intersection calculations

# 1. Twilio Credentials Setup (Using Streamlit Secrets)
try:
    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    twilio_number = st.secrets["TWILIO_PHONE_NUMBER"]
    my_number = st.secrets["MY_PHONE_NUMBER"]
    client = Client(account_sid, auth_token)
except Exception as e:
    st.warning("Twilio Secrets not fully configured. The calling feature may be disabled.")

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

    /* File uploader styling with neon border */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #00D2FF;
        background-color: rgba(0, 210, 255, 0.05);
        border-radius: 10px;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>👁️‍🗨️ NeuralVision: Smart Accident_Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='ai-subtitle'>[ SYSTEM ACTIVE ] // Analyzing traffic for anomalies...</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. Load Two Models (The Pro Move)
@st.cache_resource
def load_models():
    # Model 1: Your custom accident model (Detects visual pattern)
    model_accident = YOLO('best.pt') 
    # Model 2: Standard YOLOv8 model (Detects cars, bikes, etc.)
    model_general = YOLO('yolov8n.pt') 
    return model_accident, model_general

model_accident, model_general = load_models()

# 4. Video Upload Functionality
uploaded_file = st.file_uploader("Upload Surveillance Feed (MP4/AVI)", type=['mp4', 'avi', 'mov'])

# --- Function to check if two boxes are extremely near each other (Pro Logic) ---
def is_near(box1, box2, threshold=10):
    # Calculate Euclidean distance between centers or a simple overlap check
    # For speed and simplicity in Streamlit, a basic overlap with a small padding works best.
    x1, y1, x2, y2 = box1
    xa, ya, xb, yb = box2
    
    # Check if they are virtually touching or overlapping
    # Add a tiny padding to the boxes to make them slightly larger for the near-miss detection
    if x1 - threshold < xb + threshold and x2 + threshold > xa - threshold and \
       y1 - threshold < yb + threshold and y2 + threshold > ya - threshold:
        return True
    return False

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() # Placeholder for video frames
    
    # --- LOGIC VARIABLES ---
    accident_final_status = False
    
    st.info("Proton Engine initializing Spatio-Temporal Interaction Logic...")

    # --- THE MAIN DETECTION LOOP (With interaction check) ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        # A. Run General Model (Find all vehicles and people)
        results_general = model_general.predict(frame, conf=0.50, verbose=False)
        vehicle_boxes = [] # List to store boxes of individual vehicles [x1, y1, x2, y2]
        
        # Vehicle labels to focus on (Car, Bike, Bus, Truck) from standard YOLO
        vehicle_labels = [2, 3, 5, 7] # COCO labels for standard model
        for r in results_general:
            for box in r.boxes:
                label_id = int(box.cls[0])
                if label_id in vehicle_labels:
                    # Save the coordinates as integers for easier math
                    vehicle_boxes.append(box.xyxy[0].cpu().numpy().astype(int))

        # B. Run Custom Accident Model (Find potential accident area)
        # Using higher confidence here now because we filter False Positives later
        results_accident = model_accident.predict(frame, conf=0.60, verbose=False)
        accident_detected_final_in_loop = False
        final_annotated_frame = None

        target_labels = ['Accident', 'severe', 'severe-accident', 'car-accident', 'car-crash']
        for r in results_accident:
            for box in r.boxes:
                label = model_accident.names[int(box.cls[0])]
                if label in target_labels:
                    
                    # C. The 'VERE OPTION' Logic (The Interaction Check)
                    accident_box = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # COUNT how many individual vehicles are extremely close to this 'Accident' box
                    near_vehicle_count = 0
                    for v_box in vehicle_boxes:
                        if is_near(accident_box, v_box):
                            near_vehicle_count += 1
                    
                    # Condition: Only confirm if this accident is surrounded by interacting vehicles
                    if near_vehicle_count >= 2: # At least 2 interacting parties
                        accident_detected_final_in_loop = True
                        final_annotated_frame = results_accident[0].plot() # Prepare the frame to freeze
                        break 
        
        # D. The Final Decision & Trigger
        if accident_detected_final_in_loop:
            # Update frame to the frozen annotated frame
            final_annotated_frame = cv2.cvtColor(final_annotated_frame, cv2.COLOR_BGR2RGB)
            stframe.image(final_annotated_frame, channels="RGB", use_container_width=True)
            
            st.error("🚨 CRITICAL ACCIDENT CONFIRMED! Two vehicles interaction verified. Freezing Feed...")
            
            # Make the Twilio Call
            try:
                msg = '<Response><Say>Emergency alert! A severe car accident has been detected and confirmed.</Say></Response>'
                call = client.calls.create(twiml=msg, to=my_number, from_=twilio_number)
                st.success(f"Call successfully sent! SID: {call.sid}")
                accident_final_status = True
            except Exception as e:
                st.error(f"Call Failed: {e}")
                accident_final_status = True # Stop retrying on fail

            break # THE MAGIC WORD: Stop the video exactly here.

        # E. Update the Dashboard Display (Play Normally)
        results_normal = results_accident[0].plot() # Draw your original boxes
        results_normal = cv2.cvtColor(results_normal, cv2.COLOR_BGR2RGB)
        stframe.image(results_normal, channels="RGB", use_container_width=True)

    cap.release()
    
    # 5. Final Report
    if not accident_final_status:
        st.success("✅ Analysis Complete: No real-time accident interactions confirmed. Road is safe.")
    else:
        st.warning("⚠️ System Log: Emergency services notified of a confirmed interaction event.")
