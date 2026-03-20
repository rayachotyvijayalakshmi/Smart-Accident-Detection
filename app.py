import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
import cv2
import tempfile

# 1. Twilio Credentials Setup
try:
    account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
    auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
    twilio_number = st.secrets["TWILIO_PHONE_NUMBER"]
    my_number = st.secrets["MY_PHONE_NUMBER"]
    client = Client(account_sid, auth_token)
except Exception as e:
    st.warning("Twilio Secrets not fully configured. Call feature disabled.")

# 2. Website UI Design (YOUR ORIGINAL AWESOME DESIGN)
st.set_page_config(page_title="AI Accident Alert", page_icon="🚨", layout="centered")

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

st.markdown("<h1>👁️‍🗨️ NeuralVision: Smart Accident_Detection AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='ai-subtitle'>[ SYSTEM ACTIVE ] // Awaiting traffic surveillance feed...</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 4. Video Upload
uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() 
    
    # --- LOGIC VARIABLES ---
    accident_counter = 0
    REQUIRED_FRAMES = 5  # Wait for 5 continuous frames to confirm crash
    accident_detected_final = False

    st.info("Neural Engine analyzing video stream...")

    # --- MAIN LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        # 1. Predict (Strictness = 50%)
        results = model.predict(frame, conf=0.5, verbose=False)
        accident_in_this_frame = False

        # 2. Check Labels
        target_labels = ['Accident', 'severe', 'severe-accident', 'car-accident', 'car-crash']
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                if label in target_labels:
                    accident_in_this_frame = True
                    break

        # 3. Filter
        if accident_in_this_frame:
            accident_counter += 1
        else:
            accident_counter = 0  # Reset if it's a false alarm

        # 4. Draw Box
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # 5. Freeze & Call Logic
        if accident_counter >= REQUIRED_FRAMES:
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            st.error("🚨 CRITICAL ACCIDENT DETECTED! Video Frozen. Calling Emergency...")
            
            try:
                msg = '<Response><Say>Emergency alert! A severe car accident has been detected.</Say></Response>'
                call = client.calls.create(twiml=msg, to=my_number, from_=twilio_number)
                st.success(f"Call Sent! SID: {call.sid}")
                accident_detected_final = True
            except:
                st.error("Twilio Call Failed.")
                
            break # Stop video
            
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)
    
    cap.release()
    
    # 6. Final Status
    if not accident_detected_final:
        st.success("Analysis Complete: No accidents detected. Road is clear.")
    else:
        st.warning("System Alert: Emergency services notified.")
