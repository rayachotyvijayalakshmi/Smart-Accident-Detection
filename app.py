import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
import cv2
import tempfile

# 1. Twilio Credentials Setup
# Replace your original secret lines with these:
account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
twilio_number = st.secrets["TWILIO_PHONE_NUMBER"]
my_number = st.secrets["MY_PHONE_NUMBER"]
client = Client(account_sid, auth_token)

# 2. Website UI Design
st.set_page_config(page_title="AI Accident Alert", page_icon="🚨", layout="centered")
import streamlit as st

# --- HIGH-TECH AI DASHBOARD CSS ---
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

# --- APP UI CONTENT ---
st.markdown("<h1>👁️‍🗨️ NeuralVision: Smart Accident AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='ai-subtitle'>[ SYSTEM ACTIVE ] // Awaiting traffic surveillance feed...</p>", unsafe_allow_html=True)
st.markdown("---")

# The rest of your Python code continues below this...

# 3. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 4. Video Upload functionality
uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file for OpenCV to read
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() # Placeholder for video frames
    accident_detected = False

    st.info("Analyzing video frame by frame...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # AI Detection
        results = model.predict(frame, conf=0.60, verbose=False)
        annotated_frame = results[0].plot()
        
        # Display the video on the website
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)
        
        # Check for accidents
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls[0])]
                if class_name in ['severe', 'severe-accident', 'car-accident', 'car-crash']:
                    accident_detected = True
                    break
            if accident_detected:
                break
                
        if accident_detected:
            st.error("🚨 LIVE ACCIDENT DETECTED! Initiating emergency 108 call...")
            
            # Make Twilio Call
            try:
                emergency_message = '<Response><Say>Emergency alert! A severe car accident has been detected on the dashboard.</Say></Response>'
                call = client.calls.create(twiml=emergency_message, to=my_number, from_=twilio_number)
                st.success(f"Emergency Call Sent Successfully! Call SID: {call.sid}")
            except Exception as e:
                st.error("Call failed! Check your Twilio credentials.")
            break # Stop video after call

    cap.release()
    
    if not accident_detected:
        st.success("No accidents detected. The road is safe!")
