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
st.title("🚨 Smart Traffic Accident Detection")
st.write("Upload a traffic surveillance video. The AI will monitor it and make an automated emergency call if an accident occurs.")

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
        results = model.predict(frame, conf=0.40, verbose=False)
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