import streamlit as st
from ultralytics import YOLO
from twilio.rest import Client
import cv2
import tempfile

# 1. Twilio Setup
client = None
if "TWILIO_ACCOUNT_SID" in st.secrets:
    client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])

# 2. Load Reliable Model
model = YOLO('yolov8n.pt') 

# 3. Logic: Physics based collision
st.title("🚨 NeuralVision: Physics-Based Accident Alert")

uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    
    triggered = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Detect vehicles accurately
        results = model.predict(frame, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # PHYSICS CHECK: Are any two boxes overlapping deeply?
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                b1, b2 = boxes[i], boxes[j]
                
                # Intersection calculation
                x_inter = max(b1[0], b2[0])
                y_inter = max(b1[1], b2[1])
                w_inter = min(b1[2], b2[2]) - x_inter
                h_inter = min(b1[3], b2[3]) - y_inter
                
                if w_inter > 0 and h_inter > 0:
                    # They are overlapping! This is a crash.
                    st.error("🚨 CRASH DETECTED VIA PHYSICS OVERLAP!")
                    if client and not triggered:
                        client.calls.create(twiml='<Response><Say>Accident alert!</Say></Response>',
                                            to=st.secrets["MY_PHONE_NUMBER"], from_=st.secrets["TWILIO_PHONE_NUMBER"])
                        triggered = True
                    break
        
        annotated = results[0].plot()
        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        if triggered: break
