import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

# UI Design
st.set_page_config(page_title="Debug Mode", layout="centered")
st.markdown("<h2 style='color: yellow; text-align: center;'>🛠️ X-Ray Debug Mode (Finding the Bug)</h2>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  

model = load_model()

def get_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1 = max(x1, x3); yi1 = max(y1, y3)
    xi2 = min(x2, x4); yi2 = min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area > 0:
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area
    return 0.0

uploaded_file = st.file_uploader("Upload that 2-Car Crash Video...", type=['mp4', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty() 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break  
            
        results = model.predict(frame, conf=0.30, verbose=False)
        vehicles = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) in [2, 3, 5, 7]:
                    vehicles.append(box.xyxy[0].cpu().numpy())

        annotated_frame = frame.copy()
        
        # Draw all boxes with Numbers
        for idx, v in enumerate(vehicles):
            cv2.rectangle(annotated_frame, (int(v[0]), int(v[1])), (int(v[2]), int(v[3])), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"V{idx+1}", (int(v[0]), int(v[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show Overlap Score if any 2 vehicles touch
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                iou_score = get_iou(vehicles[i], vehicles[j])
                if iou_score > 0.01: # Even 1% touch
                    cv2.putText(annotated_frame, f"OVERLAP: {iou_score:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)
