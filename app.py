import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

@st.cache_resource 
def load_model():
    return YOLO('best.pt', task='detect')

model = load_model()

logo = Image.open('графики и лого/лого2.jpg')
st.image(logo, width=700)

video_file = st.file_uploader("Загрузите видео (до 1GB)", type=["mp4", "avi", "mov"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    video_box = st.empty()
    status_box = st.empty()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_width = 1200
    if width > max_width:
        ratio = max_width / width
        target_size = (max_width, int(height * ratio))
    else:
        target_size = (width, height)

    frame_count = 0
    skip_frames = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % (skip_frames + 1) != 0:
            continue

        frame = cv2.resize(frame, target_size)
        results = model(frame, conf=0.5, verbose=False)

        if len(results[0].boxes) > 0:
            status_box.success("ТРАНСПОРТ ЕСТЬ")
        else:
            status_box.error("ТРАНСПОРТА НЕТ")

        result_img = results[0].plot()
        display_width = 1200 
        display_height = int(display_width * (height/width))
        resized_img = cv2.resize(result_img, (display_width, display_height))
        video_box.image(resized_img, channels="BGR")

    cap.release()
    os.unlink(tfile.name)
    st.balloons()
else:
    st.info("Загрузите видеофайл")
    
    #venv310
