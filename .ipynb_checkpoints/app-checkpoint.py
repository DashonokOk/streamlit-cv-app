import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

# Убираем лимит на размер файла (до 1GB)
if not os.path.exists(os.path.expanduser('~/.streamlit')):
    os.makedirs(os.path.expanduser('~/.streamlit'))
with open(os.path.expanduser('~/.streamlit/config.toml'), 'w') as f:
    f.write("[server]\nmaxUploadSize=1000")

# Загрузка модели
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Интерфейс
logo = Image.open('графики и лого/лого2.jpg')
st.image(logo, width=800)

video_file = st.file_uploader("Загрузите видео (до 1GB)", type=["mp4", "avi", "mov"])

if video_file:
    # Сохраняем видео во временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    video_box = st.empty()
    status_box = st.empty()

    # Получаем оригинальное разрешение видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Вычисляем пропорциональное уменьшение (макс. ширина 800px)
    max_width = 800
    if width > max_width:
        ratio = max_width / width
        target_size = (max_width, int(height * ratio))
    else:
        target_size = (width, height)

    frame_count = 0
    skip_frames = 1  # Пропуск каждого 2-го кадра

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % (skip_frames + 1) != 0:
            continue

        # Пропорциональное уменьшение кадра
        frame = cv2.resize(frame, target_size)
        results = model(frame, conf=0.5, verbose=False)

        # Обновление статуса
        if len(results[0].boxes) > 0:
            status_box.success("ТРАНСПОРТ ЕСТЬ")
        else:
            status_box.error("ТРАНСПОРТА НЕТ")

        # Отображение видео с автоматическим масштабированием
        video_box.image(results[0].plot(), channels="BGR", use_column_width=True)

    cap.release()
    os.unlink(tfile.name)  # Удаляем временный файл
    st.balloons()  # Анимация вместо текста
else:
    st.info("Загрузите видеофайл любого размера")
