import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from PIL import Image

# Конфигурация страницы ДОЛЖНА быть первой
st.set_page_config(
    page_title="Детектор транспорта",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Создание конфигурации ---
if not os.path.exists('.streamlit'):
    os.makedirs('.streamlit')
    
with open('.streamlit/config.toml', 'w') as f:
    f.write("""
[server]
maxUploadSize = 1000
enableCORS = false
""")

# --- Загрузка модели ---
@st.cache_resource 
def load_model():
    return YOLO('best.pt', task='detect')

model = load_model()

# --- Интерфейс с центрированным лого ---
logo = Image.open('графики и лого/лого2.jpg')

# Создаем 3 колонки (левая пустая, центральная с лого, правая пустая)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Центральная колонка
    st.image(logo, width=800)  # Ширина может быть любой
    st.title("Детектор транспорта", anchor='center')

# Горизонтальная линия для разделения
st.markdown("---")

# --- Остальной код остается без изменений ---
video_file = st.file_uploader(
    "Загрузите видео (до 1GB)", 
    type=["mp4", "avi", "mov"],
    accept_multiple_files=False
)

if video_file:
    if video_file.size > 1000 * 1024 * 1024:
        st.error("Ошибка: Файл слишком большой. Максимальный размер: 1GB")
        st.stop()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    video_box = st.empty()
    status_box = st.empty()
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st.sidebar.info(f"""
    Параметры видео:
    - Разрешение: {width}x{height}
    - FPS: {fps:.1f}
    - Размер файла: {video_file.size / (1024*1024):.1f} MB
    """)
    
    target_width = st.sidebar.slider(
        "Ширина отображения", 
        400, 1200, 1000
    )
    
    skip_frames = st.sidebar.slider(
        "Пропуск кадров", 
        0, 5, 1,
        help="Увеличьте для ускорения обработки"
    )
    
    conf_threshold = st.sidebar.slider(
        "Порог уверенности", 
        0.1, 0.9, 0.5, 0.1
    )
    
    ratio = target_width / width
    target_size = (target_width, int(height * ratio))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % (skip_frames + 1) != 0:
            continue
            
        frame = cv2.resize(frame, target_size)
        results = model(frame, conf=conf_threshold, verbose=False)
        
        if len(results[0].boxes) > 0:
            status_box.success(f"ТРАНСПОРТ ОБНАРУЖЕН (кадр {frame_count})")
        else:
            status_box.error(f"ТРАНСПОРТА НЕТ (кадр {frame_count})")
        
        result_img = results[0].plot()
        display_height = int(target_width * (height/width))
        resized_img = cv2.resize(result_img, (target_width, display_height))
        video_box.image(resized_img, channels="BGR")
    
    cap.release()
    os.unlink(tfile.name)
    st.balloons()
    st.success("Обработка завершена!")
    st.audio(video_file)
    
else:
    st.info("Пожалуйста, загрузите видеофайл для анализа")

#venv310
