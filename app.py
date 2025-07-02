import cv2
import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import gc

# Конфигурация страницы
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Настройки сервера ---
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

# --- Логотип ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Убрано use_container_width для совместимости
    st.image(Image.open('графики и лого/лого2.jpg'), 
             width=800)  # Только width без use_container_width

st.markdown("---")

# --- Остальной код остается без изменений ---
video_file = st.file_uploader(
    "Загрузите видео (до 1GB)", 
    type=["mp4", "avi", "mov"]
)

if video_file:
    if video_file.size > 1000 * 1024 * 1024:
        st.error("Максимальный размер файла: 1GB")
        st.stop()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_file.read())
        video_path = tfile.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Ошибка при открытии видео")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        st.sidebar.info(f"""
        Характеристики видео:
        - Разрешение: {width}x{height}
        - Частота кадров: {fps:.1f}
        - Всего кадров: {total_frames}
        """)
        
        target_width = st.sidebar.slider(
            "Ширина обработки", 
            400, min(1200, width), 800
        )
        
        skip_frames = st.sidebar.slider(
            "Пропуск кадров", 
            0, 10, 2,
            help="Увеличение ускоряет обработку"
        )
        
        conf_threshold = st.sidebar.slider(
            "Минимальная уверенность", 
            0.1, 0.9, 0.5, 0.1
        )
        
        ratio = target_width / width
        target_size = (target_width, int(height * ratio))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        video_box = st.empty()
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % (skip_frames + 1) != 0:
                continue
                
            try:
                frame = cv2.resize(frame, target_size)
                results = model(frame, 
                              conf=conf_threshold, 
                              verbose=False, 
                              imgsz=640)
                
                if len(results[0].boxes) > 0:
                    status_text.success(f"Обнаружен транспорт (кадр {frame_count})")
                else:
                    status_text.warning("Транспорт не обнаружен")
                
                # Убрано use_container_width для вывода видео
                video_box.image(results[0].plot(), channels="BGR")
                
                del results
                gc.collect()
                
            except Exception as e:
                status_text.error(f"Ошибка: {str(e)}")
                continue
            
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
        cap.release()
        st.balloons()
        st.success("Анализ завершен!")
        
    except Exception as e:
        st.error(f"Ошибка обработки: {str(e)}")
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
        
else:
    st.info("Загрузите видео для начала анализа")
