import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
from googletrans import Translator  # нужно установить

# === НАСТРОЙКИ ===
api_url = "API_URL", "http://localhost:8000/detect""
DEFAULT_CONFIDENCE = 0.5  # изменено с 0.15 на 0.5

st.set_page_config(
    page_title="YOLO-World Object Detector",
    page_icon="🔍",
    layout="wide"
)

# Инициализируем переводчик
@st.cache_resource
def init_translator():
    return Translator()

translator = init_translator()

def translate_to_english(text: str) -> str:
    """Переводит текст с русского на английский"""
    if not text.strip():
        return text
    
    # Проверяем, есть ли русские буквы
    if any('\u0400' <= char <= '\u04FF' for char in text):
        try:
            translated = translator.translate(text, src='ru', dest='en')
            return translated.text
        except Exception as e:
            st.warning(f"Ошибка перевода: {e}. Использую исходный текст.")
            return text
    return text

st.title("🎯 Zero-Shot Object Detection with YOLO-World")
st.markdown("Загрузите изображение и напишите, что хотите найти (можно на русском - переведу автоматически)")

# === БОКОВАЯ ПАНЕЛЬ С ИНСТРУКЦИЕЙ ===
with st.sidebar:
    st.header("📖 Инструкция")
    st.markdown("""
    1. Загрузите изображение (JPG, PNG)
    2. Введите объекты для поиска (можно на **русском** или **английском**)
    3. Нажмите "Найти объекты!"
    
    **Примеры на русском:**
    - `человек, машина, собака`
    - `кот, рыба, миска`
    - `телефон, ноутбук, клавиатура`
    - `пицца, бургер, кофе`
    
    **Примеры на английском:**
    - `person, car, dog`
    - `cat, fish, bowl`
    
    💡 **Совет:** Модель понимает английский, я переведу автоматически!
    """)
    
    st.divider()
    
    # Настройка порога уверенности
    confidence_threshold = st.slider(
        "🎯 Порог уверенности",
        min_value=0.01,
        max_value=0.95,
        value=DEFAULT_CONFIDENCE,
        step=0.01,
        help="Чем выше, тем точнее, но меньше объектов найдется"
    )
    
    st.caption(f"Модель: YOLO-World v2 | Zero-shot detection")
    st.caption(f"Порог: {confidence_threshold}")

# === ОСНОВНОЙ ИНТЕРФЕЙС ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Шаг 1: Загрузите изображение")
    uploaded_file = st.file_uploader(
        "Выберите файл",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Показываем загруженное изображение
        image = Image.open(uploaded_file)
        st.image(image, caption="Исходное изображение", use_container_width=True)
        
        st.subheader("🔧 Шаг 2: Что ищем?")
        classes_input = st.text_input(
            "Введите классы (можно на русском или английском, через запятую)",
            placeholder="например: человек, машина, собака  или  person, car, dog",
            key="classes_input"
        )
        
        # Показываем перевод, если нужно
        if classes_input.strip():
            translated = translate_to_english(classes_input)
            if translated != classes_input:
                st.info(f"🔄 Перевод: `{classes_input}` → `{translated}`")
        
        # Кнопка запуска
        if st.button("🚀 Найти объекты!", type="primary", use_container_width=True):
            if not classes_input.strip():
                st.error("Введите хотя бы один класс для поиска!")
            else:
                # Переводим классы на английский
                classes_for_api = translate_to_english(classes_input)
                
                # Отправляем запрос к бэкенду
                with st.spinner("🔄 YOLO-World анализирует изображение..."):
                    try:
                        # Подготовка файла
                        files = {
                            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                        }
                        data = {
                            "classes": classes_for_api,
                            "confidence": confidence_threshold
                        }
                        
                        # POST запрос к FastAPI
                        response = requests.post(api_url, files=files, data=data, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            # Сохраняем результат в session_state
                            st.session_state.result = result
                            st.session_state.original_image = image
                            st.session_state.user_classes = classes_input
                            st.session_state.api_classes = classes_for_api
                            st.success(f"✅ Найдено объектов: {result['total_count']}")
                            st.rerun()
                        else:
                            st.error(f"Ошибка: {response.status_code}")
                            if response.text:
                                try:
                                    st.json(response.json())
                                except:
                                    st.write(response.text)
                            
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Не удалось подключиться к серверу. Запустите бэкенд: python api/main.py")
                    except Exception as e:
                        st.error(f"Ошибка: {str(e)}")

with col2:
    st.subheader("📸 Результат детекции")
    
    # Проверяем, есть ли результат в session_state
    if "result" in st.session_state and st.session_state.result:
        result = st.session_state.result
        original_image = st.session_state.original_image
        
        # Показываем, что искали (оригинал и перевод)
        st.caption(f"🔍 Искали: **{st.session_state.user_classes}**")
        if st.session_state.user_classes != st.session_state.api_classes:
            st.caption(f"🌐 Отправлено на сервер: **{st.session_state.api_classes}**")
        
        # Рисуем рамки на изображении
        draw_image = original_image.copy()
        draw = ImageDraw.Draw(draw_image)
        
        detected_objects = result.get("detected_objects", [])
        total_count = result.get("total_count", 0)
        
        # Цвета для разных классов
        colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FFA500", "#800080", "#00FFFF", "#FF69B4"]
        
        for i, obj in enumerate(detected_objects):
            # Получаем координаты bbox
            bbox = obj.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                class_name = obj.get("class", "unknown")
                confidence = obj.get("confidence", 0)
                
                # Показываем только объекты выше порога (двойная фильтрация)
                if confidence >= confidence_threshold:
                    # Выбираем цвет
                    color = colors[i % len(colors)]
                    
                    # Рисуем прямоугольник
                    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                    
                    # Рисуем подпись
                    label = f"{class_name} {confidence:.2f}"
                    # Задний фон для текста
                    text_bbox = draw.textbbox((x1, y1), label)
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((x1, y1), label, fill="white")
        
        # Показываем результат
        st.image(draw_image, caption=f"Найдено объектов: {total_count} (порог: {confidence_threshold})", use_container_width=True)
        
        # Детальный список найденного
        if detected_objects:
            with st.expander("📋 Детальный список найденных объектов"):
                for obj in detected_objects:
                    if obj.get("confidence", 0) >= confidence_threshold:
                        st.write(f"- **{obj['class']}** - уверенность: {obj['confidence']:.2f}")
        
        # Кнопка для сброса
        if st.button("🔄 Начать заново", use_container_width=True):
            del st.session_state.result
            del st.session_state.original_image
            st.rerun()
    else:
        st.info("👈 Загрузите изображение и введите классы для поиска")
        
        # Пример работы
        with st.expander("📖 Пример работы"):
            st.markdown("""
            **Что вы увидите после обработки:**
            - Изображение с цветными рамками вокруг найденных объектов
            - Название найденного объекта и уверенность модели
            - Общее количество найденных объектов
            
            **Пример запроса:**
            - Изображение: фото улицы
            - Запрос: `человек, машина, велосипед`
            - Результат: рамки вокруг всех людей, машин и велосипедов
            """)