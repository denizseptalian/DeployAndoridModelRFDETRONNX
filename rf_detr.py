import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import os
import gdown
import onnxruntime as ort
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Buah Sawit", layout="centered")

# --- Fungsi untuk download model dari Google Drive ---
def download_model_from_drive(file_id, destination):
    if not os.path.exists(destination):
        st.info("Mengunduh model dari Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
        st.success("Model berhasil diunduh.")
    else:
        st.info("Model sudah tersedia secara lokal.")

# --- Load model ONNX ---
@st.cache_resource
def load_model():
    file_id = "1GNBuvCB_UvoFiFjVBqT2Hpabu1zfOJtU"  # Ganti dengan ID Google Drive model kamu
    model_path = "inference_model.onnx"
    download_model_from_drive(file_id, model_path)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session

# Fungsi prediksi ONNX
def predict_image(session, image):
    img = np.array(image.convert("RGB"))
    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    inputs = {session.get_inputs()[0].name: img_input}
    outputs = session.run(None, inputs)

    return outputs  # akan diproses di draw_results

# Warna bounding box sesuai label
label_to_color = {
    "Masak": Color.RED,
    "Mengkal": Color.YELLOW,
    "Mentah": Color.BLACK
}

label_annotator = LabelAnnotator()

# Fungsi untuk parsing output ONNX RF-DETR
def draw_results(image, outputs):
    img = np.array(image.convert("RGB"))
    class_counts = Counter()

    boxes, scores, class_ids = outputs
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids).astype(int)

    for box, score, class_id in zip(boxes, scores, class_ids):
        if score < 0.3:
            continue

        x_min, y_min, x_max, y_max = box
        label_name = {0: "Mentah", 1: "Mengkal", 2: "Masak"}.get(class_id, f"Kelas {class_id}")
        label = f"{label_name}: {score:.2f}"
        color = label_to_color.get(label_name, Color.WHITE)

        class_counts[label_name] += 1

        box_annotator = BoxAnnotator(color=color)
        detection = Detections(
            xyxy=np.array([[x_min, y_min, x_max, y_max]]),
            confidence=np.array([score]),
            class_id=np.array([class_id])
        )

        img = box_annotator.annotate(scene=img, detections=detection)
        img = label_annotator.annotate(scene=img, detections=detection, labels=[label])

    return img, class_counts

# --- UI Streamlit ---
st.title("🍊 Deteksi Buah Sawit Menggunakan RF-DETR")

session = load_model()

uploaded_image = st.file_uploader("Unggah gambar buah sawit", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    with st.spinner("Mendeteksi..."):
        outputs = predict_image(session, image)
        img_result, counts = draw_results(image, outputs)

    st.image(img_result, caption="Hasil Deteksi", use_column_width=True)
    st.markdown("### Jumlah Deteksi per Kelas:")
    for label, count in counts.items():
        st.write(f"- **{label}**: {count}")
