import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import base64
from io import BytesIO
import os
import requests
import onnxruntime as ort
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Buah Sawit", layout="centered")

# --- Fungsi untuk download model dari Google Drive ---
def download_model_from_drive(file_id, destination):
    if not os.path.exists(destination):
        st.info("Mengunduh model dari Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        with open(destination, "wb") as f:
            f.write(response.content)
        st.success("Model berhasil diunduh.")
    else:
        st.info("Model sudah tersedia secara lokal.")

# --- Load model ONNX ---
@st.cache_resource
def load_model():
    file_id = "1GNBuvCB_UvoFiFjVBqT2Hpabu1zfOJtU"  # Ganti dengan file ID model ONNX kamu
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

    return outputs

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

# --- Streamlit UI ---
st.title("📸 Deteksi Buah Sawit Menggunakan RF-DETR")

uploaded_file = st.file_uploader("Unggah gambar buah sawit", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Diupload", use_column_width=True)

    session = load_model()

    with st.spinner("Melakukan deteksi..."):
        outputs = predict_image(session, image)
        result_img, counts = draw_results(image, outputs)

    st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

    st.subheader("Jumlah Deteksi:")
    for label, count in counts.items():
        st.write(f"- **{label}**: {count} buah")
