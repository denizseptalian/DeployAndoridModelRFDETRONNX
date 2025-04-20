import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import base64
from io import BytesIO
import onnxruntime as ort
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Buah Sawit", layout="centered")

# Load model ONNX
@st.cache_resource
def load_model():
    session = ort.InferenceSession("inference_model.onnx", providers=["CPUExecutionProvider"])
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

# Fungsi untuk parsing output ONNX RF-DETR (disesuaikan dengan output spesifik model)
def draw_results(image, outputs):
    img = np.array(image.convert("RGB"))
    class_counts = Counter()

    # Output parsing tergantung struktur model ONNX kamu
    boxes, scores, class_ids = outputs  # contoh umum: [N, 4], [N], [N]

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
