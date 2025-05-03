import streamlit as st
import cv2
import numpy as np
import time
import math

# Load YOLOv4
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define unique colors for different object classes
colors = {
    "car": (255, 0, 0),        # Blue
    "bus": (0, 255, 255),      # Yellow
    "truck": (0, 165, 255),    # Orange
    "motorbike": (255, 0, 255),# Magenta
    "person": (0, 255, 0),     # Green
}

# Streamlit page config
st.set_page_config(page_title="YOLOv4 Vehicle Detection", layout="wide")
st.title("🚗 Vehicle & Pedestrian Detection with Info-Theory Metrics")

# Upload video file
video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
video_frame = st.empty()
stats_placeholder = st.empty()
info_placeholder = st.empty()

if video_file:
    tfile = open("temp_video.mp4", 'wb')
    tfile.write(video_file.read())
    cap = cv2.VideoCapture("temp_video.mp4")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    resolution = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    bit_depth = 24  # Assuming 8-bit RGB
    bandwidth = 5000000  # 5 Mbps (example)

    st.sidebar.write("🎞️ **Video Resolution:**", resolution)
    st.sidebar.write("🎞️ **Frame Rate (FPS):**", frame_rate)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Detect objects
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                label = classes[class_id]
                if confidence > 0.5 and label in ["car", "bus", "truck", "motorbike", "person"]:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        label_counts = {"car": 0, "motorbike": 0, "bus": 0, "truck": 0, "person": 0}

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            label_counts[label] += 1

            color = colors.get(label, (255, 255, 255))  # default white
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Info Theory Metrics
        total_objects = sum(label_counts.values())
        probs = {}
        if total_objects > 0:
            probs = {k: v / total_objects for k, v in label_counts.items() if v > 0}
            entropy = -sum(p * math.log2(p) for p in probs.values())
        else:
            entropy = 0

        frame_size_bits = resolution[0] * resolution[1] * bit_depth
        data_rate_bps = frame_size_bits * frame_rate
        channel_capacity = bandwidth * math.log2(1 + (data_rate_bps / bandwidth))

        # Convert frame for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame.image(frame_rgb, channels="RGB", use_container_width=True)

        # Display object counts and probabilities
        stats_placeholder.markdown(f"""
        ### 🧮 Object Detection Stats
        | Object     | Count | Probability |
        |------------|-------|-------------|
        | Car        | `{label_counts["car"]}` | `{probs.get("car", 0):.2f}` |
        | Motorbike  | `{label_counts["motorbike"]}` | `{probs.get("motorbike", 0):.2f}` |
        | Bus        | `{label_counts["bus"]}` | `{probs.get("bus", 0):.2f}` |
        | Truck      | `{label_counts["truck"]}` | `{probs.get("truck", 0):.2f}` |
        | Person     | `{label_counts["person"]}` | `{probs.get("person", 0):.2f}` |
        """)

        
        # Display info theory metrics
        info_placeholder.markdown(f"""
        ### 📊 Information Theory Metrics
        - **Total Detected Objects:** `{total_objects}`
        - **Entropy:** `{entropy:.2f} bits`
        - **Channel Capacity:** `{channel_capacity/1e6:.2f} Mbps`
        - **Data Transfer Rate:** `{data_rate_bps/1e6:.2f} Mbps`
        """)

        time.sleep(0.01)

    cap.release()
else:
    st.warning("⚠️ Please upload a video file to start detection.")
