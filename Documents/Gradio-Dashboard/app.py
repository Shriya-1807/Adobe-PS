import gradio as gr
import numpy as np
import tempfile
import os
import cv2
from PIL import Image
from collections import Counter
import torch
import uuid
import time

from ultralytics import YOLOWorld
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel
from deep_sort_realtime.deepsort_tracker import DeepSort

# Set model checkpoint path
model_path = "last.pt"

def run_sahi_yolo_inference(image_pil, model_path, conf):
    image_np = np.array(image_pil.convert("RGB"))
    detection_model = UltralyticsDetectionModel(
        model_path=model_path,
        confidence_threshold=conf,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    result = get_sliced_prediction(
        image_np,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    unique_img_name = f"result_{uuid.uuid4().hex}.jpg"
    result_img_path = os.path.join(tempfile.gettempdir(), unique_img_name)
    try:
        result.export_visuals(
            export_dir=tempfile.gettempdir(),
            file_name=unique_img_name,
            text_size=0.5,
            rect_th=1,
            hide_labels=False,
            hide_conf=True,
        )
    except Exception as e:
        return None, {}, f"Export failed: {e}"
    # Read image as bytes to return to Gradio
    if not os.path.exists(result_img_path):
        return None, {}, f"Failed to create result image."
    result_img = Image.open(result_img_path)
    # Prepare class counts
    class_names = [pred.category.name for pred in result.object_prediction_list]
    class_counts = dict(Counter(class_names))
    return result_img, class_counts, "Inference successful!"

def image_inference_gradio(image, conf):
    result_img, class_counts, message = run_sahi_yolo_inference(image, model_path, conf)
    if result_img is None:
        return None, class_counts, message
    # Prepare a simple HTML string for class counts
    if class_counts:
        table_html = "<table><tr><th>Class</th><th>Count</th></tr>"
        for cls, count in class_counts.items():
            table_html += f"<tr><td>{cls}</td><td>{count}</td></tr>"
        table_html += "</table>"
    else:
        table_html = "<p>No objects detected.</p>"
    return result_img, table_html, message

def process_video_with_yolo_deepsort(video_in, conf=0.25, iou=0.4, skip_frames=2):
    # Save uploaded video to temp file
    temp_video_fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(temp_video_fd, 'wb') as f:
        f.write(video_in.read())
    yolo_model = YOLOWorld(model_path)
    tracker = DeepSort(max_age=10)
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return None, "Could not open the uploaded video."
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    base, ext = os.path.splitext(temp_video_path)
    temp_output_path = f"{base}_out{ext}"
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
    frame_count = 0
    prev_tracks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % skip_frames == 0:
            results = yolo_model.predict(frame, conf=conf, iou=iou, augment=False, verbose=False)
            results = results[0]
            detections = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                box_conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append(([x1, y1, x2 - x1, y2 - y1], box_conf, cls))
            tracks = tracker.update_tracks(detections, frame=frame)
            prev_tracks = tracks
        else:
            tracks = prev_tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, r, b = map(int, track.to_ltrb())
            track_id = track.track_id
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(frame)
    cap.release()
    out.release()
    # Return processed video as file object
    if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 1000:
        return temp_output_path, "Video processed!"
    else:
        return None, "Processed video not found or is empty."

def video_inference_gradio(video, conf, iou):
    result_path, msg = process_video_with_yolo_deepsort(video, conf, iou)
    if result_path:
        return result_path, msg
    else:
        return None, msg

# Gradio interfaces
image_inputs = [
    gr.Image(type="pil", label="Upload Drone Image"),
    gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Model Confidence")
]
image_outputs = [
    gr.Image(label="Detected Output"),
    gr.HTML(label="Object Counts"),
    gr.Textbox(label="Status Message")
]

video_inputs = [
    gr.File(label="Upload Drone Video (mp4/avi/mov/mkv)"),
    gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Model Confidence"),
    gr.Slider(0.1, 1.0, value=0.4, step=0.1, label="IOU Threshold")
]
video_outputs = [
    gr.Video(label="Processed & Tracked Video"),
    gr.Textbox(label="Status Message")
]

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🚀 Drone Footage Object Detection and Tracking (Gradio)")
    with gr.Tab("Image Detection"):
        gr.Interface(fn=image_inference_gradio, inputs=image_inputs, outputs=image_outputs)
    with gr.Tab("Video Detection & Tracking"):
        gr.Interface(fn=video_inference_gradio, inputs=video_inputs, outputs=video_outputs)

if __name__ == "__main__":
    demo.launch()

