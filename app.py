# File: C:/aiclass/oss_pomodoro/app.py
import streamlit as st
import streamlit.components.v1 as components
import time
import cv2
import pathlib
import numpy as np
from datetime import datetime

# Windows 경로 대응
import sys
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

from ultralytics import YOLO

# 모델 및 경로
MODEL_PATH = 'C:/aiclass/opensw_v8/openswbest3.pt'  ###openswbest6.pt의 경로 복사해서 붙여넣기###
model = YOLO(MODEL_PATH)
CONFIDENCE_THRESHOLD = 0.4 ###민감도 조정
IOU_THRESHOLD = 0.6  ### 민감도 조정정

# Streamlit UI 설정
TIMER_CSS = """
<style>
.circle{
  width:240px;height:240px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;margin:auto;}
.circle span{font:700 2.2rem monospace;color:#fff}
</style>"""

def draw_circle(remaining, total):
    pct = remaining / total
    angle = pct * 360
    mm, ss = divmod(remaining, 60)
    html = TIMER_CSS+f"""
    <div class="circle"
         style="background:
            conic-gradient(#e74c3c 0deg {angle}deg,
                           #eeeeee {angle}deg 360deg);">
      <span>{mm:02d}:{ss:02d}</span>
    </div>"""
    return html

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 감지 + 타이머 실행 (웹캠 스트리밍 포함)
def run_detection_with_timer(duration_sec, container_timer, container_video):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ 웹캠을 열 수 없습니다.")
        return

    end_time = time.time() + duration_sec
    last_hand_time = 0
    last_phone_time = 0
    names = model.names

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
        detected_labels = set()

        for box in results.boxes:
            cls_id = int(box.cls.item())
            cls_name = names[cls_id]
            detected_labels.add(cls_name)

            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
            color = (255, 255, 255)

            if cls_name == 'hand_with_pen':
                color = (255, 0, 0)
                last_hand_time = time.time()
            elif cls_name == 'smartphone':
                color = (0, 0, 255)
                last_phone_time = time.time()

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, cls_name, (xmin, max(0, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        container_video.image(rgb_frame, channels="RGB")

        remaining = int(end_time - time.time())
        with container_timer:
            components.html(draw_circle(remaining, duration_sec), height=260, scrolling=False)

        time.sleep(0.03)

    cap.release()

# --- Streamlit App Start ---
local_css("style.css")

st.write("""
# The Pomodoro App


""")

st.sidebar.title("Settings")
focus_min = st.sidebar.number_input("Focus Time (minutes)", 1, 60, 25)
break_min = st.sidebar.number_input("Break Time (minutes)", 1, 30, 5)

button_clicked = st.button("Start")

if button_clicked:
    focus_sec = focus_min * 60
    break_sec = break_min * 60

    st.subheader("🎥 Object Detection Running...")
    col1, col2 = st.columns([1, 2])
    container_timer = col1.empty()
    container_video = col2.empty()

    run_detection_with_timer(focus_sec, container_timer, container_video)
    st.toast("🔔 Focus complete! Time for a break.", icon="🍅")

    run_detection_with_timer(break_sec, container_timer, container_video)
    st.toast("⏰ Break is over!", icon="⏰")