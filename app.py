# File: ai_study_timer.py
import streamlit as st
import streamlit.components.v1 as components
import cv2
import time
import pathlib
import numpy as np
from datetime import datetime
from ultralytics import YOLO

if pathlib.os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath

MODEL_PATH = 'C:/aiclass/opensw_v8/openswbest3.pt'
model = YOLO(MODEL_PATH)
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.6

TIMER_CSS = """
<style>
.circle {
  width: 240px; height: 240px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; margin: auto;
}
.circle span { font: 700 2.2rem monospace; color: #fff; }
</style>"""

def draw_circle(remaining, total):
    pct = remaining / total
    angle = pct * 360
    mm, ss = divmod(remaining, 60)
    html = TIMER_CSS + f"""
    <div class="circle"
         style="background:
            conic-gradient(#e74c3c 0deg {angle}deg,
                           #eeeeee {angle}deg 360deg);">
      <span>{mm:02d}:{ss:02d}</span>
    </div>"""
    return html

def format_status_summary(status_periods, total_times, start_time, end_time):
    logs = []
    logs.append("\n[상태변화]")
    for state, start, end, duration in status_periods:
        logs.append(f"{state} / 시작시간: {start} / 종료시간: {end} / 소요시간: {duration}초")

    logs.append("\n📊 상태별 총합 시간 (초):")
    total_time_all = sum(total_times.values())
    for k, v in total_times.items():
        percentage = (v / total_time_all) * 100 if total_time_all else 0
        logs.append(f"{k}: {round(v, 2)}초 ({round(percentage, 1)}%)")

    logs.append(f"\n🕒 실행 시작 시간: {start_time}")
    logs.append(f"🕒 실행 종료 시간: {end_time}")
    logs.append(f"⏱️ 총 실행 시간: {round((end_time - start_time).total_seconds(), 2)}초")
    return '\n'.join(logs)

def run_detection_with_timer(duration_sec, container_timer, container_video, stop_signal):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ 웹캠을 열 수 없습니다.")
        return

    start_time = datetime.now()
    end_time = time.time() + duration_sec
    last_hand_time = 0
    last_phone_time = 0
    prev_time = time.time()
    status_periods = []
    total_times = {'STUDYING': 0, 'PLAYING': 0, 'NOTHING': 0}
    current_status = 'NOTHING'
    status_start_time = prev_time
    names = model.names

    while time.time() < end_time:
        if stop_signal():
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
        for box in results.boxes:
            cls_id = int(box.cls.item())
            cls_name = names[cls_id]
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

        now = time.time()
        if now - last_hand_time <= 2:
            detected_state = 'STUDYING'
        elif now - last_phone_time <= 1:
            detected_state = 'PLAYING'
        else:
            detected_state = 'NOTHING'

        if now - prev_time >= 1:
            prev_time = now
            if detected_state != current_status:
                start_dt = datetime.fromtimestamp(status_start_time)
                end_dt = datetime.fromtimestamp(now)
                duration = round(now - status_start_time, 2)
                status_periods.append((current_status, str(start_dt), str(end_dt), duration))
                total_times[current_status] += duration
                current_status = detected_state
                status_start_time = now

        remaining = int(end_time - time.time())
        with container_timer:
            components.html(draw_circle(remaining, duration_sec), height=260)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        container_video.image(rgb_frame, channels="RGB")
        time.sleep(0.03)

    cap.release()
    now = time.time()
    duration = round(now - status_start_time, 2)
    total_times[current_status] += duration
    status_periods.append((current_status, str(datetime.fromtimestamp(status_start_time)), str(datetime.fromtimestamp(now)), duration))
    end_time_dt = datetime.now()
    st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg")
    summary = format_status_summary(status_periods, total_times, start_time, end_time_dt)
    st.text_area("🧠 상태 요약", summary, height=300)

# --- Streamlit UI ---
st.title("🎓 AI Study Timer")

st.sidebar.title("🔧 설정")
focus_min = st.sidebar.number_input("📚 Focus Time (minutes)", 5, 60, 25)
break_min = st.sidebar.number_input("🛌 Break Time (minutes)", 1, 30, 5)

stop_flag = {'stop': False}

def stop_signal():
    return stop_flag['stop']

if st.button("▶ Start"):
    stop_flag['stop'] = False
    st.subheader("🎥 Object Detection Running...")
    container_timer = st.empty()
    container_video = st.empty()

    run_detection_with_timer(focus_min * 60, container_timer, container_video, stop_signal)
    if not stop_flag['stop']:
        st.toast("🔔 Focus complete! Time for a break.", icon="🍅")
        run_detection_with_timer(break_min * 60, container_timer, container_video, stop_signal)
        st.toast("⏰ Break is over!", icon="⏰")

if st.button("⏹ Stop"):
    stop_flag['stop'] = True