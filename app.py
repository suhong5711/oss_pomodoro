# File: ai_study_timer.py
import streamlit as st
import streamlit.components.v1 as components
import cv2
import time
import pathlib
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from threading import Thread

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

def format_status_summary(status_periods, total_times):
    logs = []
    logs.append("\n[상태변화]")
    for state, start, end, duration in status_periods:
        logs.append(f"{state} / 시작시간: {start} / 종료시간: {end} / 소요시간: {duration}초")

    logs.append("\n📊 상태별 총합 시간 (초):")
    total_time_all = sum(total_times.values())
    for k, v in total_times.items():
        percentage = (v / total_time_all) * 100 if total_time_all else 0
        logs.append(f"{k}: {round(v, 2)}초 ({round(percentage, 1)}%)")
    return '\n'.join(logs)

class DetectionSession:
    def __init__(self):
        self.running = False
        self.paused = False
        self.cap = None
        self.remaining = 0
        self.total = 0
        self.status_periods = []
        self.total_times = {'STUDYING': 0, 'PLAYING': 0, 'NOTHING': 0}
        self.current_status = 'NOTHING'
        self.status_start_time = time.time()
        self.last_hand_time = 0
        self.last_phone_time = 0
        self.container_video = None
        self.container_timer = None
        self.log = ""

    def start(self, total, container_timer, container_video):
        self.running = True
        self.paused = False
        self.remaining = total
        self.total = total
        self.container_video = container_video
        self.container_timer = container_timer
        Thread(target=self.run).start()

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            st.error("❌ 웹캠을 열 수 없습니다.")
            return

        prev_time = time.time()
        self.status_start_time = prev_time
        names = model.names
        end_time = time.time() + self.remaining

        while self.running and self.remaining > 0:
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
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
                    self.last_hand_time = time.time()
                elif cls_name == 'smartphone':
                    color = (0, 0, 255)
                    self.last_phone_time = time.time()
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, cls_name, (xmin, max(0, ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            now = time.time()
            if now - self.last_hand_time <= 2:
                detected_state = 'STUDYING'
            elif now - self.last_phone_time <= 1:
                detected_state = 'PLAYING'
            else:
                detected_state = 'NOTHING'

            if now - prev_time >= 1:
                prev_time = now
                if detected_state != self.current_status:
                    start_dt = datetime.fromtimestamp(self.status_start_time)
                    end_dt = datetime.fromtimestamp(now)
                    duration = round(now - self.status_start_time, 2)
                    self.status_periods.append((self.current_status, str(start_dt), str(end_dt), duration))
                    self.total_times[self.current_status] += duration
                    self.current_status = detected_state
                    self.status_start_time = now
                self.remaining = int(end_time - now)

            self.container_timer.empty()
            self.container_timer.components.html(draw_circle(self.remaining, self.total), height=260)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.container_video.image(rgb_frame, channels="RGB")
            time.sleep(0.03)

        self.cap.release()
        self.running = False
        st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg")
        self.log = format_status_summary(self.status_periods, self.total_times)

    def pause(self):
        self.paused = not self.paused
        if self.cap:
            if self.paused:
                self.cap.release()
            else:
                self.cap = cv2.VideoCapture(0)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def reset(self):
        self.stop()
        self.status_periods.clear()
        self.total_times = {'STUDYING': 0, 'PLAYING': 0, 'NOTHING': 0}
        self.current_status = 'NOTHING'
        self.status_start_time = time.time()
        self.log = ""

# --- Streamlit UI ---
st.title("🎓 AI Study Timer")
session = DetectionSession()

with st.sidebar:
    st.header("🔧 설정")
    focus_min = st.number_input("📚 Focus Time (minutes)", 5, 60, 25)
    break_min = st.number_input("🛌 Break Time (minutes)", 1, 30, 5)

    if st.button("▶ Start"):
        col1, col2 = st.columns([1, 2])
        container_timer = col1.empty()
        container_video = col2.empty()
        session.start(focus_min * 60, container_timer, container_video)

    if st.button("⏸ Pause / Resume"):
        session.pause()

    if st.button("🔁 Reset"):
        session.reset()

    if st.button("⏹ Stop"):
        session.stop()
        st.text_area("🧠 상태 요약", session.log, height=300)