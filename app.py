#ai_pomodoro_windows.py
import streamlit as st
import time
import cv2
from pathlib import Path
import numpy as np
import sys
from datetime import datetime
import torch
from ultralytics import YOLO
from streamlit import components

# ================= GPU 및 모델 로딩 =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] YOLO 디바이스: {DEVICE}")

# Windows 용 경로 (절대 경로 or 상대 경로)
MODEL_PATH = Path("/openswlast_11n.pt")  # 경로 넣으실 때, \를 /로 바꿔야 할 겁니다.
model = YOLO(str(MODEL_PATH)).to(DEVICE)

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.6
FRAME_SKIP = 3

TIMER_CSS = """<style>
.circle {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.circle span {
  font: 600 1.2rem monospace;
  color: #fff;
}
</style>"""

def draw_circle(remaining, total, color="red"):
    pct = remaining / total if total else 0
    angle = pct * 360
    mm, ss = divmod(remaining, 60)
    grad_color = "#e74c3c" if color == "red" else "#3498db"
    html = TIMER_CSS + f"""
    <div class='circle'
         style='background:
            conic-gradient({grad_color} 0deg {angle}deg,
                           #eeeeee {angle}deg 360deg);'>
      <span>{mm:02d}:{ss:02d}</span>
    </div>"""
    return html

def speak_alert():
    st.components.v1.html("""
    <script>
    const msg = new SpeechSynthesisUtterance("스마트폰을 사용하지 마세요.");
    window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

def init_state():
    defaults = {
        "cap": None,
        "running": False,
        "paused": False,
        "time_left": 0,
        "set_index": 1,
        "cycle_type": "focus",
        "start_requested": False,
        "stop_flag": False,
        "start_time": None,
        "hand_time": 0,
        "phone_time": 0,
        "neutral_time": 0,
        "completed": False,
        "last_frame_time": None,
        "startup_latency": 0,
        "smartphone_detected": False,
        "hand_with_pen_detected": False,
        "mode": "기본",
        "frame_skip_count": 0,
        "last_results": None,
        "last_alert_time": 0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# Sidebar 설정
st.sidebar.title("설정")
focus_sec = st.sidebar.number_input("집중 시간 (초)", 10, 3600, 20)
break_sec = st.sidebar.number_input("쉬는 시간 (초)", 1, 1800, 5)
total_sets = st.sidebar.number_input("세트 수", 1, 10, 2)
st.sidebar.text_area("📜 오늘 할 일 목록")
st.session_state.mode = st.sidebar.selectbox(
    "📲 감지 모드",
    ["기본", "폰 감지 시 정지", "스마트폰 감지 시 알림", "펜만 감지 시 작동"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("ℹ️ **버튼이 잘 실행되지 않는다면 여러 번 눌러보세요.**")

message_placeholder = st.empty()

# 컨트롤 버튼
btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 2])
with btn1:
    if st.button("▶ 시작"):
        st.session_state.running = True
        st.session_state.paused = False
        st.session_state.set_index = 1
        st.session_state.cycle_type = "focus"
        st.session_state.time_left = focus_sec
        st.session_state.start_requested = True
        st.session_state.stop_flag = False
        st.session_state.completed = False
        st.session_state.hand_time = 0
        st.session_state.phone_time = 0
        st.session_state.neutral_time = 0
        st.session_state.startup_latency = 0
        st.session_state.frame_skip_count = 0
        st.session_state.last_alert_time = 0

with btn2:
    if st.button("⏯ 일시정지 / 재시작"):
        if st.session_state.running:
            st.session_state.running = False
            st.session_state.paused = True
        elif st.session_state.paused and st.session_state.time_left > 0:
            st.session_state.running = True
            st.session_state.paused = False
            st.rerun()

with btn3:
    if st.button("🔄 전체 초기화"):
        st.session_state.running = False
        st.session_state.paused = False
        st.session_state.set_index = 1
        st.session_state.cycle_type = "focus"
        st.session_state.time_left = focus_sec
        st.session_state.completed = False
        message_placeholder.success("🔄 세트와 타이머가 초기화되었습니다. [▶ 시작] 버튼을 눌러주세요.")

with btn4:
    if st.button("⏲ 타이머 중지 (리셋)"):
        st.session_state.time_left = focus_sec if st.session_state.cycle_type == "focus" else break_sec
        st.session_state.running = True
        st.session_state.paused = False
        st.session_state.start_requested = True
        message_placeholder.success("⏲ 타이머만 초기화되었습니다. 타이머 중지 (리셋) 버튼을 다시 눌러주세요.")


colL, colR = st.columns([1, 3])
status_placeholder = st.empty()
with colL:
    st.markdown("### 🔵 타이머")
    container_timer = st.empty()
with colR:
    container_video = st.empty()
    status_text = st.empty()

def show_frame():
    ret, frame = st.session_state.cap.read()
    if not ret:
        return

    frame = cv2.resize(frame, (640, 640))
    now = time.time()
    delta = now - st.session_state.last_frame_time
    st.session_state.last_frame_time = now

    if st.session_state.frame_skip_count == 0:
        results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
        st.session_state.last_results = results
    else:
        results = st.session_state.last_results

    st.session_state.frame_skip_count = (st.session_state.frame_skip_count + 1) % FRAME_SKIP

    smartphone_detected = False
    hand_with_pen_detected = False
    found = False

    for box in results.boxes:
        cls_id = int(box.cls.item())
        cls_name = model.names[cls_id]
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
        color = (255, 0, 0) if cls_name == 'hand_with_pen' else (0, 0, 255) if cls_name == 'smartphone' else (255, 255, 255)

        if cls_name == 'hand_with_pen':
            st.session_state.hand_time += delta
            hand_with_pen_detected = True
        elif cls_name == 'smartphone':
            st.session_state.phone_time += delta
            smartphone_detected = True
            if st.session_state.mode == "스마트폰 감지 시 알림":
                now = time.time()
                if now - st.session_state.last_alert_time > 5:
                    speak_alert()
                    st.session_state.last_alert_time = now

        found = True
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, cls_name, (xmin, max(0, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if not found:
        st.session_state.neutral_time += delta

    st.session_state.smartphone_detected = smartphone_detected
    st.session_state.hand_with_pen_detected = hand_with_pen_detected

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    container_video.image(rgb, channels="RGB")

def update_timer_ui(duration):
    color = "blue" if st.session_state.cycle_type == "break" else "red"
    html = draw_circle(st.session_state.time_left, duration, color)
    with container_timer:
        components.v1.html(html, height=120)

def run_timer(duration):
    if st.session_state.cycle_type == "focus":
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        while not st.session_state.cap.isOpened():
            time.sleep(0.05)
        st.session_state.last_frame_time = time.time()
        st.session_state.startup_latency += time.time() - st.session_state.last_frame_time

    prev_time = time.time()

    while st.session_state.time_left > 0:
        now = time.time()
        delta = now - prev_time

        if not st.session_state.running:
            update_timer_ui(duration)
            time.sleep(0.05)
            continue

        if st.session_state.cycle_type == "focus":
            show_frame()
            if delta >= 1:
                if st.session_state.mode == "폰 감지 시 정지":
                    if not st.session_state.smartphone_detected:
                        st.session_state.time_left -= 1
                elif st.session_state.mode == "펜만 감지 시 작동":
                    if st.session_state.hand_with_pen_detected:
                        st.session_state.time_left -= 1
                else:
                    st.session_state.time_left -= 1
                prev_time = now
        else:
            container_video.markdown("💤 휴식 중입니다")
            if delta >= 1:
                st.session_state.time_left -= 1
                prev_time = now

        st.session_state.time_left = max(0, st.session_state.time_left)
        update_timer_ui(duration)
        status = f"{total_sets}세트 중 {st.session_state.set_index}세트 {('집중' if st.session_state.cycle_type == 'focus' else '휴식중')}"
        status_text.subheader(status)
        time.sleep(0.01)

    if st.session_state.cap and st.session_state.cycle_type == "focus":
        st.session_state.cap.release()
        st.session_state.cap = None

def run_timer_cycle():
    while st.session_state.running and st.session_state.set_index <= total_sets:
        duration = focus_sec if st.session_state.cycle_type == "focus" else break_sec
        run_timer(duration)

        if not st.session_state.running or st.session_state.paused:
            break

        if st.session_state.cycle_type == "focus":
            st.session_state.cycle_type = "break"
            st.session_state.time_left = break_sec
        else:
            st.session_state.set_index += 1
            if st.session_state.set_index > total_sets:
                st.session_state.running = False
                st.session_state.completed = True
                status_placeholder.success("🎉 모든 세트 완료!")
                break
            else:
                st.session_state.cycle_type = "focus"
                st.session_state.time_left = focus_sec

if st.session_state.running:
    run_timer_cycle()
else:
    update_timer_ui(focus_sec if st.session_state.cycle_type == "focus" else break_sec)
    if st.session_state.paused:
        container_video.markdown("⏸ 지금은 정지 상태입니다.")
    elif st.session_state.stop_flag:
        container_video.markdown("⏹ 지금은 중지 상태입니다.")

if st.session_state.completed:
    pen_time = max(0, st.session_state.hand_time - st.session_state.startup_latency)
    phone_time = max(0, st.session_state.phone_time - st.session_state.startup_latency)
    neutral_time = max(0, st.session_state.neutral_time - st.session_state.startup_latency)
    study_total = pen_time + phone_time + neutral_time
    break_total = total_sets * break_sec
    break_min, break_sec_rem = divmod(break_total, 60)

    st.markdown("### 🏁 결과 요약")
    st.markdown("#### 🟥 공부 시간 요약")
    st.table({
        "카테고리": ["1. studying", "2. playing", "3. nothing", "4. 총 시간"],
        "값": [f"{pen_time:.1f}s", f"{phone_time:.1f}s", f"{neutral_time:.1f}s", f"{study_total:.1f}s"]
    })
    st.markdown("#### 🟦 휴식 시간 요약")
    st.table({
        "카테고리": ["1. 총 휴식 시간"],
        "값": [f"{int(break_min):02d}:{int(break_sec_rem):02d}"]
    })