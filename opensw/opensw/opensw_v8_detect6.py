# opensw_v8_detect6.py
import os
import sys
import cv2
import time
import pathlib
from datetime import datetime

# Windows 경로 문제 해결
if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

# YOLOv8 설치 확인 및 임포트
try:
    from ultralytics import YOLO
except ImportError:
    print("🚀 YOLOv8 패키지가 없습니다. 설치 중...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# 모델 경로 지정 (절대 경로)
MODEL_PATH = 'C:/aiclass/opensw_v8/openswbest3.pt'
model = YOLO(MODEL_PATH)

# 감지 민감도 설정값
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.6

# 웹캠 장치 열기 (인덱스 확인 필요: HCAM01L을 위한 번호로 설정)
cap = cv2.VideoCapture(0)   ### 웸켐 번호는 0~4까지 있는데, 0 아님 1
if not cap.isOpened():
    print("❌ USB 웹캠을 열 수 없습니다.")
    sys.exit()

print("✅ YOLOv8 웹캠 감지 시작! 'q'를 눌러 종료합니다.")

# 상태 추적 관련 변수 초기화
prev_time = time.time()
total_times = {'STUDYING': 0, 'PLAYING': 0, 'NOTHING': 0}
current_status = 'NOTHING'
status_start_time = prev_time
status_periods = []
program_start_time = datetime.now()

# 이벤트 기록용 리스트
event_log = []
last_hand_time = 0
last_phone_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
    names = model.names
    detected_labels = set()

    for box in results.boxes:
        cls_id = int(box.cls.item())
        cls_name = names[cls_id]
        detected_labels.add(cls_name)

        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
        color = (255, 255, 255)  # 기본: 흰색

        if cls_name == 'hand_with_pen':
            color = (255, 0, 0)  # 파란색
            last_hand_time = time.time()
        elif cls_name == 'smartphone':
            color = (0, 0, 255)  # 붉은색
            last_phone_time = time.time()

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, cls_name, (xmin, max(0, ymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    now = time.time()

    # object_checker 로직
    if now - last_hand_time <= 2:
        detected_state = 'STUDYING'
    elif now - last_phone_time <= 1:
        detected_state = 'PLAYING'
    else:
        detected_state = 'NOTHING'

    # state_checker 로직 (1초마다 체크)
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

    # OpenCV 실시간 bounding box 출력
    cv2.imshow("YOLOv8 USB Webcam Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        now = time.time()
        start_dt = datetime.fromtimestamp(status_start_time)
        end_dt = datetime.fromtimestamp(now)
        duration = round(now - status_start_time, 2)
        status_periods.append((current_status, str(start_dt), str(end_dt), duration))
        total_times[current_status] += duration
        break

cap.release()
cv2.destroyAllWindows()

# 종료 후 전체 정보 출력
print("\n[상태변화]")
for state, start, end, duration in status_periods:
    print(f"{state} / 시작시간: {start} / 종료시간: {end} / 소요시간: {duration}초")

print("\n📊 상태별 총합 시간 (초):")
total_time_all = sum(total_times.values())
program_end_time = datetime.now()

for k, v in total_times.items():
    percentage = (v / total_time_all) * 100 if total_time_all else 0
    print(f"{k}: {round(v, 2)}초 ({round(percentage, 1)}%)")

print(f"\n🕒 실행 시작 시간: {program_start_time}")
print(f"🕒 실행 종료 시간: {program_end_time}")
print(f"⏱️ 총 실행 시간: {round((program_end_time - program_start_time).total_seconds(), 2)}초")
