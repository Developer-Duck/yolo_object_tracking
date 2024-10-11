import cv2
from ultralytics import YOLO
import numpy as np

# 학습한 YOLO 모델 경로를 지정합니다.
model_path = "C:/Users/user/Desktop/find_phone/yolo11x.pt"

# 학습한 YOLO 모델 로드
model = YOLO(model_path)

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 기본 웹캠 사용

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)  # 너비 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 높이 설정

while True:
    # 프레임 캡처
    ret, frame = cap.read()

    if not ret:
        break

    # 객체 감지 수행
    results = model(frame)

    # 감지된 객체의 정보를 가져옵니다.
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
    confidences = results[0].boxes.conf.cpu().numpy()  # 신뢰도
    class_ids = results[0].boxes.cls.cpu().numpy()  # 클래스 ID

    # 감지된 객체가 없는 경우 패스
    if len(confidences) == 0:
        cv2.imshow('YOLO Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # 프레임에 결과 렌더링
    annotated_frame = results[0].plot()  # plot()을 사용하여 프레임에 결과를 그림

    # 결과 프레임 표시
    cv2.imshow('YOLO Object Detection', annotated_frame)

    # 'q'를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
