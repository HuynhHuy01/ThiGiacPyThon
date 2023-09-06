#pip install tensorflow opencv-python mediapipe sklearn matplotlib
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
#NHỚ BẤM Q VÀO CHƯƠNG TRÌNH ĐỂ TẮT

#Holistic Model nhận diện đối tượng và drawing util vẽ ra các line tương ứng 
mp_holistic = mp.solutions.holistic #Holistic model https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

#https://www.youtube.com/watch?v=doDUihpj6ro&t=422s 15:40 bắt đầu giải thích, 28:50 có demo xem chuyển màu
#tóm tắt là chuyển màu bình thường về một dạng màu khác (ko rõ có phải là trắng đen hay ko) để tiết kiệm bộ nhớ 
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Đổi màu lượt đầu
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #Đổi màu lại như cũ
    return image, results

# vẽ các line lên màn hình
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) #Vẽ các điểm nối mặt , có thể dùng FACEMESH_TESSELATION thay thế
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) #Vẽ các điểm nối dáng
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Vẽ các điểm tay trái
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Vẽ các điểm tay phải
# Như trên nhưng đổi màu và độ dày các đường vẽ
def draw_styled_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) #Vẽ các điểm nối mặt , có thể dùng FACEMESH_TESSELATION thay thế
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) #Vẽ các điểm nối dáng
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) #Vẽ các điểm tay trái
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) #Vẽ các điểm tay phải

# Show màn hình
cap = cv2.VideoCapture(0)
#Đặt mediapipe model https://www.youtube.com/watch?v=doDUihpj6ro&t=422s 21:00 có giải thích
#có thể chỉnh hai thông số cho phù hợp
with mp_holistic.Holistic(min_detection_confidence= 0.5,min_tracking_confidence= 0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read() #frame là hình ảnh lấy được từ camera
        
        #Bắt đầu nhận diện
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        #Vẽ các đường nối
        draw_styled_landmarks(image, results)
        
        cv2.imshow('OpenCV Feed', image)
        # Tắt camera bằng nút q
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
#####