#pip install tensorflow opencv-python mediapipe sklearn matplotlib scikit-learn
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import pyautogui
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, plot_model
import tensorflow as tf
print(tf.__version__)
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.callbacks import TensorBoard

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
actions = np.array(["father","hello","I","mother","see_you_later","what","again"])
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) #Vẽ các điểm nối mặt , có thể dùng FACEMESH_TESSELATION thay thế
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) #Vẽ các điểm nối dáng
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Vẽ các điểm tay trái
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Vẽ các điểm tay phải
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
def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
model = Sequential()
model.add(LSTM(50, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(50, return_sequences=True, activation='relu'))
model.add(LSTM(50, return_sequences=False, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.fit(X, y, epochs=100, callbacks=[tb_callback])
# model.save('action2.keras')
model.load_weights("action1.keras")
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)
holis = False
scaling_factor = 1.5
mp_hands = mp.solutions.hands
sequence = []
threshold = 0.7
count_frame = 0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
font_color = (255, 255, 255)
predicted_actions = []
display_text = ''
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, frame = cap.read()  # frame là hình ảnh lấy được từ camera
                if (holis == False):
                    image, results = mediapipe_detection(frame, hands)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <= hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y and hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y <= hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y and hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y <= hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y and hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y:
                                holis = True
                                time.sleep(2)
                    cv2.rectangle(image, (0, 0), (900, 60), (0, 0, 255), 2)
                    cv2.putText(image, display_text, (10, 40), font, font_scale, font_color, font_thickness)  # Display the actions
                    cv2.imshow('OpenCV Feed', image)
                else:
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    sequence.append(extract_keypoints(results))
                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        if res[np.argmax(res)] > threshold:
                            predicted_action = actions[np.argmax(res)]
                            print(predicted_action, res[np.argmax(res)])
                            predicted_actions.append(predicted_action)
                            predicted_actions = predicted_actions[-7:]
                            display_text = ' '.join(predicted_actions)
                            holis = False
                        
                        # Reset the sequence for the next 30 frames
                        sequence = []
                    cv2.imshow('OpenCV Feed', image)
                # Tắt camera bằng nút q
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()