import cv2
import mediapipe as mp
import numpy as np
import math
import time
from pynput.keyboard import Controller

mpHands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1,
                      min_detection_confidence=0.8, min_tracking_confidence=0.8)
keyboard = Controller()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()
cap.set(2, 150)

class Button:
    def __init__(self, pos, text, size=[70, 70]):
        self.pos = pos
        self.size = size
        self.text = text

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "CL", "Caps"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "SP"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "SW"]]

text, delay, app = "", 0, 0
finger_tips, sequence, last_update_time = [4, 8, 12, 16, 20], [], time.time()
update_interval, pattern = 0.5, [2, 3, 5]
screen_width, screen_height = 1920, 1080
buttonList = [Button([80 * j + 10, 80 * i + 10], key) for i in range(len(keys)) for j, key in enumerate(keys[i])]

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (96, 96, 96), cv2.FILLED)
        cv2.putText(img, button.text, (x + 10, y + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return img

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

x, y = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57], \
       [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)

def check_pattern(sequence, pattern):
    seq_len, pat_len = len(sequence), len(pattern)
    return seq_len >= pat_len and pattern == sequence[-pat_len:]

correct_keys, total_keypresses = 0, 0
start_time, caps_lock_on = time.time(), False

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    frame = cv2.resize(frame, (1000, 580))
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if app == 0:
        frame = drawAll(frame, buttonList)
        results = hands.process(rgb_frame)
        landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    hl, wl, cl = frame.shape
                    cx, cy = int(lm.x * wl), int(lm.y * hl)
                    landmarks.append([id, cx, cy])

        if landmarks:
            try:
                x5, y5, x17, y17 = landmarks[5][1], landmarks[5][2], landmarks[17][1], landmarks[17][2]
                dis = calculate_distance(x5, y5, x17, y17)
                A, B, C = coff
                distanceCM = A * dis ** 2 + B * dis + C

                if 20 < distanceCM < 50:
                    x, y, x3, y3 = landmarks[8][1], landmarks[8][2], landmarks[12][1], landmarks[12][2]

                    for button in buttonList:
                        xb, yb = button.pos
                        wb, hb = button.size
                        if xb < x < xb + wb and yb < y < yb + hb:
                            cv2.rectangle(frame, (xb - 5, yb - 5), (xb + wb + 5, yb + hb + 5), (160, 160, 160), cv2.FILLED)
                            cv2.putText(frame, button.text, (xb + 20, yb + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                            if calculate_distance(x, y, x3, y3) < 50 and delay == 0:
                                total_keypresses += 1
                                if button.text == "SP":
                                    text += " "
                                    keyboard.press(" ")
                                    correct_keys += 1
                                elif button.text == "CL":
                                    text = text[:-1]
                                    keyboard.press("\b")
                                    correct_keys += 1
                                elif button.text == "SW":
                                    app = 1
                                elif button.text == "Caps":
                                    caps_lock_on = not caps_lock_on
                                else:
                                    key = button.text.upper() if caps_lock_on else button.text.lower()
                                    text += key
                                    keyboard.press(key)
                                    correct_keys += 1
                                delay = 1

            except Exception as e:
                print(f"Error during hand processing: {e}")

        cv2.putText(frame, text, (20, 520), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    elif app == 1:
        results = hands.process(rgb_frame)
        total_fingers = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                fingers_up = [1 if landmarks[finger_tips[i]].y < landmarks[finger_tips[i] - 2].y else 0 for i in range(1, 5)]
                fingers_up.insert(0, 1 if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x else 0)
                total_fingers = sum(fingers_up)
                if time.time() - last_update_time > update_interval:
                    if len(sequence) == 0 or sequence[-1] != total_fingers:
                        sequence.append(total_fingers)
                    last_update_time = time.time()

        if check_pattern(sequence, pattern):
            app, sequence = 0, []

        sequence_text = "Sequence: " + " ".join(map(str, sequence))
        cv2.putText(frame, sequence_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Fingers: {total_fingers}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    if delay != 0:
        delay += 1
        if delay == 10:
            delay = 0

    elapsed_time = time.time() - start_time
    wpm = (len(text.split()) / elapsed_time) * 60 if elapsed_time > 0 else 0
    accuracy = (correct_keys / total_keypresses) * 100 if total_keypresses > 0 else 0

    cv2.putText(frame, f"WPM: {wpm:.2f}", (400, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (400, 370), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Correct Keys: {correct_keys}", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Keypresses: {total_keypresses}", (400, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Keyboard", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
