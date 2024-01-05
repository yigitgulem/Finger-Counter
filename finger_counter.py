import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 482)

video = cv2.VideoWriter('hand.mp4',  
                        cv2.VideoWriter_fourcc(*'XVID'), 
                        float(16), (1280,960)) 

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

    if len(lmList) != 0:
        fingers = []
 
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        if fingers[1] == 0 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            cv2.putText(img, "love you", (20, 150), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (255, 0, 0), 5)
        elif lmList[8][2] > lmList[4][2]:  
            cv2.putText(img, "Nah", (20, 150), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (255, 0, 0), 5)
        else:
            cv2.putText(img, str(fingers.count(1)), (20, 150), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (48, 214, 200), 5)
    image_r = cv2.resize(img, (1280,960))
    video.write(image_r) 
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
