import numpy as np
import pyautogui
import imutils
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Acessando os pontos de referência pela sua posição
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Array para manter verdadeiro ou falso se o dedo estiver dobrado    
            finger_fold_status = []
            for tip in finger_tips:
                # Obtendo a posição da ponta do ponto de referência e desenhando o círculo azul
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                # Escrevendo a condição para verificar se o dedo está dobrado
                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

            # Verificando se todos os dedos estão dobrados
            if all(finger_fold_status):
                # Simular um clique do mouse quando todos os dedos estão dobrados
                pyautogui.click()

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("Rastreamento de Maos", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Pressione 'Esc' para sair
        break

cap.release()
cv2.destroyAllWindows()





