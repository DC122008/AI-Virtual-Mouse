import cv2
import mediapipe as mp
import time
import math

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        # Initialize HandDetector with parameters
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        # Process the image to find hands
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks and draw:
            # Draw landmarks and connections on the image
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        # Find hand landmarks and bounding box
        x_list = []
        y_list = []
        bbox = []
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                self.lm_list.append([id, cx, cy])
                if draw:
                    # Draw circles at each landmark point
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = xmin - 20, ymin - 20, xmax + 20, ymax + 20

            if draw:
                # Draw a bounding box around the hand
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        return self.lm_list, bbox

    def fingers_up(self):
        # Check if each finger is up or down
        fingers = [1 if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1] else 0]

        for id in range(1, 5):
            fingers.append(1 if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2] else 0)

        return fingers

    def find_distance(self, point1, point2, img, draw=True, r=15, t=3):
        # Find the distance between two points on the hand
        x1, y1 = self.lm_list[point1][1:]
        x2, y2 = self.lm_list[point2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # Draw line and circles to visualize the distance
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

    @staticmethod
    def main():
        # Main function for testing the HandDetector class
        p_time = 0
        c_time = 0
        cap = cv2.VideoCapture(1)
        detector = HandDetector()
        
        while True:
            success, img = cap.read()
            img = detector.find_hands(img)
            lm_list, bbox = detector.find_position(img)
            
            if len(lm_list) != 0:
                print(lm_list[4])
                
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    HandDetector.main()
