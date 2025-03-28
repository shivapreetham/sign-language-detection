import os
import cv2
import pickle
import numpy as np
import mediapipe as mp

# Load the trained model and define the labels dictionary.
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Right'
}

# Weight factors (must match the ones used during landmark extraction)
WEIGHT_HAND = 1.0
WEIGHT_FACE = 0.5
WEIGHT_POSE = 0.3

def signDetection():
    cap = cv2.VideoCapture(0)
    
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    face_landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
    face_connection_spec = mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1)
    hand_landmark_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
    hand_connection_spec = mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2)
    pose_landmark_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    pose_connection_spec = mp_drawing.DrawingSpec(color=(0, 0, 200), thickness=2)

    with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3) as holistic:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            H, W, _ = frame.shape
            frame = cv2.flip(frame, 1)  # Mirror view for natural interaction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            # Optionally draw face and pose landmarks (for context)
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    results.face_landmarks, 
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=face_landmark_spec,
                    connection_drawing_spec=face_connection_spec)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=pose_landmark_spec,
                    connection_drawing_spec=pose_connection_spec)
            
            # Build the combined weighted feature vector.
            features = []
            # ----- HAND FEATURES -----
            if results.left_hand_landmarks:
                hand = results.left_hand_landmarks
            elif results.right_hand_landmarks:
                hand = results.right_hand_landmarks
            else:
                hand = None

            if hand:
                x_vals = [lm.x for lm in hand.landmark]
                y_vals = [lm.y for lm in hand.landmark]
                for lm in hand.landmark:
                    features.append((lm.x - min(x_vals)) * WEIGHT_HAND)
                    features.append((lm.y - min(y_vals)) * WEIGHT_HAND)
                # Draw hand landmarks.
                mp_drawing.draw_landmarks(
                    frame,
                    hand,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_spec,
                    connection_drawing_spec=hand_connection_spec)
            else:
                features.extend([0.0] * (21 * 2))
            
            # ----- FACE FEATURES -----
            if results.face_landmarks:
                face = results.face_landmarks
                face_x = [lm.x for lm in face.landmark]
                face_y = [lm.y for lm in face.landmark]
                for lm in face.landmark:
                    features.append((lm.x - min(face_x)) * WEIGHT_FACE)
                    features.append((lm.y - min(face_y)) * WEIGHT_FACE)
            else:
                features.extend([0.0] * (468 * 2))
            
            # ----- POSE FEATURES -----
            if results.pose_landmarks:
                pose = results.pose_landmarks
                pose_x = [lm.x for lm in pose.landmark]
                pose_y = [lm.y for lm in pose.landmark]
                for lm in pose.landmark:
                    features.append((lm.x - min(pose_x)) * WEIGHT_POSE)
                    features.append((lm.y - min(pose_y)) * WEIGHT_POSE)
            else:
                features.extend([0.0] * (33 * 2))
            
            # Use the combined feature vector for prediction.
            try:
                prediction = model.predict([np.asarray(features)])
                predicted_character = labels_dict[int(prediction[0])]
            except Exception as e:
                predicted_character = "Error"
                print("Prediction error:", e)
            
            # For visualization, if hand was detected, draw a bounding box around it.
            if hand:
                x1 = int(min(x_vals) * W) - 10
                y1 = int(min(y_vals) * H) - 10
                x2 = int(max(x_vals) * W) + 10
                y2 = int(max(y_vals) * H) + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, predicted_character, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            
            cv2.imshow('Sign Detection (Holistic Tracking)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    signDetection()
