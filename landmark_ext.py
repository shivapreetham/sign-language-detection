import os
import pickle
import cv2
import mediapipe as mp

# Define weight factors for each modality
WEIGHT_HAND = 1.0
WEIGHT_FACE = 0.5
WEIGHT_POSE = 0.3

DATA_DIR = './data'
data = []
labels = []

mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.3) as holistic:
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue

        for img_path in os.listdir(dir_path):
            img_file = os.path.join(dir_path, img_path)
            img = cv2.imread(img_file)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)
            feature_vector = []

            # ----- HAND FEATURES -----
            # Prefer left hand; if not, use right hand
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
                    feature_vector.append((lm.x - min(x_vals)) * WEIGHT_HAND)
                    feature_vector.append((lm.y - min(y_vals)) * WEIGHT_HAND)
            else:
                # 21 landmarks × 2 features each
                feature_vector.extend([0.0] * (21 * 2))

            # ----- FACE FEATURES -----
            if results.face_landmarks:
                face = results.face_landmarks
                face_x = [lm.x for lm in face.landmark]
                face_y = [lm.y for lm in face.landmark]
                for lm in face.landmark:
                    feature_vector.append((lm.x - min(face_x)) * WEIGHT_FACE)
                    feature_vector.append((lm.y - min(face_y)) * WEIGHT_FACE)
            else:
                # 468 landmarks × 2 features each
                feature_vector.extend([0.0] * (468 * 2))

            # ----- POSE FEATURES -----
            if results.pose_landmarks:
                pose = results.pose_landmarks
                pose_x = [lm.x for lm in pose.landmark]
                pose_y = [lm.y for lm in pose.landmark]
                for lm in pose.landmark:
                    feature_vector.append((lm.x - min(pose_x)) * WEIGHT_POSE)
                    feature_vector.append((lm.y - min(pose_y)) * WEIGHT_POSE)
            else:
                # 33 landmarks × 2 features each
                feature_vector.extend([0.0] * (33 * 2))

            data.append(feature_vector)
            labels.append(dir_)

# Save the extracted features and labels to a pickle file.
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
