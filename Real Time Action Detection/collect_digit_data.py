import os
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )


def extract_keypoints(results):
    # Extract hand landmarks
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([lh, rh])


def collect_digit_data():
    # Path for exported data
    DATA_PATH = os.path.join('Data', 'digits')
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    # Actions/digits to detect
    digits = np.array(['3', '4', '5', '6', '7', '8', '9'])

    # Number of samples per digit
    no_samples = 300  # Increased samples for better training

    # Create folders for each digit
    for digit in digits:
        digit_path = os.path.join(DATA_PATH, digit)
        if not os.path.exists(digit_path):
            os.makedirs(digit_path)

    cap = cv2.VideoCapture(1)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for digit in digits:
            sample_count = 0

            while sample_count < no_samples:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Show instructions
                cv2.putText(image, f'Show digit: {digit}', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f'Samples: {sample_count}/{no_samples}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(image, 'Press "s" to save sample', (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.imshow('OpenCV Feed', image)

                key = cv2.waitKey(10)
                if key & 0xFF == ord('q'):
                    return
                elif key & 0xFF == ord('s') and (results.left_hand_landmarks or results.right_hand_landmarks):
                    # Save the keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, digit, f"{sample_count}.npy")
                    np.save(npy_path, keypoints)
                    sample_count += 1
                    print(f"Saved sample {sample_count} for digit {digit}")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    collect_digit_data()