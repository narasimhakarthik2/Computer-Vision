import os
import cv2
import numpy as np
from main import extract_keypoints, mediapipe_detection, draw_styled_landmarks, mp_holistic
import mediapipe as mp


def create_directories(data_path, actions, start_folder, no_sequences):
    """Create necessary directories for data collection"""
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for action in actions:
        action_path = os.path.join(data_path, action)

        # Create action directory
        if not os.path.exists(action_path):
            os.makedirs(action_path)

        # Create sequence directories
        for sequence in range(start_folder, start_folder + no_sequences):
            sequence_path = os.path.join(action_path, str(sequence))
            if not os.path.exists(sequence_path):
                os.makedirs(sequence_path)


def collect_data():
    # Initialize variables
    DATA_PATH = os.path.join('Data/operations')
    actions = np.array(['Add'])
    no_sequences = 60
    sequence_length = 20
    start_folder = 0

    # Create required directories
    create_directories(DATA_PATH, actions, start_folder, no_sequences)

    # Initialize webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise ValueError("Could not open webcam")

    def countdown_timer(image, timer_counter):
        """Display countdown timer on image"""
        # Position the timer in the center
        org = (int(image.shape[1] / 2) - 50, int(image.shape[0] / 2))

        if timer_counter > 0:
            cv2.putText(image, str(timer_counter), org,
                        cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 0, 0), 4, cv2.LINE_AA)
        else:
            # Show 'GO!' when counter is 0
            cv2.putText(image, 'GO!', org,
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4, cv2.LINE_AA)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Loop through actions
        for action in actions:
            # Loop through sequences
            for sequence in range(start_folder, start_folder + no_sequences):
                # Countdown before starting each sequence
                for timer_counter in range(3, -1, -1):  # 3,2,1,GO!
                    ret, frame = cap.read()
                    if not ret:
                        raise ValueError("Could not read frame")

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    # Display countdown
                    countdown_timer(image, timer_counter)

                    # Display sequence info
                    cv2.putText(image, f'Preparing for {action} Sequence {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1000)  # Wait for 1 second between countdown numbers

                # Loop through sequence length
                for frame_num in range(sequence_length):
                    # Read feed
                    ret, frame = cap.read()
                    if not ret:
                        raise ValueError("Could not read frame")

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    # Display collection status
                    if frame_num == 0:
                        cv2.putText(image, 'COLLECTING FRAMES', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    # Add progress bar
                    progress = int((frame_num / sequence_length) * 640)
                    cv2.rectangle(image, (0, 400), (progress, 420), (0, 255, 0), -1)

                    cv2.imshow('OpenCV Feed', image)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    try:
                        np.save(npy_path, keypoints)
                    except Exception as e:
                        print(f"Error saving file {npy_path}: {str(e)}")
                        continue

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        return

                # Short pause between sequences
                ret, frame = cap.read()
                if ret:
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    cv2.putText(image, 'SEQUENCE COMPLETE', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)  # 2 second pause between sequences

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        collect_data()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()