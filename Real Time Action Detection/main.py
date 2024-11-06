import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import colorsys

# Initialize mediapipe
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
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([lh, rh])


def prob_viz(res, actions, input_frame, colors, start_y):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, start_y + num * 40), (int(prob * 100), start_y + 30 + num * 40), colors[num],
                      -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}", (0, start_y + 25 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(hue, 0.8, 0.9))
        bgr = (rgb[2], rgb[1], rgb[0])
        colors.append(bgr)
    return colors


def main():
    # Initialize variables
    actions = np.array(['Add', 'Divide', 'Equal', 'Multiply', 'Subtract'])
    action_symbols = {
        'Add': '+',
        'Subtract': '-',
        'Multiply': '*',
        'Divide': '/',
        'Equal': '='
    }
    digits = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    # Detection variables
    sequence = []
    threshold = 0.9

    # Expression building
    current_expression = []
    detected_digit = None
    detected_operation = None
    result = None

    # Add state variable to track what we're looking for
    detection_state = "DIGIT"  # Start with digit detection

    # Load models
    try:
        operation_model = load_model('models/best_model.keras')
        digit_model = load_model('models/hand_digits.keras')
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)

                if detection_state == "DIGIT":
                    print("Looking for digit...")
                    # Digit Recognition
                    digit_input = np.expand_dims(keypoints, axis=0)
                    digit_res = digit_model.predict(digit_input, verbose=0)[0]

                    if np.max(digit_res) > threshold:
                        new_digit = digits[np.argmax(digit_res)]
                        if detected_digit != new_digit:
                            detected_digit = new_digit
                            current_expression.append(detected_digit)
                            print(f"Detected digit: {detected_digit}")
                            detection_state = "OPERATION"  # Switch to operation detection
                            sequence = []  # Reset operation sequence

                elif detection_state == "OPERATION":
                    print("Looking for operation...")
                    sequence.append(keypoints)
                    sequence = sequence[-20:]

                    if len(sequence) == 20:
                        sequence_np = np.array(sequence)
                        operation_input = np.expand_dims(sequence_np, axis=0)
                        operation_res = operation_model.predict(operation_input, verbose=0)[0]

                        if np.max(operation_res) > threshold:
                            new_operation = actions[np.argmax(operation_res)]
                            if detected_operation != new_operation:
                                detected_operation = new_operation
                                print(f"Detected operation: {detected_operation}")

                                if detected_operation == 'Equal':
                                    # Calculate result
                                    try:
                                        expression_str = ''.join(current_expression)
                                        result = eval(expression_str)
                                        current_expression.append('=')
                                        current_expression.append(str(result))
                                        print(f"Expression result: {result}")
                                        detection_state = "DIGIT"  # Reset to digit detection
                                        detected_digit = None
                                    except:
                                        result = "Error"
                                        print("Error evaluating expression")
                                else:
                                    # Add operation symbol to expression
                                    current_expression.append(action_symbols[detected_operation])
                                    detected_digit = None
                                    detection_state = "DIGIT"  # Switch back to digit detection
                                detected_operation = None

            # Show current detection state on screen
            cv2.rectangle(image, (0, 0), (640, 80), (245, 117, 16), -1)

            # Display expression and result
            expression_text = ' '.join(current_expression)
            cv2.putText(image, expression_text, (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Display current detection state
            cv2.putText(image, f"Detecting: {detection_state}", (3, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Sign Language Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()