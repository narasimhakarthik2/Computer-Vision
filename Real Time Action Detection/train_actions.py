import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
DATA_PATH = os.path.join('Data', 'operations')
actions = np.array(['Add'])
sequence_length = 20
MODEL_PATH = os.path.join('models')

# Create model directory if it doesn't exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


def load_data():
    """Load and preprocess the data"""
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            raise ValueError(f"Directory not found: {action_path}")

        for sequence in np.array(os.listdir(action_path)).astype(int):
            window = []
            for frame_num in range(sequence_length):
                frame_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                try:
                    res = np.load(frame_path)
                    # Take only the first set of keypoints if duplicated
                    if len(res.shape) > 1:  # If shape is (2, 126)
                        res = res[0]  # Take first set
                    window.append(res)
                except Exception as e:
                    print(f"Error loading file {frame_path}: {str(e)}")
                    raise
            sequences.append(window)
            labels.append(label_map[action])

    return np.array(sequences), np.array(labels)

def create_model(input_shape, num_classes):
    """Create the LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def main():
    # Load and preprocess data
    print("Loading data...")
    try:
        sequences, labels = load_data()
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)

        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

        # Create logs directory
        log_dir = os.path.join('Logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Callbacks
        tb_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )

        # Updated file extension to .keras
        checkpoint_path = os.path.join(MODEL_PATH, 'best_model.keras')
        checkpoint_callback = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=100,
            restore_best_weights=True,
            verbose=1
        )

        # Create and compile model
        print("Creating model...")
        model = create_model(input_shape=(sequence_length, 126), num_classes=actions.shape[0])

        model.compile(
            optimizer='Adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )

        # Model summary
        model.summary()

        # Train model
        print("Training model...")
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=2000,
            batch_size=32,
            callbacks=[tb_callback, checkpoint_callback, early_stopping]
        )

        # Evaluate model
        print("Evaluating model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy * 100:.2f}%")

        # Save final model with .keras extension
        final_model_path = os.path.join(MODEL_PATH, 'final_model.keras')
        model.save(final_model_path)
        print(f"Model saved to {final_model_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()