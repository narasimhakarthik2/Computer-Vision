import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# Path for exported data
DATA_PATH = os.path.join('Data', 'digits')

# Load the data
data = []
labels = []

# Actions/digits
digits = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# Load data from numpy arrays
for digit in digits:
    digit_path = os.path.join(DATA_PATH, digit)
    for sample_path in os.listdir(digit_path):
        try:
            # Load keypoints
            sample_data = np.load(os.path.join(digit_path, sample_path))
            data.append(sample_data)
            labels.append(int(digit))
        except Exception as e:
            print(f"Error loading {sample_path}: {e}")
            continue

# Convert to numpy arrays
X = np.array(data)
y = to_categorical(labels).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and compile model
model = Sequential([
    Dense(128, activation='relu', input_shape=(126,)),  # 126 = 21 landmarks * 3 (x,y,z) * 2 hands
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')  # 10 digits
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# Callbacks
tb_callback = TensorBoard(log_dir=os.path.join('Logs'))
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Print model summary
model.summary()
print(f"\nTraining data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Train the model
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[tb_callback, early_stopping]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc*100:.2f}%")

# Save the model
model.save('models/hand_digits.keras')

# Optional: Print confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=digits, yticklabels=digits)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Print classification report
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=digits))