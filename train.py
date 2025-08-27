import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import train_generator, test_generator
import pickle

# Define CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # Prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for stability
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(train_generator, validation_data=test_generator, epochs=10)

# Save Model
model.save("skin_cancer_model.h5")
print("✅ Model Training Complete & Saved!")

# Save training history
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Training History Saved!")
