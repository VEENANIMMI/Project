import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Assuming you have a dataset with csv consisting of patient_id, series_id, instance_number, injury_name
csv_path = './image_level_labels.csv'

# Load CSV
train_data = pd.read_csv(csv_path)
print(train_data)

# Label encode the 'injury_name' column
label_encoder = LabelEncoder()
train_data['injury_label'] = label_encoder.fit_transform(train_data['injury_name'])

# Drop unnecessary columns
features_to_drop = ['patient_id', 'injury_name']  # Adjust based on your dataset
train_features = train_data.drop(features_to_drop, axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_features, train_data['injury_label'], test_size=0.2, random_state=42
)

# Define your neural network architecture for structured data
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

