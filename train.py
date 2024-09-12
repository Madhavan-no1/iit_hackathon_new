# Import necessary libraries
import tensorflow as tf
from zipfile import ZipFile
import os, glob
import cv2
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, MaxPooling2D
from keras.layers import BatchNormalization, Flatten, GlobalAveragePooling2D
from keras.applications import vgg16
from keras.models import Model
import matplotlib.pyplot as plt

# Set the path to your downloaded dataset (adjust to your own directory)
dataset_path = r'C:\path\to\your\dataset'  # Set to your actual dataset path
yes_path = os.path.join(dataset_path, 'yes')  # Tumor images folder
no_path = os.path.join(dataset_path, 'no')    # Non-tumor images folder

# Load images from the dataset and preprocess them
X = []
y = []

# Load tumor images (Yes)
os.chdir(yes_path)
for i in tqdm(os.listdir()):
    img = cv2.imread(i)
    img = cv2.resize(img, (224, 224))
    X.append(img)
    y.append('Y')

# Load non-tumor images (No)
os.chdir(no_path)
for i in tqdm(os.listdir()):
    img = cv2.imread(i)
    img = cv2.resize(img, (224, 224))
    X.append(img)
    y.append('N')

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Display sample images
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(X[i])
    plt.axis('off')
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Encode the labels
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# Convert the training and testing sets to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Load the VGG16 model with pre-trained ImageNet weights
img_rows, img_cols = 224, 224
vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

# Freeze the VGG16 layers
for layer in vgg.layers:
    layer.trainable = False

# Define the top model (fully connected head)
def lw(bottom_model, num_classes):
    """Create the top model to stack on top of the base (VGG16)"""
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

# Stack the top model onto VGG16
num_classes = 2
FC_Head = lw(vgg, num_classes)
model = Model(inputs=vgg.input, outputs=FC_Head)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1, initial_epoch=0)

# Save the model after training
model.save('brain_tumor_detection_model.h5')
print('Model saved as brain_tumor_detection_model.h5')

# Plot the training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
