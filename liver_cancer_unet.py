import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Paths to your dataset
train_images_path = r'C:\Users\91741\Downloads\Implement\train_images\train_images'
train_masks_path = r'C:\Users\91741\Downloads\Implement\train_masks\train_masks'

def load_data(images_path, masks_path, limit=100):
    images = []
    masks = []
    image_files = sorted(os.listdir(images_path))[:limit]  # Limit the number of images
    mask_files = sorted(os.listdir(masks_path))[:limit]

    for image_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(images_path, image_file)
        mask_path = os.path.join(masks_path, mask_file)
        
        # Read and process the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))  # Resize to a smaller size for faster training
        images.append(img)
        
        # Read and process the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128, 128))  # Resize masks to 128x128
        masks.append(mask)

    images = np.array(images) / 255.0  # Normalize images
    masks = np.array(masks) / 255.0  # Normalize masks
    masks = np.expand_dims(masks, axis=-1)  # Add channel dimension

    return np.array(images), np.array(masks)

# Load a smaller dataset
images, masks = load_data(train_images_path, train_masks_path, limit=100)

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# U-Net model
def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    # Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Bottleneck
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    
    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv3)
    concat1 = concatenate([up1, conv2], axis=-1)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(concat1)
    
    up2 = UpSampling2D(size=(2, 2))(conv4)
    concat2 = concatenate([up2, conv1], axis=-1)
    conv5 = Conv2D(32, 3, activation='relu', padding='same')(concat2)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(conv5)
    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the U-Net model
model = unet_model()

# Early stopping to stop training if it takes too long
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,  # You can limit to fewer epochs if needed
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1
)

# Save the model
model.save('liver_cancer_unet.h5')
