import os
import numpy as np
import cv2
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog, local_binary_pattern
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import tensorflow as tf

# ---------------------------
# 1. Load the LFW Dataset
# ---------------------------
# Set the data_home path to use the local dataset directory "D:\MLAssignment"
DATA_PATH = r"D:\MLAssignment"

# Load LFW with a minimum of 70 images per person
lfw = fetch_lfw_people(data_home=DATA_PATH, min_faces_per_person=70, resize=0.5)
images = lfw.images   # Grayscale images; shape: [n_samples, height, width]
labels = lfw.target
label_names = lfw.target_names

print("Dataset loaded:")
print("Number of samples:", images.shape[0])
print("Image shape (before resizing):", images[0].shape)
print("Classes:", label_names)

# ---------------------------
# 2. Preprocessing
# ---------------------------
TARGET_SIZE = 64  # Define the target size for resizing

def preprocess_images(imgs, target_size=TARGET_SIZE):
    """
    Resize images to target_size x target_size and normalize pixel values to [0,1].
    """
    preprocessed = []
    for img in imgs:
        # Resize image
        resized = cv2.resize(img, (target_size, target_size))
        # Normalize pixel values
        norm = resized / 255.0
        preprocessed.append(norm)
    return np.array(preprocessed)

# Preprocess the images
images_proc = preprocess_images(images, TARGET_SIZE)
print("Preprocessed images shape:", images_proc.shape)

# ---------------------------
# 3. Traditional Feature Extraction
# ---------------------------

# 3.1 HOG Feature Extraction
def extract_hog_features(imgs):
    features = []
    for img in imgs:
        # Compute HOG features. You can adjust parameters as needed.
        hog_feat = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        features.append(hog_feat)
    return np.array(features)

hog_features = extract_hog_features(images_proc)
print("HOG features shape:", hog_features.shape)

# 3.2 LBP Feature Extraction
def extract_lbp_features(imgs, P=8, R=1):
    features = []
    for img in imgs:
        # Compute LBP using 'uniform' method.
        lbp = local_binary_pattern(img, P, R, method="uniform")
        # Build histogram for LBP values.
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=True)
        features.append(hist)
    return np.array(features)

lbp_features = extract_lbp_features(images_proc)
print("LBP features shape:", lbp_features.shape)

# 3.3 Edge Detection Feature Extraction (Canny)
def extract_edge_features(imgs):
    features = []
    for img in imgs:
        # Convert normalized image back to 8-bit format for Canny edge detection
        img_uint8 = (img * 255).astype('uint8')
        edges = cv2.Canny(img_uint8, 100, 200)
        # Flatten the edge map to form a feature vector
        features.append(edges.flatten())
    return np.array(features)

edge_features = extract_edge_features(images_proc)
print("Edge features shape:", edge_features.shape)

# ---------------------------
# 4. Deep Learning-based Feature Extraction using VGG16
# ---------------------------
# VGG16 expects 3-channel images, so we need to convert our grayscale images.
def convert_to_rgb(imgs):
    # Duplicate the single grayscale channel three times.
    imgs_rgb = np.stack([imgs, imgs, imgs], axis=-1)
    return imgs_rgb

images_rgb = convert_to_rgb(images_proc)
print("RGB images shape for VGG16:", images_rgb.shape)

# Load pre-trained VGG16 model (without the top classification layers)
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE, TARGET_SIZE, 3))
# Create a model to output the deep features from the VGG16 base
vgg_model = Model(inputs=vgg_base.input, outputs=vgg_base.output)

def extract_deep_features(model, imgs):
    features = model.predict(imgs, batch_size=32)
    # Flatten the feature maps to create feature vectors
    features_flat = features.reshape(features.shape[0], -1)
    return features_flat

deep_features = extract_deep_features(vgg_model, images_rgb)
print("Deep features shape:", deep_features.shape)

# ---------------------------
# 5. Classifier Training and Evaluation
# ---------------------------
def train_and_evaluate(features, labels, method_name):
    print(f"\n=== Training classifier on {method_name} features ===")
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    # Initialize Random Forest classifier (or substitute with Logistic Regression, KNN, Decision Trees, etc.)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    # Predict on the test set
    y_pred = clf.predict(X_test)
    # Evaluate classifier performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on {method_name} features: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

# Evaluate each feature extraction method

# Traditional Methods
train_and_evaluate(hog_features, labels, "HOG")
train_and_evaluate(lbp_features, labels, "LBP")
train_and_evaluate(edge_features, labels, "Edge Detection")

# Deep Learning-based Method
train_and_evaluate(deep_features, labels, "VGG16 Deep")