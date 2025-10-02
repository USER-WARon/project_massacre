from imutils import paths
import numpy as np
import pickle
import cv2
import os

# --- Configuration ---
dataset = "dataset"
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface_nn4.small2.v1.t7" # Model for embedding Pytorch

# NOTE ON EFFICIENCY: The Caffe face detector (deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel)
# is intentionally removed/bypassed here. Since the images in the 'dataset' directory 
# are already pre-cropped and centered on the face (from the robust data collection script),
# running face detection again would be redundant and drastically slow down the process.

# --- Initialization ---
print("[INFO] Loading PyTorch facial embedder model...")
# Extracting facial embeddings via deep learning feature extraction
try:
    embedder = cv2.dnn.readNetFromTorch(embeddingModel)
except Exception as e:
    print(f"ERROR: Could not load the OpenFace embedding model ({embeddingModel}). Details: {e}")
    exit()

print("[INFO] Gathering image paths...")
# Getting image paths from the dataset folder created by the robust data collector
imagePaths = list(paths.list_images(dataset))

# Initialization
knownEmbeddings = []
knownNames = []
total = 0
MIN_FACE_SIZE = 20 # Minimum required size for the face image

print(f"[INFO] Starting efficient embedding generation for {len(imagePaths)} images...")

# --- Processing Loop (Optimized for Pre-Cropped Images) ---
# We read images one by one and apply feature extraction directly.
for (i, imagePath) in enumerate(imagePaths):
    # Print status to track progress
    print("Processing image {}/{}...".format(i + 1, len(imagePaths)))
    
    # Extract the person's name from the folder name (e.g., 'dataset/Joe/00001_normal.png' -> 'Joe')
    name = imagePath.split(os.path.sep)[-2]
    
    # Read the image (which is already a cropped face)
    image = cv2.imread(imagePath)
    
    if image is None:
        print(f"Warning: Could not read image at {imagePath}. Skipping.")
        continue
    
    (fH, fW) = image.shape[:2]

    # Simple validation to ensure the image is a valid size
    if fW < MIN_FACE_SIZE or fH < MIN_FACE_SIZE:
        print(f"Warning: Image too small ({fW}x{fH}). Skipping.")
        continue

    # --- Pre-processing for Embedder (Most Efficient Step) ---
    # 1. The image is now treated as the Face ROI.
    # 2. It is directly converted to a blob for the OpenFace embedder (96x96 input)
    # 3. No intermediate detection or resizing steps are needed.
    faceBlob = cv2.dnn.blobFromImage(
        image, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
    )
    
    # Extract the 128-D facial embedding vector
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    
    # Store the results
    knownNames.append(name)
    knownEmbeddings.append(vec.flatten())
    total += 1

# --- Final Output ---
print("[INFO] Total embeddings generated: {0} ".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}

# Save the generated embeddings to disk
print("[INFO] Writing embeddings to disk...")
try:
    with open(embeddingFile, "wb") as f:
        f.write(pickle.dumps(data))
except Exception as e:
    print(f"ERROR: Could not write embedding file to {embeddingFile}. Details: {e}")

print("[SUCCESS] Process Completed")
