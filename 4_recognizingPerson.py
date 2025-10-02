import numpy as np
import imutils
import pickle
import time
import cv2
import os

# --- Configuration (Ensure these files/folders exist relative to execution path) ---
embeddingModel = "openface_nn4.small2.v1.t7"
embeddingFile = "output/embeddings.pickle"
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"
conf = 0.5 # Minimum confidence for face detection (not recognition)

# ML Engineer Control: Set maximum runtime in seconds for graceful shutdown
MAX_RUN_TIME_SECONDS = 10 # Changed to 5 seconds as requested.

# Recognition Goal: Stop the process immediately after a person is detected with this confidence.
TARGET_CONFIDENCE = 0.80 # 95% confidence required for a successful recognition and immediate stop

# Check for required model files before proceeding
if not os.path.exists(embeddingModel):
    print(f"ERROR: Embedding model file not found: {embeddingModel}")
    # You would typically download or provide this model here
if not os.path.exists(recognizerFile) or not os.path.exists(labelEncFile):
    print("ERROR: Recognition model or label files not found in 'output/' directory.")
    print("Please ensure you have trained the model and saved recognizer.pickle and le.pickle.")
# --- End Configuration Checks ---

print("[INFO] Loading face detector...")
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"

# Attempt to load detector, gracefully handle missing files
try:
    detector = cv2.dnn.readNetFromCaffe(prototxt, model)
except Exception as e:
    print(f"ERROR: Could not load Caffe detector model. Check paths to deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel. Details: {e}")
    exit()

print("[INFO] Loading face recognizer...")
try:
    embedder = cv2.dnn.readNetFromTorch(embeddingModel)
    recognizer = pickle.loads(open(recognizerFile, "rb").read())
    le = pickle.loads(open(labelEncFile, "rb").read())
except Exception as e:
    print(f"ERROR: Could not load Torch embedder or pickle files. Details: {e}")
    exit()

box = []
print(f"[INFO] Starting video stream... Running for a maximum of {MAX_RUN_TIME_SECONDS} seconds to detect a person.")
cam = cv2.VideoCapture(0)
time.sleep(2.0)

# Initialize timing variables and the exit flag
start_time = time.time()
frame_count = 0
fps_history = []
person_detected_and_confirmed = False # New flag for immediate exit
# Variables to store the final successful detection result
detected_person_name = "Unknown" 
detected_person_proba = 0.0

while (time.time() - start_time) < MAX_RUN_TIME_SECONDS:
    
    # Start timer for current frame processing
    frame_start_time = time.time()
    
    _, frame = cam.read()
    
    # Check if frame read was successful
    if frame is None:
        # print("[ERROR] Could not read frame from webcam. Stopping process.") # Suppressing inner loop error messages
        break
        
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    
    # Pre-process the frame for face detection
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Ensure the face ROI is large enough
            if fW < 20 or fH < 20:
                continue

            # Compute face embedding
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Perform classification
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            
            # # Console Output - Suppressed for silent operation
            # print(f"[RECOGNIZED] Detected '{name}' | Confidence: {proba * 100:.2f}% | Position: ({startX}, {startY})")
            
            # --- Single-Shot Exit Logic ---
            if proba >= TARGET_CONFIDENCE:
                # Store the successful result before breaking
                detected_person_name = name
                detected_person_proba = proba
                person_detected_and_confirmed = True
                # print(f"[SUCCESS] High-confidence detection achieved! Stopping process.") # Suppressed for silent operation
                break # Break out of the inner 'for' loop immediately
            # ------------------------------

    frame_count += 1
    
    # Check if the single-shot condition was met
    if person_detected_and_confirmed:
        break # Break out of the outer 'while' loop immediately

    # Calculate FPS for the frame (Efficiency Monitoring)
    frame_end_time = time.time()
    frame_time = frame_end_time - frame_start_time
    
    # Avoid division by zero
    if frame_time > 0:
        current_fps = 1.0 / frame_time
        fps_history.append(current_fps)
        
        # # Print status every 10 frames for cleaner output - Suppressed for silent operation
        # if frame_count % 10 == 0:
        #     avg_fps = np.mean(fps_history[-10:]) if fps_history else 0
        #     remaining_time = int(MAX_RUN_TIME_SECONDS - (frame_end_time - start_time))
        #     print(f"--- [STATUS] FPS: {avg_fps:.2f} | Frames Processed: {frame_count} | Time Remaining: {remaining_time}s ---")

# --- Cleanup ---
cam.release()
final_avg_fps = np.mean(fps_history) if fps_history else 0
total_run_time = time.time() - start_time

# --- Final Output: Only the required answer ---
if person_detected_and_confirmed:
    print(f"\n[FINAL RESULT] Person Identified: '{detected_person_name}' with {detected_person_proba * 100:.2f}% confidence.")
    print(f"[STATUS] Process successfully stopped after {total_run_time:.2f} seconds and {frame_count} frames.")
else:
    print(f"\n[FINAL RESULT] Identification Failed: No person was detected with a confidence of {TARGET_CONFIDENCE * 100:.0f}% within the {MAX_RUN_TIME_SECONDS} second limit.")
    print(f"[STATUS] Process stopped after {total_run_time:.2f} seconds.")
    
# print(f"[EFFICIENCY] Overall Average FPS: {final_avg_fps:.2f}") # Suppress general efficiency metrics
