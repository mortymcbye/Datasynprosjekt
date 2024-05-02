import cv2
import numpy as np

def initialize_feature_detector():
    return cv2.SIFT_create()

def feature_extractor(frame, final_detections):
    bboxes = final_detections.xyxy  # Array of bounding boxes
    detector = initialize_feature_detector()
    features = []
    
    # Create player ids
    player_ids = range(len(bboxes)) 
    
    for bbox, player_id in zip(bboxes, player_ids):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi = frame[y1:y2, x1:x2]  # Define the region of interest based on the bounding box
        if roi.size == 0:
            continue  # Skip empty regions
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        
        # Store the player_id and descriptors
        if descriptors is not None:
            features.append([player_id, descriptors])
        else:
            features.append([player_id, np.array([])])  # Handle cases where no keypoints are detected

    return features

#Only for testing
if __name__ == '__main__':

    frame = cv2.imread('The_first_frame/first_frame.jpg')
    # Initializing a final_detections that has xyxy attribute of correct structure
    class FinalDetections:
        def __init__(self, xyxy):
            self.xyxy = np.array(xyxy)

    final_detections = FinalDetections([
        [440., 531., 471., 596.],
        [68., 321., 105., 378.],
        # Add more as needed
    ])

    # Example to load a frame (normally within a loop or from a video capture)
    features = feature_extractor(frame, final_detections)
    print("Extracted features from bounding boxes:")
    for feature in features:
        player_id, descriptors = feature
        print(f"Player ID: {player_id}, Number of Descriptors: {len(descriptors)}, Actual description: {descriptors}")
