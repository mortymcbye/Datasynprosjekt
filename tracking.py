

# Should first make it work on trakcing between two spesific frames that are both annoted by detection.py and feature extraccted.

import numpy as np
import time
import os
import cv2 # import cv2 to helo load our image

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


def split_video_into_frames():
    """Function to extract frames from input video file
    and save them as separate frames in an output directory."""
    # Ensure the output directory exists within the current directory
    os.makedirs('frames_representing_entire_video', exist_ok=True)
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture('mall.mp4')
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Number of frames:", video_length)
    count = 0
    print("Converting video...\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no frame is returned
        # Write the results back to output location
        frame_filename = os.path.join('frames_representing_entire_video', f"{count+1:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        count += 1
    # Release the feed
    cap.release()
    # Print stats
    time_end = time.time()
    print(f"Done extracting frames.\n{count} frames extracted")


def track_obj_after_initial_frame():
    print("Hei")



if __name__ == '__main__':

    print("HEI")
    # Step 1 - 'split_video_into_frames' function:
        # Input: mp4 video
        # Output: Location of all frames
        
        # split_video_into_frames()

    # Step 2 - Initialize deepsort:
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    model_filename = 'model_data/mars-small128.pb'  # Path to the feature extractor model
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    total_number_of_frames = 10 # Må i realiteten hentes/beregnes etter vi faktisk har splitta inn i frames
    for each_frame in range (total_number_of_frames):
        # Generate detections for each detection in the format required by Deep SORT
        features = encoder(each_frame, boxes)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]

        # Run non-maxima suppression to remove detections that overlap too much
        boxes = np.array([d.detection for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)



    # Step 2 - 'track_obj_after_initial_frame' function:
        # Input: Sequence of frames (output of split_video_into_frames), annotaed initial frame (output from detect_obj_initial_frame)
        # Output: Annoted series of frames (i.e takes annoted first frame, and un-annoted rest of frames and uses first to annote rest)
        # detect_obj_initial_frame()

        # Tips: For tracking objects across frames use a tracking algorithm such as SORT or DeepSort
    
    # Step 3 - 're_assemble_video' function:
        # Input: Sequence of frames (output of track_obj_after_initial_frame)
        # Output: (Annotede) mp4 video
    
    # Step 4 - 'evaluate_tracking' function:
        # Input: Annoted mp4 video, original (un-annoted) mp4 video
        # Output: Accuracy value, precision value

        # Tips: Use MOTA for accuracy and MOTP for precision
