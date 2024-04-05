
# The overall steps are as follows:
"""
    1. Detection: Run an object detection model on each frame of the video to detect players. This will give you a list of bounding boxes for each frame.
    2. Feature Extraction: For every detected player (bounding box) in each frame, you extract a feature vector using a neural network. This vector represents the appearance of the player and is used to distinguish between different players.
    3. Tracking with Deep SORT:
        - Prediction: For each new frame, Deep SORT first predicts new bounding box locations for all trackers (representing players) based on their last known state and a motion model.
        Association: The predicted locations and new detections are matched based on a similarity measure. This measure combines motion information (how much the location has changed) and appearance information (the feature vectors from step 2) to match detections to existing trackers.
        Update: After association, the tracker updates its state with the new detection. If there's no matching detection (e.g., a player is occluded), the tracker's state is updated based on the motion model alone.
        Creation and Deletion of Trackers: If a detection has no corresponding tracker, a new tracker is created. If a tracker doesn't receive any matching detections for a certain number of frames, it's deleted.
        Output Generation: During or after tracking, generate the output. This usually involves drawing bounding boxes and identifiers (like an ID number) on each player in the video frames.
        - Reassembly: If you processed frames individually, reassemble the video from the annotated frames to create a final video output.
        - Evaluation: If you have ground truth data (the real paths of players), compare your tracking results to this data to calculate performance metrics like MOTA (Multiple Object Tracking Accuracy) and MOTP (Multiple Object Tracking Precision).

"""

# Current: Complete the feature_extraction.py:

if __name__ == '__main__':
    # Ideally we should be able to run the entire program ery seperated like so..

    """
    detection.main()
    feature_extraction.main()
    tracking.main()
    
    """
