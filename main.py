
import detection
import tracking

# PLAN:
# 1. Fix fargene til boksene slik at de stemmer med info gitt av final_detections
# 2. Få boxene til å faktisk tracke / følge sine respektive spillere
# 3. Utvid til egne bokser for dommere (det kan vell gjøres ved å vite at de alltid har samme farger som hverandre og er to stk)
# 4. Fiks så keeperne assosieres med riktige lag (hvordan though - må se på laget som flest har rygg mot keeper i.e forvsarer målet?)

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
    
    # Detection part:
    input_video_location = 'soccer.mp4'
    output_initial_frame_location = 'The_first_frame'                      # This is where the first frame will be saved
    raw_detections, annotated_image, final_detections = detection.full_detection(input_video_location, output_initial_frame_location) 

    # Feature extraction: (Can be added if want to, if needs extra help to track by seperating between features of players):
    # Does not seem neccesary just yet.

    # Tracking part:
    tracking.full_tracking(input_video_location, final_detections)
