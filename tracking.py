

import cv2
import numpy as np

def convert_bboxes(bboxes):
    # Convert from [[100., 50., 40., 70.][150., 55., 40., 70.]]
    converted = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        converted.append((int(x1), int(y1), int(w), int(h)))
    return converted

def full_tracking(input_video_location, final_detections):
    bboxes = final_detections.xyxy
    colors = final_detections.class_id
    player_ids = final_detections.data["class_name"]

    # Load the video
    cap = cv2.VideoCapture(input_video_location)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Convert bounding boxes and initialize trackers
    bboxes = convert_bboxes(bboxes)
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video file.")
        return

    trackers = []
    for bbox in bboxes:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        trackers.append(tracker)

    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break

        # Update and draw bounding boxes for each tracker
        for tracker, color, player_id in zip(trackers, colors, player_ids):
            success, box = tracker.update(frame)

            # For correct colors (seperating between teams)
            if color == 1: 
                color = (0, 0, 255)
            elif color == 2: # If you do not use 'elif' here, the 'else' statement WILL always be entered if not 'if' entered. Effectively overriding the 'elif'
                color = (255, 255, 255)
            else: # Color for referees
                color = (0, 0, 0)

            if success:
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                cv2.rectangle(frame, p1, p2, color, 2, 1)
                cv2.putText(frame, player_id, (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Handling tracking failure, if needed
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # Example usage
    """
    video_path = 'soccer.mp4'

    Detections(xyxy=array(
       [[ 440.,  531.,  471.,  596.],
       [ 478.,  168.,  491.,  205.]]), mask=None, confidence=array([0.8881253 , 0.87274611, 0.87224352, 0.86617452, 0.85796571,
       0.85334438, 0.84024572, 0.83696854, 0.8246336 , 0.82408929,
       0.82127774, 0.81580412, 0.81312847, 0.8106792 , 0.81011415,
       0.79962969, 0.7921868 , 0.79076731, 0.77907324, 0.77451396,
       0.64464688]), class_id=array([2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1]), tracker_id=None, data={'class_name': array(['1', '2', '3', '1', '2', '3', '4', '5', '4', '5', '6', '6', '7',
       '8', '7', '8', '9', '10', '9', '11', '12'], dtype='<U7')})

    full_tracking(input_video_location, final_detections)
    """
