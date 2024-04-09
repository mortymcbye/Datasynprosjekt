import numpy as np
import cv2

def initialize_multi_tracker():
    return cv2.legacy.MultiTracker_create()

def add_objects_to_tracker(multi_tracker, initial_frame, detections):
    for box in detections.xyxy:
        # Convert from (xmin, ymin, xmax, ymax) to (x, y, w, h) if necessary
        xmin, ymin, xmax, ymax = box
        w, h = xmax - xmin, ymax - ymin
        bbox = (int(xmin), int(ymin), int(w), int(h))
        tracker = cv2.legacy.TrackerCSRT_create()
        multi_tracker.add(tracker, initial_frame, bbox)

def track_objects(input_video_location, multi_tracker, labels):
    cap = cv2.VideoCapture(input_video_location)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        scaling_factor = 1.3
        display_frame = cv2.resize(frame, (0, 0), fx=scaling_factor, fy=scaling_factor)

        success, boxes = multi_tracker.update(frame)
        for i, box in enumerate(boxes):
            # Use the class_id or class_name to set the color
            label = labels[i]  # Assuming labels are extracted beforehand
            if label == '1':
                color = (0, 255, 0)  # Green for label '1'
            elif label == '2':
                color = (0, 0, 255)  # Blue for label '2'
            else:
                color = (255, 0, 0)  # Default to Red for other labels

            p1 = (int(box[0] * scaling_factor), int(box[1] * scaling_factor))
            p2 = (int((box[0] + box[2]) * scaling_factor), int((box[1] + box[3]) * scaling_factor))
            cv2.rectangle(display_frame, p1, p2, color, 2, 1)

        cv2.imshow('Tracking', display_frame)
        display_delay = 500
        if cv2.waitKey(display_delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def full_tracking(initial_frame_path, input_video_location, detections):
    initial_frame = cv2.imread(initial_frame_path)
    # Extract labels (e.g., class names) from detections for use in coloring
    labels = detections.data['class_name']  # Adjust based on your class naming

    multi_tracker = initialize_multi_tracker()
    add_objects_to_tracker(multi_tracker, initial_frame, detections)

    track_objects(input_video_location, multi_tracker, labels)

if __name__ == '__main__':
    initial_frame_path = 'first_frame.jpg'
    input_video_location = 'soccer.mp4'
    # Assume 'detections' is already defined as per your structure
    # full_tracking(initial_frame_path, input_video_location, detections)
