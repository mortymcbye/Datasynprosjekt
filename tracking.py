import cv2
import numpy as np
import argparse
import keypoint_detection

def convert_bboxes(bboxes):
    converted = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        converted.append((int(x1), int(y1), int(w), int(h)))
    return converted

def init():
    #Files needed
    parse=argparse.ArgumentParser()
    parse.add_argument('--weights', type=str, default='Weights_CFG/best.weights', help='weights path')
    parse.add_argument('--cfg', type=str, default='Weights_CFG/best.cfg', help='cfg path')
    parse.add_argument('--image', type=str, default='Images/test.jpg', help='image path')
    parse.add_argument('--video', type=str, default='Images/soccer.mp4', help='video path')
    parse.add_argument('--img_size', type=int, default='320', help='size of w*h')
    opt = parse.parse_args()
    obj = keypoint_detection.Yolov4(opt)  # constructor called and executed
    return obj, opt

def full_tracking(final_detections):
    #Necessities
    all_real_life_positions = []
    timestamps = []
    detect_keypoints = True   #Set to True to add keypoint detection as well

    #Keypoint detection model
    obj, opt = init()

    #Player detection
    bboxes = final_detections.xyxy
    colors = final_detections.class_id
    player_ids = final_detections.data["class_name"]

    # For writing results
    cap = cv2.VideoCapture(opt.video)
    fps = 30 #cv2.CAP_PROP_FPS
    width = cap.get(3)
    height = cap.get(4)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter("demo.avi", fourcc, fps, (int(width), int(height)))

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

    #Main loop for tracking
    while cap.isOpened():
        # Read a new frame and increase frame tracking
        ret, frame = cap.read()
        obj.frame_nbr += 1
        
        if ret == True:
            # Update bounding boxes for each tracker
            for tracker, color, player_id in zip(trackers, colors, player_ids):
                success, box = tracker.update(frame)

                #Draw bounding boxes with labels for each player
                # For correct colors (seperating between teams)
                if color == 1: 
                    color = (0, 0, 255)
                elif color == 2: 
                    color = (255, 255, 255)
                else: # Color for referees
                    color = (0, 0, 0)

                if success:
                    p1 = (int(box[0]), int(box[1]))
                    p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
                    cv2.rectangle(frame, p1, p2, color, 2, 1)
                    cv2.putText(frame, player_id, (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    # Handling tracking failure
                    cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            #Keypoint detection (if activated)
            if detect_keypoints:
                #Infer keypoint model
                outcome, coordinate_dict = obj.Inference(image=frame, original_width=width, original_height=height)
                if outcome is None:
                    #Writes frame with no detections to final video
                    output.write(frame)
                else:
                    player_coord = [i[:2] for i in final_detections.xyxy]   #Extracts x_1 and y_1 coordinate of players
                    real_life_points = obj.myTransformation(coordinate_dict, player_coord)  #Transforms coordinates to real life coordinates
                    #For velocity and accl. calculations at end of run
                    #Frame rate = 30 fps
                    timestamps.append(obj.frame_nbr/30)
                    all_real_life_positions.append(real_life_points)
                    #Writes results to video
                    output.write(outcome)

            # Display result
            cv2.imshow("Tracking", frame)

            # Exit if q pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    #Prints all visible and non-visible keypoints for each frame if detected
    if detect_keypoints:
        obj.keypointPrint()

    cap.release()
    cv2.destroyAllWindows()

    #Returns for use in velocity and accl. calculation
    return all_real_life_positions, timestamps


#Only for testing
if __name__ == '__main__':
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
