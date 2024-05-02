import time
import os
from inference import get_model # Utility function for loading Roboflow models
import supervision as sv
import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_color(image, boxes, n_colors=1):
    dominant_colors = []
    for box in boxes:
        #Extract single coordinates
        x1, y1, x2, y2 = map(int, box.tolist())

        # Extract the region of interest (ROI) from the image using the bounding box
        roi = image[y1:y2, x1:x2]

        # Reshape the ROI to a 2D array where each row is a pixel's color
        roi = roi.reshape(-1, 3)

        # Apply KMeans clustering to find the dominant color in the ROI
        clt = KMeans(n_clusters=n_colors)
        clt.fit(roi)

        # The first center of the clusters will be considered the dominant color
        dominant_color = clt.cluster_centers_[0]
        dominant_colors.append(dominant_color)

    return dominant_colors


def assign_teams_by_dominant_color(dominant_colors):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(dominant_colors)
    labels = kmeans.labels_
    team_assignment_array = labels + 1
    
    return team_assignment_array

def separate_teams_by_dominant_color(image, detections):
    boxes = detections.xyxy  # Assuming this is a numpy array of shape (N, 4)
    dominant_colors = extract_dominant_color(image, boxes, n_colors=1)
    team_assignment_array = assign_teams_by_dominant_color(dominant_colors)
    return team_assignment_array


def extract_first_frame(input_loc, output_loc):
    """Extracts the first frame from the input video file and saves it as a JPEG in the output directory."""
    # Ensure the output directory exists within the current directory
    os.makedirs(output_loc, exist_ok=True)
    # Start capturing the video feed
    cap = cv2.VideoCapture(input_loc)
    # Read the first frame
    ret, frame = cap.read()
    if ret:
        # Construct the filename for the first frame
        first_frame_filename = os.path.join(output_loc, "first_frame.jpg")
        # Save the first frame
        cv2.imwrite(first_frame_filename, frame)
        print("First frame extracted and saved.")
    else:
        print("Error: Unable to read the first frame from the video.")
    # Release the video capture object
    cap.release()
    return first_frame_filename


def detect_obj_initial_frame(initial_frame_path): 
    """Detects objects in frame and plots their boxes"""
    #define the image url to use for inference
    image_file = initial_frame_path
    image = cv2.imread(image_file)

    # load a pre-trained yolov8n model
    model = get_model(model_id="football-players-detection-3zvbc/8", api_key="T76AhqZoC7nuDHNMGLyV")

    # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
    results = model.infer(image)

    # load the results into the supervision Detections api:
    raw_detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

    return raw_detections, image

def seperate_teams_by_color(image, detections):
    """
    Calculate the average RGB color inside each detection box and categorize them into two teams.
    
    :param image: The image from which detections were made, as a numpy array.
    :param detections: The detections object with bounding box information.
    :return: An array indicating team 1 or 2 for each detection.
    """
    # Extract bounding box coordinates
    boxes = detections.xyxy  # Assuming this is a numpy array of shape (N, 4)
    
    # Placeholder for average colors
    avg_colors = []
    
    # Calculate average color for each box
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        roi = image[y1:y2, x1:x2]
        avg_color = np.mean(roi.reshape(-1, 3), axis=0)
        avg_colors.append(avg_color)
    
    # Convert list of averages to a suitable format for KMeans
    avg_colors = np.array(avg_colors)
    
    # Use KMeans to categorize into two teams based on average color
    kmeans = KMeans(n_clusters=2, random_state=0).fit(avg_colors)
    labels = kmeans.labels_
    
    # Adjust labels to match your desired output (1 or 2 instead of 0 or 1)
    team_assignment_array = labels + 1
    
    return team_assignment_array


def add_additional_detection_stats(raw_detections, team_assignment_array):
    """
    These are the detections:  
Detections(xyxy=array([[ 440.,  531.,  471.,  596.],
       [  68.,  321.,  105.,  378.],
       [ 672.,  576.,  702.,  640.],
       [ 681.,  274.,  699.,  318.],
       [1081.,  362., 1101.,  413.],
       [ 727.,  476.,  749.,  532.],
       [ 109.,   94.,  127.,  113.],
       [ 942.,  254.,  961.,  295.],
       [ 140.,  422.,  163.,  479.],
       [ 641.,  146.,  656.,  180.],
       [1225.,  443., 1251.,  512.],
       [ 908.,  457.,  931.,  521.],
       [ 976.,  483., 1002.,  543.],
       [ 851.,  191.,  873.,  228.],
       [ 161.,  201.,  175.,  241.],
       [ 462.,  374.,  484.,  428.],
       [ 783.,  336.,  799.,  386.],
       [ 752.,  249.,  768.,  290.],
       [ 947.,  219.,  962.,  255.],
       [ 189.,  371.,  208.,  422.],
       [ 478.,  168.,  491.,  205.]]), mask=None, confidence=array([0.8881253 , 0.87274611, 0.87224352, 0.86617452, 0.85796571,
       0.85334438, 0.84024572, 0.83696854, 0.8246336 , 0.82408929,
       0.82127774, 0.81580412, 0.81312847, 0.8106792 , 0.81011415,
       0.79962969, 0.7921868 , 0.79076731, 0.77907324, 0.77451396,
       0.64464688]), class_id=array([2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1]), tracker_id=None, data={'class_name': array(['1', '2', '3', '1', '2', '3', '4', '5', '4', '5', '6', '6', '7',
       '8', '7', '8', '9', '10', '9', '11', '12'], dtype='<U7')})
    """

    # class_id (attribute) of raw_detections determine box color. Same id has same color (seperates betweem teams)
    # tracker_id (attribute) of raw_detections is currently un-used. But i think is for their jersey number
    # data (attribute) of raw_detections hold their names, here we can assign number to the players

    # Assign correct class_id (to seperate between teams and box-colors):
    player_team_one = 1
    player_team_two = 1
    for i, class_id in enumerate(team_assignment_array):
        # Assign correct team for each player:
        raw_detections.class_id[i] = class_id
        # Assign a reasonable number for each player within each team:
        # Note that it has to be 'class_name' since thats the attribute name within the Detections object.
        if class_id == 1:
            raw_detections.data['class_name'][i] = player_team_one
            player_team_one += 1
        else:
            raw_detections.data['class_name'][i] = player_team_two
            player_team_two += 1      

    final_detections = raw_detections
    return final_detections


def show_annotated_image(raw_detections, image):

    # create supervision annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=raw_detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=raw_detections)
    
    sv.plot_image(annotated_image)

def full_detection(input_video_location, output_initial_frame_location):
    # Note that this implementation is vulnerable to jersies of similar colors. For instance light and dark blue
    # Step 1 - 'extract_first_frame' function:
        # Input: mp4 video
        # Output: Location of first frame

        first_frame_filename = extract_first_frame(input_video_location, output_initial_frame_location)


    # Step 2 - 'detect_obj_initial_frame' function:
        # Input: location of first frame
        # Output: raw_detections (the boxes), annotated_image

        raw_detections, annotated_image = detect_obj_initial_frame(first_frame_filename)

    # Step 3 - 'extract_dominant_color' function:
        # Input: annotated_image, raw_detections.xyxy, n_colors=1
        # Output: dominant_colors

        # Identifies dominant color within each box, as this is more accurate than average color that is vulnerable to shadows and light
        dominant_colors = extract_dominant_color(annotated_image, raw_detections.xyxy, n_colors=1)

    # Step 4 - 'assign_teams_by_dominant_color' function:
        # Input: dominant_colors
        # Output: team_assignment_array

        team_assignment_array = assign_teams_by_dominant_color(dominant_colors)
    
    # Step 5 - 'add_additional_detection_stats':
        # Input: raw_detections, annotated_image, team_assignment_array
        # Output: final_detections, annotated_image
        final_detections = add_additional_detection_stats(raw_detections, team_assignment_array)
        #show_annotated_image(final_detections, annotated_image) # Just for troubleshooting. Don't really need to display image this early

        return final_detections



#Only for testing
if __name__ == '__main__':
    input_video_location = 'soccer.mp4'
    output_initial_frame_location = 'The_first_frame'                      # This is where the first frame will be saved
    raw_detections, annotated_image, final_detections = full_detection(input_video_location, output_initial_frame_location)   

    print(final_detections.xyxy[0])


