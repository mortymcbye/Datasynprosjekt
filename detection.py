


# For split_video_into_frames
import time
import os

# For detect_obj_initial_frame:
from inference import get_model # import a utility function for loading Roboflow models
import supervision as sv # import supervision to visualize our results
import cv2 # import cv2 to helo load our image

# For identifying different teams:
import numpy as np
from sklearn.cluster import KMeans

def extract_dominant_color(image, boxes, n_colors=1):
    dominant_colors = []
    for box in boxes:
        # Convert the numpy array to a list and then to integers
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

# This function replaces 'seperate_teams_by_color' and uses the new dominant color extraction
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


def detect_obj_initial_frame(initial_frame_path): # Function that detects objects in frame and plots their boxes
    # define the image url to use for inference
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


def add_additional_detection_stats(raw_detections, image, team_assignment_array):
    """
    These are the detections:  
    Detections(xyxy=array([
       [2234.,  904., 2420., 1176.],
       [2855.,  851., 3028., 1129.],
       [1595.,  818., 1712., 1031.],
       [2412., 1333., 2549., 1674.],
       [1867., 1220., 2043., 1507.],
       [ 994., 1092., 1207., 1318.]
       ]), mask=None, confidence=array([0.89516556, 0.86010325, 0.85414046, 0.75840878, 0.72360027,
       0.7132448 ]), class_id=array([2, 2, 2, 2, 2, 2]), tracker_id=None, data={'class_name': array(['player', 'player', 'player', 'player', 'player', 'player'], 
      dtype='<U6')})
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
    return final_detections, image


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




if __name__ == '__main__':
    # Note that this implementation is vulnerable to jersies of similar colors. For instance light and dark blue

    # Plan:
    # Step 1 - 'extract_first_frame' function:
        # Input: mp4 video
        # Output: Location of first frame
        
        input_loc = 'soccer.mp4'                            # Replace with your actual video file name
        output_loc = 'The_first_frame'                    # This is where the first frame will be saved
        extract_first_frame(input_loc, output_loc)
        # first_frame_path = os.path.join(output_loc, "first_frame.jpg")

        first_frame_path = 'test.jpg'

    # Step 2 - 'detect_obj_initial_frame' function:
        # Input: location of first frame
        # Output: raw_detections (the boxes), annotated_image

        raw_detections, annotated_image = detect_obj_initial_frame(first_frame_path)

    # Step 3 - 'extract_dominant_color' function:
        # Input: annotated_image, raw_detections.xyxy, n_colors=1
        # Output: dominant_colors

        # Identifies dominant color within each box, as this is more accurate then average color that is vulnerable to shadows and light
        dominant_colors = extract_dominant_color(annotated_image, raw_detections.xyxy, n_colors=1)

    # Step 4 - 'assign_teams_by_dominant_color' function:
        # Input: dominant_colors
        # Output: team_assignment_array

        team_assignment_array = assign_teams_by_dominant_color(dominant_colors)
    
    # Step 5 - 'add_additional_detection_stats':
        # Input: raw_detections, annotated_image, team_assignment_array
        # Output: final_detections, annotated_image

        final_detections, annotated_image = add_additional_detection_stats(raw_detections, annotated_image, team_assignment_array)
        show_annotated_image(final_detections, annotated_image) # Just for troubleshooting. Don't really need to display image this early
