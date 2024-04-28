


"""
Main idea is to translate player position ON SCREEN into REAL-WORLD field measurments:
1.  Calibrate camera view against fields physical dimmensions
    - Decide keypoints (i think corner of the court, corner of keeperfelt and dot in the senter of the court should be sufficient and smart)
    - Label each frame's keypoints as visible/not-visible
    - Use visible keypoints to calibrate camera view and ready for analysis (at least three keypoints must be visible)
"""

import cv2
import numpy as np


######## Preparing for easier keypoint detection ########


# Save time and effort by identifying the field anc creating a mask that covers it:
def create_field_mask(image):
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of green color in HSV
    lower_green = np.array([30, 40, 40])    # These values might need adjustment
    upper_green = np.array([70, 255, 255])  # These values might need adjustment

    # Create a mask that captures areas of the field in green
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Optional: Dilate the mask to fill in gaps, then invert the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# Now we an invert this mask and color everything BUT the insides of this mask black to disregard it when looking for keypoints:
def disregard_non_field(image, mask):
    # Apply the inverted mask to the image, turning non-field areas black
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image


# We also want to disregard the players:
def disregard_players(image, final_detections):
    # Ensure that coordinates are integers
    final_detections = final_detections.astype(int)
    
    # Draw black rectangles over the bounding boxes
    for (x_min, y_min, x_max, y_max) in final_detections:
        # Fill the bounding box with black color
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)

    # Convert image to grayscale
    filtered_gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return filtered_gray_image


######## Detect keypoints ########    


import cv2
import numpy as np

def detect_field_keypoints(image):
    # Detect corners using the Harris corner detection algorithm
    dst = cv2.cornerHarris(image, 2, 3, 0.04)

    # Dilate the corners to make them more prominent
    dst = cv2.dilate(dst, None)

    # Threshold the image to highlight the corners
    keypoints = np.argwhere(dst > 0.01 * dst.max())

    # Convert keypoints to (x, y) coordinates
    keypoints_coordinates = [(point[1], point[0]) for point in keypoints]

    # Draw keypoints on the original image
    keypoints_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x, y in keypoints_coordinates:
        cv2.circle(keypoints_image, (x, y), 3, (0, 0, 255), -1)  # Draw a red circle

    return keypoints_coordinates, keypoints_image


# We ideally want to use a pre-trained model for this identifying:
# Seems like functionality is missing?
def identify_keypoints(keypoints_coordinates):
    # Pre-trained model identification part:
    real_world_coordinates = [] # Shall hold the real_world coordinates that are returned in the end

    # Information on real_world metrics of a standard soccer field:
    field_width, field_height = 105.0, 68.0

    # Corners of the Field
    top_left_corner = (0, 0)
    top_right_corner = (field_width, 0)
    bottom_left_corner = (0, field_height)
    bottom_right_corner = (field_width, field_height)
    
    # Center Dot
    center_dot = (field_width / 2, field_height / 2)
    
    # Penalty Spots (Assuming 10.97 meters from the goal line)
    penalty_spot_left = (field_width / 2 - 10.97, 0)
    penalty_spot_right = (field_width / 2 + 10.97, 0)
    
    # Midfield Circle (Assuming radius of 9.15 meters)
    midfield_circle_center = center_dot
    midfield_circle_radius = 9.15
    
    # Center Line
    center_line_start = (field_width / 2, 0)
    center_line_end = (field_width / 2, field_height)
    
    # Corner Flags (Assuming corner flag positions are at the corners themselves)
    top_left_corner_flag = top_left_corner
    top_right_corner_flag = top_right_corner
    bottom_left_corner_flag = bottom_left_corner
    bottom_right_corner_flag = bottom_right_corner

    return real_world_coordinates # This should return the detected keypoint's real_life corespondants


######## Translate keypoints into real_world ########


def translate_into_real_world(final_detections, timestamps_of_frames, keypoints_coordinates, real_world_keypoints_coordinates):

    final_detections_in_real_world = []

    # Calculate keypoints scale once if all detections share the same keypoints
    keypoints_scale = {}
    for keypoint_name, keypoint_real_coordinates in real_world_keypoints_coordinates.items():
        if keypoint_name in keypoints_coordinates:
            keypoint_image_coordinates = keypoints_coordinates[keypoint_name]
            scale_x = keypoint_real_coordinates[0] / keypoint_image_coordinates[0]
            scale_y = keypoint_real_coordinates[1] / keypoint_image_coordinates[1]
            keypoints_scale[keypoint_name] = (scale_x, scale_y)
        else:
            print(f"Keypoint '{keypoint_name}' not found in frame keypoints.")

    for detection in final_detections:
        x_min, y_min, x_max, y_max = detection
        player_center = ((x_max - x_min) / 2 + x_min, (y_max - y_min) / 2 + y_min)

        real_world_position = None

        for keypoint_name, keypoint_real_coordinates in real_world_keypoints_coordinates.items():
            if keypoint_name in keypoints_coordinates:
                keypoint_image_coordinates = keypoints_coordinates[keypoint_name]

                if keypoint_name in keypoints_scale:
                    scale_x, scale_y = keypoints_scale[keypoint_name]
                    distance_x = player_center[0] - keypoint_image_coordinates[0]
                    distance_y = player_center[1] - keypoint_image_coordinates[1]
                    real_world_x = keypoint_real_coordinates[0] + distance_x * scale_x
                    real_world_y = keypoint_real_coordinates[1] + distance_y * scale_y
                    real_world_position = (real_world_x, real_world_y)
                    break
                else:
                    print(f"Scaling information for keypoint '{keypoint_name}' not found.")
            else:
                print(f"Keypoint '{keypoint_name}' not found in frame keypoints.")

        if real_world_position:
            final_detections_in_real_world.append(real_world_position)

    return final_detections_in_real_world, timestamps_of_frames




######## Komprimert full-function ########


def full_keypoint_detection(original_image, final_detections):
    # First, create an inverted mask for the field
    inverted_field_mask = create_field_mask(original_image)

    # Then disregard everything not in the field
    original_image = disregard_non_field(original_image, inverted_field_mask)

    # Now disregard the players
    filtered_gray_image = disregard_players(original_image.copy(), final_detections)

    # Detect keypoints on the grayscale:
    keypoints_coordinates, keypoints_image = detect_field_keypoints(filtered_gray_image)
    
    # Display the original image and keypoints image - only for debugging purposes:
    cv2.imshow('Key Points Detected', keypoints_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # identify which keypoints we have found and return their corrosponding real_world_coordinates:
    """ As for now - this does not actually identify as we want to use a pre-trained model for this """
    #keypoints_coordinates should be equal to the data from our model
    real_world_keypoints_coordinates = identify_keypoints(keypoints_coordinates)

    # Use info on real_world and detected keypoint coordinates to translate final_detections into real_world:
    real_world_final_detections, timestamps_of_frames = translate_into_real_world(final_detections, timestamps_of_frames, keypoints_coordinates, real_world_keypoints_coordinates)

    return real_world_final_detections, timestamps_of_frames

        
    



if __name__ == '__main__':
    original_image_path = 'Images/test.jpg'
    original_image = cv2.imread(original_image_path)

    final_detections = np.array([
        [440, 531, 471, 596],
        [68, 321, 105, 378],
        [672, 576, 702, 640],
        [681, 274, 699, 318],
        [1081, 362, 1101, 413],
        [727, 476, 749, 532],
        [109, 94, 127, 113],
        [942, 254, 961, 295],
        [140, 422, 163, 479],
        [641, 146, 656, 180],
        [1225, 443, 1251, 512],
        [908, 457, 931, 521],
        [976, 483, 1002, 543],
        [851, 191, 873, 228],
        [161, 201, 175, 241],
        [462, 374, 484, 428],
        [783, 336, 799, 386],
        [752, 249, 768, 290],
        [947, 219, 962, 255],
        [189, 371, 208, 422],
        [478, 168, 491, 205]
    ])

    # Run the full function for testing:
    real_world_final_detections, timestamps_of_frames = full_keypoint_detection(original_image, final_detections)

    # Print the results:
    print("Real-world positions:")
    for pos in real_world_final_detections:
        print(pos)
