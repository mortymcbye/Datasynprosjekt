


"""
Main idea is to translate player position ON SCREEN into REAL-WORLD field measurments:
1.  Calibrate camera view against fields physical dimmensions
    - Decide keypoints (i think corner of the court, corner of keeperfelt and dot in the senter of the court should be sufficient and smart)
    - Label each frame's keypoints as visible/not-visible
    - Use visible keypoints to calibrate camera view and ready for analysis (at least three keypoints must be visible)
"""

import cv2
import numpy as np


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

    
# Using Shi-Tomasi corner detection:
def detect_keypoints(gray_image, original_image):
    # Detect corners
    corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=35, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        corners = np.int0(corners)
        # Draw corners on the original colored image in red
        for i in corners:
            x, y = i.ravel()
            cv2.circle(original_image, (x, y), 5, (255, 0, 0), -1)  # Blue color ans size (5) for the keypoint-dots
    
    # Display the original image with keypoints in red
    cv2.imshow('Keypoints', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corners


def translate_into_real_world(keypoints):
    # Logic to translate the keypoints into real-world coordinates:
        # I really dont know how to do this yet..

    real_world_positions, timestamps_of_frames = 1  # Placeholder values
    return real_world_positions, timestamps_of_frames


def full_keypoint_detection(original_image, final_detections):
    # First, create an inverted mask for the field
    inverted_field_mask = create_field_mask(original_image)

    # Then disregard everything not in the field
    original_image = disregard_non_field(original_image, inverted_field_mask)

    # Now disregard the players
    filtered_gray_image = disregard_players(original_image.copy(), final_detections)

    # Detect keypoints on the grayscale image but draw them on the original image
    keypoints = detect_keypoints(filtered_gray_image, original_image)

    # Translate keypoints into real-world positions
    real_world_positions, timestamps_of_frames = translate_into_real_world(keypoints)

    return real_world_positions, timestamps_of_frames



if __name__ == '__main__':

    original_image_path = 'annoted_image.png'
    original_image = cv2.imread(original_image_path)

    # Check if image is read correctly
    if original_image is None:
        raise FileNotFoundError(f"File {original_image_path} not found.")

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

    real_world_positions, timestamps_of_frames = full_keypoint_detection(original_image, final_detections)
