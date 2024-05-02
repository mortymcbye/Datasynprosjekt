import numpy as np

# Define the dictionary of real-world coordinates for each keypoint
keypoint_dict = {
    'centre_circle_top_intersec': [0, 9.15, 0],
    'centre_spot': [0, 0, 0],
    'right_spot': [11, 0, 0],
    'left_spot': [-11, 0, 0],
    'top_center': [0, 34, 0],
    'bottom_center': [0, -34, 0],
    'top_left_corner': [-52.5, 34, 0],
    'top_right_corner': [52.5, 34, 0],
    'bottom_left_corner': [-52.5, -34, 0],
    'bottom_right_corner': [52.5, -34, 0],
    'penalty_right_top': [16.5, 34, 0],
    'penalty_right_bot': [16.5, -34, 0],
    'penalty_left_top': [-16.5, 34, 0],
    'penalty_left_bot': [-16.5, -34, 0],
    'penalty_circ_left_bot': [-11, -11, 0],  # A rough estimate
    'penalty_circ_left_top': [-11, 11, 0],
    'penalty_circ_right_bot': [11, -11, 0],
    'penalty_circ_right_top': [11, 11, 0]
}


def find_visible_keypoint_screen_coordinates(visible_keypoints):
    # Fetch the screen coordinates for the visible keypoints
    screen_coordinates = []
    for keypoint in visible_keypoints:
        # Implement actual screen coordinate acquisition here
        screen_coordinates.append([sx, sy, sz])  # Replace with actual data acquisition
    return np.array(screen_coordinates)

def find_corresponding_real_keypoints_coordinates(visible_keypoints):
    # Fetch the corresponding real-world coordinates for visible keypoints
    real_coordinates = []
    for keypoint in visible_keypoints:
        if keypoint in keypoint_dict:
            real_coordinates.append(keypoint_dict[keypoint])
    return np.array(real_coordinates)

def calculate_rotation_matrix(screen_coordinates, real_coordinates):
    # Conversion of the variable names inside the function
    screen_keypoints = np.array(screen_coordinates)
    real_keypoints = np.array(real_coordinates)
    
    mean_screen = np.mean(screen_keypoints, axis=0)
    mean_real = np.mean(real_keypoints, axis=0)
    
    normalized_screen = screen_keypoints - mean_screen
    normalized_real = real_keypoints - mean_real
    
    covariance_matrix = np.dot(normalized_screen.T, normalized_real)
    U, _, Vt = np.linalg.svd(covariance_matrix)
    
    rotation_matrix = np.dot(Vt.T, U.T)
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = np.dot(Vt.T, U.T)
    return rotation_matrix

def project_onto_ground(detections, rotation_matrix):
    real_life_boxes = []
    for box in detections['xyxy']:
        top_left = np.array([box[0], box[1], 1])
        bottom_right = np.array([box[2], box[3], 1])
        projected_top_left = np.dot(rotation_matrix, top_left)
        projected_bottom_right = np.dot(rotation_matrix, bottom_right)
        real_life_boxes.append([projected_top_left, projected_bottom_right])
    return real_life_boxes

def full_real_life_translation(visible_keypoints, detections):
    screen_coordinates = find_visible_keypoint_screen_coordinates(visible_keypoints)
    real_coordinates = find_corresponding_real_keypoints_coordinates(visible_keypoints)
    
    rotation_matrix = calculate_rotation_matrix(screen_coordinates, real_coordinates)
    real_life_boxes = project_onto_ground(detections, rotation_matrix)
    return real_life_boxes

if __name__ == '__main__':
    visible_keypoints = ['top_center', 'penalty_circ_left_top', 'centre_circle_top_intersec']
    detections = {'xyxy': np.random.rand(2, 4) * 1024}  # Example detection boxes

    real_life_result = full_real_life_translation(visible_keypoints, detections)
    print(real_life_result)



