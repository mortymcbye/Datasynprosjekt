import detection
import tracking
import feature_extraction
import analysis

import cv2


if __name__ == '__main__':
    
    # Detection part:
    input_video_location = 'Images/soccer.mp4'
    output_initial_frame_location = 'The_first_frame'                      # This is where the first frame will be saved
    raw_detections, annotated_image, final_detections = detection.full_detection(input_video_location, output_initial_frame_location) 

    # Feature extract:
    initial_frame_location = cv2.imread('The_first_frame/first_frame.jpg')
    features_from_initial_frame = feature_extraction.feature_extractor(initial_frame_location, final_detections)

    # Tracking part:
    real_world_positions, timestamps_of_frames = tracking.full_tracking(final_detections)

    # Velocity, accl. analysis:
    velocities, accelerations = analysis.calculate_velocity_and_acceleration(real_world_positions, timestamps_of_frames)
    analysis.plot_velocity(velocities, timestamps_of_frames)
    analysis.plot_acceleration(accelerations, timestamps_of_frames)


