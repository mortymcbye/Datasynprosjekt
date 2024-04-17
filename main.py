import detection
import tracking
import feature_extraction

import keypoint_detection
import analysis

import cv2


"""
PLAN:
1. Fix del 2:
    - Finjuster keypoint_extraction så den suksessfullt farger ut spillernes bokser helt
    - Fullfør translation til real_world via keypoints. Hvordan, for trenger detected keypoints, ekte keypoints men også link mellom hvilke detected keypoints vi har funnet
2. Tweek del 1:
    - Påse at tracking er helt korrekt, at ikke spillere bytter ID osv, selv ved exit og enter of field/frame
    - Legg inn feature extraction dersom behov for enklere/bedre tracking!
"""

if __name__ == '__main__':
    
    # Detection part:
    input_video_location = 'soccer.mp4'
    output_initial_frame_location = 'The_first_frame'                      # This is where the first frame will be saved
    raw_detections, annotated_image, final_detections = detection.full_detection(input_video_location, output_initial_frame_location) 

    # Note that 'output_initial_frame_location' ikke kan brukes direkte i feature_extractor da den mangler mappenavnet, derfor 'initial_frame_location'

    # Feature extract:
    initial_frame_location = cv2.imread('The_first_frame/first_frame.jpg')
    features_from_initial_frame = feature_extraction.feature_extractor(initial_frame_location, final_detections)

    # Tracking part:
    tracking.full_tracking(input_video_location, final_detections)


    ########### Part two  ###########

    # Keypoint detection:
        # Return string of positions of ALL detected objects
        # Viktig at deres ID ivaretas gjennom hele sekvens her kan feature extraction bli nødvendig
    real_world_positions, timestamps_of_frames = keypoint_detection.full_keypoint_detection(output_initial_frame_location, final_detections) 

    # Analysis:
        # Ta inn disse to vektorene og bruk enkel matte kombinert med deres ID for å beregne deres stats
        # Display ved grafer tenker jeg kan være greit. Én graf per stat med alle 22 spillere vist gjennom de 1 minutter med video
    velocities, accelerations = analysis.calculate_velocity_and_acceleration(real_world_positions, timestamps_of_frames)


