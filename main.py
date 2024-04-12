
import detection
import tracking
import feature_extraction

import cv2

# PLAN:
# 1. Feature extractor && Tracking: Fiks at når spiller utenfor skjerm, forsvinner tracking box - IKKE plasseres randomly på publikum
#   - Gjøres ved å først feature extracte, deretter bruke dette til å hjelpe avgjøre hvem som forsvinner ut og kommer inn igjen så samme spiller ivaretar samme nummer gjennom HELE klippet
""" Disse to punktene kan dog fikses direkte med en bedre trent modell slik at final_detections direkte får riktig info i player_id osv, det gjør det super enkelt
    å integrere med min alt eksisterende tracking funksjon"""
# 2. Detections: Utvid til å separere mellom to lag, linjedommere og hoveddommer.
#   - Ved egne bokser for dommere. Gjør ved sammenligne innad i laget og splitte ut de som er mest forskjellig?)
# 3. Detections: Fiks så keeperne assosieres med riktige lag.
#   - Hvordan though - må se på laget som flest har rygg mot keeper i.e forvsarer målet?



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
