import cv2
import numpy as np
import time
import argparse
from sklearn.metrics import mean_squared_error

#Based on: https://github.com/kamal3344/v4-Inference/blob/main/Inference_image.py
#and: https://github.com/Hmzbo/Football-Analytics-with-Deep-Learning-and-Computer-Vision/blob/master/Football%20Object%20Detection%20With%20Tactical%20Map.ipynb


#Keypoints are equal to the center of the bounding box

class Yolov4:
    def __init__(self, opt):
        self.weights = opt.weights  # loading weights
        self.cfg = opt.cfg  # loading cfg file
        self.classes = ['centre_circle_bottom_intersec',
                        'centre_circle_top_intersec',
                        'centre_spot',
                        'right_spot',
                        'left_spot',
                        'top_center',
                        'bottom_center',
                        'top_left_corner',
                        'top_right_corner',
                        'bottom_left_corner',
                        'bottom_right_corner',
                        'penalty_right_top',
                        'penalty_right_bot',
                        'penalty_left_top',
                        'penalty_left_bot',
                        'penalty_circ_left_bot',
                        'penalty_circ_left_top',
                        'penalty_circ_right_bot',
                        'penalty_circ_right_top']
        
        self.Neural_Network = cv2.dnn.readNetFromDarknet(self.cfg, self.weights)
        self.outputs = self.Neural_Network.getUnconnectedOutLayersNames()
        self.COLORS = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        self.image_size = opt.img_size
        self.frame_nbr = 0
        self.keypoints_displacement_mean_tol = 10 #Pixels

        #Visible and invisible keypoints for ecah frame
        self.visible = {}
        self.invisible = {}

        #For temporary storage 
        self.h_matrix = None
        self.prev_coords = None

    def bounding_box(self, detections):
        """Filters out detections to be drawn on frame"""
        # Coordinates refer to an array in the following shape: [x,y,w,h]
        # Each coordinate has its own ID
        visible = []

        try:
            confidence_score = []
            ids = []
            coordinates = []
            Threshold = 0.5
            for i in detections:
                for j in i:
                    probs_values = j[5:]
                    class_ = np.argmax(probs_values)
                    confidence_ = probs_values[class_]

                    #Only use coordinate if the detection is good enough
                    if confidence_ > Threshold:
                        w, h = int(j[2] * self.image_size), int(j[3] * self.image_size)
                        x, y = int(j[0] * self.image_size - w / 2), int(j[1] * self.image_size - h / 2)
                        #Check if detection already exists in list (to avoid certain double detections sometimes appearing)
                        if class_ not in ids:   
                            coordinates.append([x, y, w, h])
                            ids.append(class_)
                            confidence_score.append(float(confidence_))
            final_box = cv2.dnn.NMSBoxes(coordinates, confidence_score, Threshold, .6)

            #Prints visible and non visible points
            for i in ids:
                visible.append(self.classes[i])

            self.visible[self.frame_nbr] = visible
            self.invisible[self.frame_nbr] = [i for i in self.classes if i not in self.visible[self.frame_nbr]]
            
            return final_box, coordinates, confidence_score, ids

        except Exception as e:
            print(f'Error in : {e}')

    def predictions(self, prediction_box, bounding_box, confidence, class_labels, width_ratio, height_ratio, end_time,
                       image):
        """Draws bounding boxes for each present coordinate"""
        #Dict to store each present cordinate and their values
        coordinate_dict = {}
        try:
            for j in prediction_box.flatten():
                x, y, w, h = bounding_box[j]
                x = int(x * width_ratio) 
                y = int(y * height_ratio)
                w = int(w * width_ratio)
                h = int(h * height_ratio)
                label = str(self.classes[class_labels[j]])
                coordinate_dict[label] = (x,y) 
                conf_ = str(round(confidence[j], 2))
                color = [int(c) for c in self.COLORS[class_labels[j]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label + ' ' + conf_, (x, y - 2), cv2.FONT_HERSHEY_COMPLEX, .5, color, 2)
                time = f"Inference time: {end_time:.3f}"
                cv2.putText(image, time, (10, 13), cv2.FONT_HERSHEY_COMPLEX, .5, (156, 0, 166), 1)
            return image, coordinate_dict

        except Exception as e:
            print(f'Error in : {e}')

    def Inference(self, image, original_width, original_height):
        try:
            blob = cv2.dnn.blobFromImage(image, 1 / 255, (320, 320), True, crop=False)
            self.Neural_Network.setInput(blob)
            start_time = time.time()
            output_data = self.Neural_Network.forward(self.outputs)
            end_time = time.time() - start_time
            final_box, coordinates, confidence_score, ids = self.bounding_box(output_data)
            outcome, coordinate_dict = self.predictions(final_box, coordinates, confidence_score, ids, original_width / 320,
                                       original_height / 320, end_time, image)
            return outcome, coordinate_dict
        except Exception as e:
            print(f'Error in : {e}')

    def keypointPrint(self):
        """Utility for printing detected keypoints"""
        for frame in self.visible:
            print("_____________________________________________")
            print("_____________________________________________")
            print("Frame " + str(frame))
            print("---------------------------------------------")
            print("Visible keypoints: ")
            print(str(self.visible[frame]))
            print("---------------------------------------------")
            print("Invisible keypoints: ")
            print(str(self.invisible[frame]))

    def identify_keypoints(self, coordinate_dict):
        """For fetching detected keypoints real world coordinates"""

        real_world_coordinates = {}

        # Information on real_world metrics of a standard soccer field:
        field_width, field_height = 105.0, 68.0

        # Corners of the Field
        top_left_corner = (0, 0)
        top_right_corner = (field_width, 0)
        bottom_left_corner = (0, field_height)
        bottom_right_corner = (field_width, field_height)

        # Centre spot
        centre_spot = (field_width / 2, field_height / 2)

        # Penalty Spots (Assuming 10.97 meters from the goal line)
        left_spot = (10.97, field_height/2)
        right_spot = (field_width - 10.97, field_height/2)
        # Penalty corners, assuming 16.5 m from goal line
        penalty_right_top = (field_width-16.5, 13-84)
        penalty_right_bot = (field_width-16.5, field_height-13.84)
        penalty_left_top = (16.5, 13.84)
        penalty_left_bot = (16.5, field_height-13.84)
        #Penalty circle
        penalty_circ_left_bot = (16.5, field_height/2+7.3)
        penalty_circ_left_top = (16.5, field_height/2-7.3)
        penalty_circ_right_bot = (field_width-16.5, field_height/2+7.3)
        penalty_circ_right_top = (field_width-16.5, field_height/2-7.3)
        
        # Midfield Circle (Assuming radius of 9.15 meters)
        midfield_circle_radius = 9.15
        centre_circle_bottom_intersec= (field_width/2, field_height/2-midfield_circle_radius)
        centre_circle_top_intersec = (field_width/2, field_height/2+midfield_circle_radius)

        # Center Line
        top_center = (field_width / 2, 0)
        bottom_center = (field_width / 2, field_height)


        field_coordinates = {
            "centre_circle_bottom_intersec": centre_circle_bottom_intersec,
            "centre_circle_top_intersec": centre_circle_top_intersec,
            "centre_spot":centre_spot,
            "right_spot":right_spot,
            "left_spot":left_spot,
            "top_center":top_center,
            "bottom_center":bottom_center,
            "top_left_corner":top_left_corner,
            "top_right_corner":top_right_corner,
            "bottom_left_corner":bottom_left_corner,
            "bottom_right_corner":bottom_right_corner,
            "penalty_right_top":penalty_right_top,
            "penalty_right_bot":penalty_right_bot,
            "penalty_left_top":penalty_left_top ,
            "penalty_left_bot":penalty_left_bot ,
            "penalty_circ_left_bot":penalty_circ_left_bot ,
            "penalty_circ_left_top":penalty_circ_left_top ,
            "penalty_circ_right_bot":penalty_circ_right_bot,
            "penalty_circ_right_top":penalty_circ_right_top
        }
        
        #Creates dict with real world value for each present coordinate
        for i in coordinate_dict:
            if i in field_coordinates:
                real_world_coordinates[i] = field_coordinates[i]
                field_coordinates.pop(i)

        return real_world_coordinates 

    def homography(self, keypoints):
        """Creates homography matrix using detected keypoints"""

        #Real world coordinates of detected keypoints
        real_world_coordinates = self.identify_keypoints(keypoints)

        if len(keypoints) > 3:  #We need at least 4 keypoints for homography matrix
            if self.frame_nbr > 1: #Creates matrix if first frame
                common_keypoints = set(self.prev_coords) & set(keypoints) 
                #Check if change is big enough to bother creating a new matrix
                if len(common_keypoints) > 3:
                    #Calculating ms-error between previous and current coordinates
                    coor_prev = [self.prev_coords[i] for i in common_keypoints ]
                    coor_curr = [keypoints[i] for i in common_keypoints]
                    coor_error = mean_squared_error(coor_prev, coor_curr)
                    update_homography = coor_error > self.keypoints_displacement_mean_tol 
                else:
                    update_homography = True
            else:
                update_homography = True

            if update_homography:
                src_points = np.array([keypoints[i] for i in keypoints])
                dst_points = np.array([real_world_coordinates[i] for i in real_world_coordinates])
                h, mask = cv2.findHomography(src_points, dst_points)

                if h is not None:
                    #Only update if we get a matrix
                    self.h_matrix = h

            self.prev_coords = keypoints

    def myTransformation(self, keypoints, player_coords):
        #Updates h_matrix
        self.homography(keypoints)

        #Transform player coordinates to field
        pred_dst_pts = []                                                           
        for pt in player_coords:                                            # Loop over players frame coordiantes
            pt = np.append(np.array(pt), np.array([1]), axis=0)                     # Convert to homogeneous coordinates
            dest_point = np.matmul(self.h_matrix, np.transpose(pt))                 # Apply homography transformation
            dest_point = dest_point/dest_point[2]                                   # Revert to 2D-coordinates
            pred_dst_pts.append(list(np.transpose(dest_point)[:2]))                 # Update players tactical map coordiantes list
        pred_dst_pts = np.array(pred_dst_pts)

        return pred_dst_pts



if __name__ == "__main__":
    parse=argparse.ArgumentParser()
    parse.add_argument('--weights', type=str, default='Weights_CFG/best.weights', help='weights path')
    parse.add_argument('--cfg', type=str, default='Weights_CFG/best.cfg', help='cfg path')
    parse.add_argument('--image', type=str, default='Images/test.jpg', help='image path')
    parse.add_argument('--video', type=str, default='Images/soccer.mp4', help='video path')
    parse.add_argument('--img_size', type=int, default='320', help='size of w*h')
    opt = parse.parse_args()
    obj = Yolov4(opt)  # constructor called and executed


    if opt.image:
        try:
            image = cv2.imread(opt.image, 1)
            original_width , original_height = image.shape[1] , image.shape[0]
            obj.Inference(image=image,original_width=original_width,original_height=original_height)
            cv2.imshow('Inference ',image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f'Error in : {e}')

    if opt.video:
        try:
            cap = cv2.VideoCapture(opt.video)
            fps = 30 #cv2.CAP_PROP_FPS
            print("FPS" + str(fps))
            width = cap.get(3)
            height = cap.get(4)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter("demo.avi", fourcc, fps, (int(width), int(height)))
            while cap.isOpened():
                obj.frame_nbr += 1
                res, frame = cap.read()
                if res == True:
                    #Keypoints detection
                    outcome, cordinate_dict = obj.Inference(image=frame, original_width=width, original_height=height)
                    cv2.imshow('demo', frame)
                    #Saving results in video demo.avi
                    if outcome is None:
                        output.write(frame)
                    else:
                        output.write(outcome)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
                
            obj.keypointPrint()
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f'Error in : {e}')