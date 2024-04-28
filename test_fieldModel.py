import cv2
import numpy as np
import time
import argparse

#Based on: https://github.com/kamal3344/v4-Inference/blob/main/Inference_image.py
#Keypoints are equal to the center of the bounding box
class Yolov4:
    def __init__(self):
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
                        'penalty_left_bot']
        self.Neural_Network = cv2.dnn.readNetFromDarknet(self.cfg, self.weights)
        self.outputs = self.Neural_Network.getUnconnectedOutLayersNames()
        self.COLORS = np.random.randint(0, 255, size=(len(self.classes), 3), dtype="uint8")
        self.image_size = opt.img_size

    def bounding_box(self, detections):
        visible = []
        invisible = []
        try:
            confidence_score = []
            ids = []
            cordinates = []
            Threshold = 0.5
            for i in detections:
                for j in i:
                    probs_values = j[5:]
                    class_ = np.argmax(probs_values)
                    confidence_ = probs_values[class_]

                    if confidence_ > Threshold:
                        w, h = int(j[2] * self.image_size), int(j[3] * self.image_size)
                        x, y = int(j[0] * self.image_size - w / 2), int(j[1] * self.image_size - h / 2)
                        cordinates.append([x, y, w, h])
                        ids.append(class_)
                        confidence_score.append(float(confidence_))
            final_box = cv2.dnn.NMSBoxes(cordinates, confidence_score, Threshold, .6)
            #Prints visible and non visible points
            for i in ids:
                visible.append(self.classes[i])
            invisible = [i for i in self.classes if i not in visible]
            print(invisible)
            print(visible)
            
            return final_box, cordinates, confidence_score, ids

        except Exception as e:
            print(f'Error in : {e}')

    def predictions(self, prediction_box, bounding_box, confidence, class_labels, width_ratio, height_ratio, end_time,
                       image):
        try:
            for j in prediction_box.flatten():
                x, y, w, h = bounding_box[j]
                x = int(x * width_ratio)
                y = int(y * height_ratio)
                w = int(w * width_ratio)
                h = int(h * height_ratio)
                label = str(self.classes[class_labels[j]])
                conf_ = str(round(confidence[j], 2))
                color = [int(c) for c in self.COLORS[class_labels[j]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label + ' ' + conf_, (x, y - 2), cv2.FONT_HERSHEY_COMPLEX, .5, color, 2)
                time = f"Inference time: {end_time:.3f}"
                cv2.putText(image, time, (10, 13), cv2.FONT_HERSHEY_COMPLEX, .5, (156, 0, 166), 1)
            return image

        except Exception as e:
            print(f'Error in : {e}')

    def Inference(self, image, original_width, original_height):
        try:
            blob = cv2.dnn.blobFromImage(image, 1 / 255, (320, 320), True, crop=False)
            self.Neural_Network.setInput(blob)
            start_time = time.time()
            output_data = self.Neural_Network.forward(self.outputs)
            end_time = time.time() - start_time
            final_box, cordinates, confidence_score, ids = self.bounding_box(output_data)
            outcome = self.predictions(final_box, cordinates, confidence_score, ids, original_width / 320,
                                       original_height / 320, end_time, image)
            return outcome
        except Exception as e:
            print(f'Error in : {e}')

if __name__ == "__main__":
    parse=argparse.ArgumentParser()
    parse.add_argument('--weights', type=str, default='yolov4.weights', help='weights path')
    parse.add_argument('--cfg', type=str, default='yolov4.cfg', help='cfg path')
    parse.add_argument('--image', type=str, default='', help='image path')
    parse.add_argument('--video', type=str, default='', help='video path')
    parse.add_argument('--img_size', type=int, default='', help='size of w*h')
    opt = parse.parse_args()
    obj = Yolov4()  # constructor called and executed


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
            fps = cv2.CAP_PROP_FPS
            width = cap.get(3)
            height = cap.get(4)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter("demo.avi", fourcc, fps, (int(width), int(height)))
            while cap.isOpened():
                res, frame = cap.read()
                if res == True:
                    outcome = obj.Inference(image=frame, original_width=width, original_height=height)
                    cv2.imshow("demo", outcome)
                    output.write(outcome)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f'Error in : {e}')