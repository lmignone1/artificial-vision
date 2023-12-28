import torch
import cv2
import numpy as np
import os, sys
from deep_sort_realtime.deepsort_tracker import *

FILE_NAME =  os.path.dirname(os.path.realpath(__file__)) + '/IMG_3064.MOV'

VIDEO_SRC = 1
class YoloDetector():

    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device:', self.device)
    
    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))

        results = self.model(frame)
        
        #print(results.pandas().xyxy[0])
        labels, cord = results.xyxyn[0].cpu(), results.xyxyn[0].cpu()
        labels, cord = labels[:, -1].numpy(), cord[:, :-1].numpy()
        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame, confidence=0.3, show = False):

        labels, cord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]

            if row[4]>=confidence and self.class_to_label(labels[i]) == 'person':
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                
                if show:
                    bgr = (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                
                detections.append([[x1, y1, x2 - x1, y2- y1], confidence, int(labels[i])])

                #     x_center = x1 + (x2-x1)
                #     y_center = y1 + ((y2-y1) / 2)

        return frame, detections
    

if __name__ == '__main__':
    detector = YoloDetector()
    tracker = DeepSort(embedder=EMBEDDER_CHOICES[1], embedder_model_name= 'osnet_x0_75', max_cosine_distance=0.5, max_age=600)
    #https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html

    if VIDEO_SRC == 0:
        video = cv2.VideoCapture(FILE_NAME)
    elif VIDEO_SRC == 1:
        video = cv2.VideoCapture(0)
    else:
        frame = cv2.imread('test.jpg')
        height, width = frame.shape[:2]
        results = detector.score_frame(frame)
        frame, detections = detector.plot_boxes(results, frame, show=True)
        print(detections)
        cv2.imshow('frame', cv2.resize(frame, (int(width/2), int(height/2))))
        cv2.waitKey(0)
        sys.exit(0)



    while True:
        _, frame = video.read()
        if frame is None:
            print('last frame')
            break
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        results = detector.score_frame(frame)
        frame, detections = detector.plot_boxes(results, frame)
        tracks = tracker.update_tracks(detections, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed(): 
                continue
            track_id = track.track_id 
            ltrb = track.to_ltrb() 
            x_min, y_min, x_max, y_max = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "ID: "+str(track_id), (x_min-5, y_min-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame', cv2.resize(frame, (int(width/2), int(height/2))))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()