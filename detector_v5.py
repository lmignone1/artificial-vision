import torch
import cv2
import numpy as np
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

FILE_NAME =  os.path.dirname(os.path.realpath(__file__)) + '/IMG_3064.MOV'

class YoloDetector():

    def __init__(self):
        # Using yolov5s for our purposes of object detection, you may use a larger model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yolov7-tiny.torchscript.pt')
        # self.model = torch.jit.load(model_path)
        self.classes = self.model.names
        #print(self.classes)
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
    
    def plot_boxes(self, results, frame, height, width, confidence=0.2):

        labels, cord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]

            if row[4]>=confidence and self.class_to_label(labels[i]) == 'person':
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 0, 255)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                #cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                #detections.append(((x1, y1), (x2, y2), row[4], int(labels[i])))
                detections.append([[x1, y1, x2 - x1, y2- y1], confidence, int(labels[i])])
                #In this demonstration, we will only be detecting persons. You can add classes of your choice
                # if self.class_to_label(labels[i]) == 'person':

                #     x_center = x1 + (x2-x1)
                #     y_center = y1 + ((y2-y1) / 2)

                #     tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype = np.float32)
                #     confidence = float(row[4].item())
                #     feature = 'car'

                #     detections.append(([x1, y1, int(x2), int(y2)], row[4].item(), 'person'))
        
        return frame, detections
    

if __name__ == '__main__':
    detector = YoloDetector()
    tracker = DeepSort()

    # frame = cv2.imread('test.jpg')
    # height, width = frame.shape[:2]
    # results = detector.score_frame(frame)
    # frame, detections = detector.plot_boxes(results, frame, height, width)
    # print(detections)
    # cv2.imshow('frame', cv2.resize(frame, (int(width/2), int(height/2))))
    # cv2.waitKey(0)



    # webcam = cv2.VideoCapture(0)
    # while True:
    #     _, frame = webcam.read()
    #     #frame = cv2.flip(frame, 1)
    #     height, width = frame.shape[:2]
    #     results = detector.score_frame(frame)
    #     frame, detections = detector.plot_boxes(results, frame, height, width)
    #     for i in detections:
    #         i = i[0]
    #         cv2.rectangle(frame, (int(i[0]),int(i[1])),(int(i[2]),int(i[3])),(0,0,255),2)
    #     cv2.imshow('frame', cv2.resize(frame, (int(width/2), int(height/2))))
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # webcam.release()
    # cv2.destroyAllWindows()


    video = cv2.VideoCapture(FILE_NAME)

    # Setting resolution for webcam

    while True:
        _, frame = video.read()
        if frame is None:
            print('last frame')
            break
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        results = detector.score_frame(frame)
        frame, detections = detector.plot_boxes(results, frame, height, width)
        tracks = tracker.update_tracks(detections, frame=frame) # serve per aggiornare il tracker con le detection individuate nella fase precedente.
        for track in tracks:
            if not track.is_confirmed(): # se il track non è confermato, si ignora
                continue
            track_id = track.track_id # per ottenere l'id del track
            ltrb = track.to_ltrb() # per ottenere il bounding box
            x_min, y_min, x_max, y_max = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, "ID: "+str(track_id), (x_min-5, y_min-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('frame', cv2.resize(frame, (int(width/2), int(height/2))))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()