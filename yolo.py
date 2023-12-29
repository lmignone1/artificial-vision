from deep_sort_realtime.deepsort_tracker import *
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import logging, time, os, sys, torch, cv2

logging.basicConfig(level=logging.DEBUG)

PATH = os.path.dirname(__file__)
FILE_NAME =  os.path.join(PATH, 'video', 'video0.mp4')
TARGET = 'person'
CLASSES = [0, 24, 26, 28]

#https://docs.ultralytics.com/modes/predict/#inference-arguments
HEIGHT = 640
WIDTH = 480

VIDEO_SRC = 0
SAMPLE_TIME = 1/10   # 1/fps dove fps = #immagini/secondi

class Detector():

    def __init__(self):
        self.model = YOLO('yolov5su.pt')
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logging.info('Using Device: %s', self.device)
    
    def _extract_detections(self, res : Results, resized_frame, frame):
        boxes : Boxes = res.boxes
        confidences = boxes.conf
        coord = boxes.xyxyn.cpu().numpy()  # si normalizza in modo da mantenere le dimensioni e per facilita di interpretazione 
        labels = boxes.cls
        
        logging.debug('Number of boxes found: %s', len(boxes))
        logging.debug('Confidence: %s', str(confidences))
        logging.debug('Coordinates: %s', str(coord))
        logging.debug('Labels: %s', str(labels))

        x_shape = resized_frame.shape[1]
        y_shape = resized_frame.shape[0]

        x_shape_plot = frame.shape[1]
        y_shape_plot = frame.shape[0]

        plot_bb = []
        detections = []
        for i in range(len(labels)):
            row = coord[i]

            if self._class_to_label(labels[i]) == TARGET:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                detections.append([[x1, y1, x2-x1, y2-y1], confidences[i], int(labels[i])])

                x1, y1, x2, y2 = int(row[0]*x_shape_plot), int(row[1]*y_shape_plot), int(row[2]*x_shape_plot), int(row[3]*y_shape_plot)
                plot_bb.append([[x1, y1, x2, y2], confidences[i], int(labels[i])])

                #     x_center = x1 + (x2-x1)
                #     y_center = y1 + ((y2-y1) / 2)
        
        return detections, plot_bb


    
    def predict(self, frame, confidence=0.60, show=False):
        self.model.to(self.device)

        resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
    
        res = self.model.predict(frame, imgsz=[HEIGHT, WIDTH], conf = confidence, classes = CLASSES)[0] # there is only one result in the list

        logging.info('Prediction done')

        detections, plot_bb = self._extract_detections(res, resized_frame, frame)

        if show:

            frame_to_show = frame.copy()            
            bgr = (0, 0, 255) 

            for i in plot_bb:
                bb = i[0]
                x1, y1, x2, y2 = bb
                cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), bgr, 2)
            
            frame_to_show = cv2.resize(frame_to_show, (540, 640))
            cv2.imshow('frame', frame_to_show)
            if cv2.waitKey(1) & 0xFF:
                pass
    
        return resized_frame, detections
        

    def _class_to_label(self, x):
        return self.classes[int(x)]
    
   

if __name__ == '__main__':
    detector = Detector()
    tracker = DeepSort(max_age=60)  
    #tracker = DeepSort(embedder=EMBEDDER_CHOICES[1], embedder_model_name= 'osnet_x0_75', max_cosine_distance=0.5, max_age=600)
    #https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html

    if VIDEO_SRC == 0:
        video = cv2.VideoCapture(FILE_NAME)
        fps = video.get(cv2.CAP_PROP_FPS) # frames per second
        totalNoFrames = video.get(cv2.CAP_PROP_FRAME_COUNT) # total number of frames
        durationInSeconds = totalNoFrames / fps
        
        logging.info("Video duration: %s s", str(durationInSeconds))

    elif VIDEO_SRC == 1:
        video = cv2.VideoCapture(0)
    else:
        frame = cv2.imread('test.jpg')
        height, width = frame.shape[:2]
        frame, detections = detector.predict(frame, show=True)
        print(detections)
        cv2.imshow('frame', cv2.resize(frame, (int(width/2), int(height/2))))
        cv2.waitKey(0)
        sys.exit(0)

    
    sec = 0
    while True:
        video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames, frame = video.read()
        sec += SAMPLE_TIME
        
        if sec > durationInSeconds or not hasFrames:
            logging.info('Video ended')
            break

        sec = round(sec, 2)
        
        logging.debug('Frame shape: %s', str(frame.shape))
        
        resized_frame, detections = detector.predict(frame, show=True)

        time.sleep(SAMPLE_TIME)
        
        # tracks = tracker.update_tracks(detections, frame=resized_frame)
        
        # for track in tracks:
        #     if not track.is_confirmed(): 
        #         continue
        #     track_id = track.track_id 
        #     ltrb = track.to_ltrb() 
        #     x_min, y_min, x_max, y_max = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        #     cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        #     cv2.putText(frame, "ID: "+str(track_id), (x_min-5, y_min-8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.imshow('frame', cv2.resize(frame, (int(width/2), int(height/2))))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    video.release()
    cv2.destroyAllWindows()
    torch.cuda.empty_cache()