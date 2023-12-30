from deep_sort_realtime.deepsort_tracker import *
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import logging,torch, cv2
from deep_sort_realtime.deep_sort.track import Track

detector_logger = logging.getLogger('yolo')
detector_logger.setLevel(logging.INFO)

tracker_logger = logging.getLogger('deepsort')
tracker_logger.setLevel(logging.INFO)

# TRACKER
MAX_IOU_DISTANCE = 0.6
MAX_AGE = 75 # 75/15 = 5 secondi
N_INIT = 15 # 15/15 = 1 secondo
MAX_COSINE_DISTANCE = 0.25
NN_BUDGET = 5

# DETECTOR
#https://docs.ultralytics.com/modes/predict/#inference-arguments
# inserendo HEIGHT e WIDTH pari a 640 e 480 abbiamo prestazioni peggiori del detector rispetto a quando sono 576 e 352

HEIGHT = 576 # se metto 640 abbiamo uguali performance 
WIDTH = 352
TARGET = 'person'
CLASSES = [0, 24, 26, 28] # 0 = person, 24 = handbag, 26 = backpack, 28 = suitcase

class System():

    def __init__(self):
        self.detector = YOLO('yolov8s.pt')
        self.classes = self.detector.names
        
        self.tracker = DeepSort(max_iou_distance=MAX_IOU_DISTANCE, max_age=MAX_AGE, n_init=N_INIT, max_cosine_distance=MAX_COSINE_DISTANCE, nn_budget=NN_BUDGET)  
        # max_io_distance con 0.7 significa che 2 bb devono avere una distanza massima del 70 %. piu è alto e piu tollerante è il tracker
        # max_age = 30 è il numero di frame in cui un oggetto non viene rilevato prima di essere eliminato. Essendo fps = 10, allora abbiamo 3 secondi prima di rimuovere l oggettoà
        # n_init = 10  è il numero di frame in cui un oggetto deve essere rilevato prima di essere considerato un oggetto vero e proprio. Impiega 1 secondo
        # max_cosine_distance = 0.15 è la distanza massima tra 2 feature per essere considerate uguali. Più è alto e piu tollerante è il tracker
        # nn_budget = 5 frame precdenti del feature vector devono essere considerati. Se none allora le considero tutte

        #tracker = DeepSort(embedder=EMBEDDER_CHOICES[1], embedder_model_name= 'osnet_x0_75', max_cosine_distance=0.5, max_age=600)
        #https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        detector_logger.info('Using Device: %s', self.device)
    
    def _extract_detections(self, res : Results, frame):
        boxes : Boxes = res.boxes
        confidences = boxes.conf
        coord = boxes.xyxyn.cpu().numpy()  # si normalizza in modo da mantenere le dimensioni e per facilita di interpretazione 
        labels = boxes.cls
        
        detector_logger.info('Number of boxes found: %s', len(boxes))
        detector_logger.debug('Confidence: %s', str(confidences))
        detector_logger.debug('Coordinates: %s', str(coord))
        detector_logger.debug('Labels: %s', str(labels))

        x_shape = frame.shape[1]
        y_shape = frame.shape[0]

        plot_bb = []
        detections = []
        for i in range(len(labels)):
            row = coord[i]

            if self._class_to_label(labels[i]) == TARGET:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                detections.append([[x1, y1, x2-x1, y2-y1], confidences[i], int(labels[i])]) # sistema di riferimento al contrario (origine è top left del frame)
                plot_bb.append([[x1, y1, x2, y2], confidences[i], int(labels[i])])
        
        return detections, plot_bb

    def update_tracks(self, detections, frame, show=False):
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        frame_to_show = frame.copy()
        
        track : Track
        for track in tracks:
         
            if track.is_tentative():
                tracker_logger.debug('track is tentative %s', track.track_id)
                continue

            if track.is_confirmed():
                tracker_logger.debug('track is confirmed %s', track.track_id)

            if not track.is_confirmed() or track.is_deleted(): 
                continue

            id = track.track_id 
            bb = track.to_ltrb() 
            x_min, y_min, x_max, y_max = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]) # sono top left e bottom right. Con il sistema di riferimento al contrario le coordinate di bottom right sono piu grandi
            
            if show:
                cv2.rectangle(frame_to_show, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame_to_show, f"ID: {id}", (x_min+5, y_min-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        if show:
            cv2.imshow('DEEP SORT', frame_to_show)
            if cv2.waitKey(1) & 0xFF:
                pass
        
        return tracks
    
    def predict(self, frame, confidence=0.6, show=False):
        self.detector.to(self.device)
        res = self.detector.predict(frame, imgsz=[HEIGHT, WIDTH], conf = confidence, classes = CLASSES)[0] # there is only one result in the list
        detections, plot_bb = self._extract_detections(res, frame)

        if show:
            frame_to_show = frame.copy()
            for i in plot_bb:
                bb = i[0]
                x1, y1, x2, y2 = bb
                cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            cv2.imshow('YOLO', frame_to_show)
            if cv2.waitKey(1) & 0xFF:
                pass
    
        return detections
        

    def _class_to_label(self, x):
        return self.classes[int(x)]
    
