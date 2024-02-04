from deep_sort_realtime.deepsort_tracker import *
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
import logging,torch, cv2
from deep_sort_realtime.deep_sort.track import Track
from tracks import *
import os
from par import *
import torch
import torchvision.transforms as transforms
import numpy as np
from json_module import *

detector_logger = logging.getLogger('yolo')
detector_logger.setLevel(logging.INFO)

tracker_logger = logging.getLogger('deepsort')
tracker_logger.setLevel(logging.INFO)

par_logger = logging.getLogger('par')
par_logger.setLevel(logging.DEBUG)

SHOW_CROP = False
SAMPLE_TIME = 1/15   # 1/fps dove fps = #immagini/secondi

# TRACKER
MAX_IOU_DISTANCE = 0.6
MAX_AGE = 75 # 75/15 = 5 secondi
N_INIT = 15 # 15/15 = 1 secondo
MAX_COSINE_DISTANCE = 0.25
NN_BUDGET = 8

# DETECTOR
#https://docs.ultralytics.com/modes/predict/#inference-arguments
# inserendo HEIGHT e WIDTH pari a 640 e 480 abbiamo prestazioni peggiori del detector rispetto a quando sono 576 e 352
MODEL = 'yolov8s.pt'
HEIGHT = 576 # se metto 640 abbiamo uguali performance 
WIDTH = 352
TARGET = 'person'
CLASSES = [0] # 0 = person

class System():

    def __init__(self, path_roi):
        self.detector = YOLO(os.path.join(os.path.dirname(__file__), 'models', MODEL))
        self.classes = self.detector.names
        
        self.tracker = DeepSort(max_iou_distance=MAX_IOU_DISTANCE, max_age=MAX_AGE, n_init=N_INIT, max_cosine_distance=MAX_COSINE_DISTANCE, nn_budget=NN_BUDGET, override_track_class=CustomTrack)  
        # max_io_distance con 0.7 significa che 2 bb devono avere una distanza massima del 70 %. piu è alto e piu tollerante è il tracker
        # max_age = 30 è il numero di frame in cui un oggetto non viene rilevato prima di essere eliminato. Essendo fps = 10, allora abbiamo 3 secondi prima di rimuovere l oggettoà
        # n_init = 10  è il numero di frame in cui un oggetto deve essere rilevato prima di essere considerato un oggetto vero e proprio. Impiega 1 secondo
        # max_cosine_distance = 0.15 è la distanza massima tra 2 feature per essere considerate uguali. Più è alto e piu tollerante è il tracker
        # nn_budget = 5 frame precdenti del feature vector devono essere considerati. Se none allora le considero tutte

        #tracker = DeepSort(embedder=EMBEDDER_CHOICES[1], embedder_model_name= 'osnet_x0_75', max_cosine_distance=0.5, max_age=600)
        #https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html

        reader = FileJson(path_roi)
        roi1, roi2 = reader.read_roi()

        roi1 = (int(WIDTH * roi1['x']), int(HEIGHT * roi1['y']), int(WIDTH * roi1['width']), int(HEIGHT * roi1['height'])) 
        roi2 = (int(WIDTH * roi2['x']), int(HEIGHT * roi2['y']), int(WIDTH * roi2['width']), int(HEIGHT * roi2['height']))

        self._roi1_x, self._roi1_y, self._roi1_w, self._roi1_h = roi1
        self._roi2_x, self._roi2_y, self._roi2_w, self._roi2_h = roi2

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
        
        if show:
            frame_to_show = frame.copy()
        
            track : CustomTrack
            for track in tracks:

                if track.is_tentative() or not track.is_confirmed() or track.is_deleted():
                    continue
                    
                id = track.track_id 
                bb = track.to_ltrb() 
                x_min, y_min, x_max, y_max = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]) # sono top left e bottom right. Con il sistema di riferimento al contrario le coordinate di bottom right sono piu grandi

                cv2.rectangle(frame_to_show, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame_to_show, f"ID: {id}", (x_min+5, y_min-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

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
    
    def _crop_image(self, track, frame, show=False):
        frame_to_crop = frame.copy()

        bb = track.to_ltrb()
        x, y, w, h = map(int, bb)
        
        cropped_img = frame_to_crop[y:y+h, x:x+w]

        par_logger.debug(f"Bounding box coordinates: x={x}, y={y}, w={w}, h={h}")
        par_logger.debug(f"Cropped image shape: {cropped_img.shape if cropped_img is not None else None}")

        if show:
            cv2.imshow("Cropped Image", cropped_img)

        cropped_img = cv2.resize(cropped_img, (WIDTH_PAR, HEIGHT_PAR))
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        cropped_img = torch.transforms.ToTensor()(cropped_img)
        cropped_img = torch.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])(cropped_img)
        cropped_img = cropped_img.to(self.device)
        return cropped_img
    

    def update_roi(self, track : CustomTrack):
        bb = track.to_ltrb()
        x_min, y_min, x_max, y_max = map(int, bb)
        
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Verifica se il punto si trova all'interno di ROI1
        if self._roi1_x <= center_x <= (self._roi1_x + self._roi1_w) and self._roi1_y <= center_y <= (self._roi1_y + self._roi1_h):
            if not track._roi1_inside:
                track._roi1_transit += 1
                track._roi1_inside = True
            track._roi1_time += SAMPLE_TIME
        else:
            track._roi1_inside = False
        
        if self._roi2_x <= center_x <= self._roi2_x + self._roi2_w and self._roi2_y <= center_y <= self._roi2_y + self._roi2_h:
            if not track._roi2_inside:
                track._roi2_transit += 1
                track._roi2_inside = True
            track._roi2_time += SAMPLE_TIME
        else:
            track._roi2_inside = False
        
        par_logger.debug(f"ID {track.track_id}: ROI1 - Time: {track._roi1_time}, Entrances: {track._roi1_transit}")
        par_logger.debug(f"ID {track.track_id}: ROI2 - Time: {track._roi2_time}, Entrances: {track._roi2_transit}")

    def update_par(self, track : CustomTrack, frame):
        self._crop_image(track, frame, show=SHOW_CROP)
        pass

    def print_roi(self, frame):
        frame_to_show = frame.copy()
        cv2.rectangle(frame_to_show, (self._roi1_x, self._roi1_y), (self._roi1_x + self._roi1_w, self._roi1_y + self._roi1_h), (0, 255, 0), 2)
        cv2.rectangle(frame_to_show, (self._roi2_x, self._roi2_y), (self._roi2_x + self._roi2_w, self._roi2_y + self._roi2_h), (0, 0, 255), 2)
        
        cv2.putText(frame_to_show, "ROI1", (self._roi1_x + 5, self._roi1_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame_to_show, "ROI2", (self._roi2_x + 5, self._roi2_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow('ROI', frame_to_show)
        
        if cv2.waitKey(1) & 0xFF:
            pass