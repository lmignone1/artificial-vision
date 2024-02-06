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
from torch.nn import functional as F

detector_logger = logging.getLogger('yolo')
detector_logger.setLevel(logging.INFO)

tracker_logger = logging.getLogger('deepsort')
tracker_logger.setLevel(logging.INFO)

par_logger = logging.getLogger('par')
par_logger.setLevel(logging.DEBUG)

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
        path_to_dir = os.path.dirname(__file__)

        self.detector = YOLO(os.path.join(path_to_dir, 'models', MODEL))
        self.detector_classes = self.detector.names
        
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

        self.par_model = AttributeRecognitionModel(num_attributes=5)
        #self.par_model.load_state_dict(torch.load(os.path.join(path_to_dir, 'par_models', 'best_model.pth')))
        self.par_model.eval()

        self.tracks_collection = dict()

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
        frame_tracks = frame.copy()
        tracks = self.tracker.update_tracks(detections, frame=frame_tracks)
        
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
        frame_predict = frame.copy()
        self.detector.to(self.device)
        res = self.detector.predict(frame_predict, imgsz=[HEIGHT, WIDTH], conf = confidence, classes = CLASSES)[0] # there is only one result in the list
        detections, plot_bb = self._extract_detections(res, frame_predict)

        if show:
            frame_to_show = frame.copy()
            for i in plot_bb:
                bb = i[0]
                x1, y1, x2, y2 = bb
                cv2.rectangle(frame_to_show, (x1, y1), (x2, y2), (0, 0, 255), 2) # top left, bottom right
            
            cv2.imshow('YOLO', frame_to_show)
            if cv2.waitKey(1) & 0xFF:
                pass
                 
        return detections
        

    def _class_to_label(self, x):
        return self.detector_classes[int(x)]
    
    def _crop_image(self, track : CustomTrack, frame, show=False):
        frame_to_crop = frame.copy()

        bb = track.to_ltwh()
        x, y, w, h = map(int, bb)
        
        cropped_img = frame_to_crop[y:y+h, x:x+w]

        par_logger.debug(f"Bounding box coordinates: x={x}, y={y}, w={w}, h={h}")
        par_logger.debug(f"Cropped image shape: {cropped_img.shape if cropped_img is not None else None}")

        if show:
            cv2.imshow(f"Cropped Image {track.track_id} ", cropped_img)
            cv2.waitKey(1) & 0xFF

        try:
            cropped_img = cv2.resize(cropped_img, (WIDTH_PAR, HEIGHT_PAR))
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            cropped_img = transforms.ToTensor()(cropped_img)
            cropped_img = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])(cropped_img)

            # cropped_img = cropped_img.permute(1, 2, 0).numpy()  
            # cropped_img = (cropped_img * 255).astype(np.uint8)
            # cv2.imshow(f"Cropped Image {track.track_id} ", cropped_img)
            # cv2.waitKey(1) & 0xFF
            
        except Exception:
            par_logger.error(f"Error in cropping image: {track.track_id}")
            cropped_img = None
        
        par_logger.info(f"Crop image with id {track.track_id} done")
        return cropped_img
    

    def update_roi(self, track : CustomTrack):
        bb = track.to_ltrb()
        x_min, y_min, x_max, y_max = map(int, bb)
        
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        # Verifica se il punto si trova all'interno di ROI1
        if self._roi1_x <= center_x <= (self._roi1_x + self._roi1_w) and self._roi1_y <= center_y <= (self._roi1_y + self._roi1_h):
            if not track.roi1_inside:
                track.roi1_transit += 1
                track.roi1_inside = True
            track.roi1_time += SAMPLE_TIME
        else:
            track.roi1_inside = False
        
        if self._roi2_x <= center_x <= self._roi2_x + self._roi2_w and self._roi2_y <= center_y <= self._roi2_y + self._roi2_h:
            if not track.roi2_inside:
                track.roi2_transit += 1
                track.roi2_inside = True
            track.roi2_time += SAMPLE_TIME
        else:
            track.roi2_inside = False
        
        par_logger.debug(f"ID {track.track_id}: ROI1 - Time: {round(track.roi1_time, 2)}, Passages: {track.roi1_transit}")
        par_logger.debug(f"ID {track.track_id}: ROI2 - Time: {round(track.roi2_time, 2)}, Passages: {track.roi2_transit}")

    def update_par(self, track : CustomTrack, frame):
        frame_par = frame.copy()
        self.par_model.to(self.device)
        
        if not track._is_par_confirmed:
            cropped_img = self._crop_image(track, frame_par, show=SHOW_CROP)
            
            if cropped_img is not None:
                cropped_img = cropped_img.float()
                cropped_img = cropped_img.unsqueeze(0).to(self.device)
                o = self.par_model(cropped_img)

                for task_index in range(len(o)):
                    pred = o[task_index]

                    if task_index < 2:  # multiclasse
                        pred = F.softmax(pred, dim=1)
                        index_class = torch.argmax(pred, dim=1).item()
                    else:
                        pred = pred.squeeze()
                        pred = pred > 0.5
                        index_class = int(pred.item())

                    track.add_par_measurement(task_index, index_class)
                
                track.check_limit_par_measurements()
            

    def print_roi(self, frame):
        frame_to_show = frame.copy()
        cv2.rectangle(frame_to_show, (self._roi1_x, self._roi1_y), (self._roi1_x + self._roi1_w, self._roi1_y + self._roi1_h), (0, 0, 0), 3)
        cv2.rectangle(frame_to_show, (self._roi2_x, self._roi2_y), (self._roi2_x + self._roi2_w, self._roi2_y + self._roi2_h), (0, 0, 0), 3)
        
        cv2.putText(frame_to_show, "1", (self._roi1_x + 5, self._roi1_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame_to_show, "2", (self._roi2_x + 5, self._roi2_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        
        cv2.imshow('ROI', frame_to_show)
        
        if cv2.waitKey(1) & 0xFF:
            pass
        return frame_to_show
    
    def write_par(self, path):
        writer = FileJson(path)
        writer.write_par(self.tracks_collection)
    
    def add_track(self, track : CustomTrack):
        self.tracks_collection[int(track.track_id)] = track
    
    def is_observed(self, track : CustomTrack):
        return int(track.track_id) in self.tracks_collection
    
    def print_scene(self, frame):
        frame_to_show = frame.copy()
        frame_to_show = self.print_roi(frame_to_show)
        in_roi1 = 0
        in_roi2 = 0
        outside_roi = 0
        passages_roi1 = 0
        passages_roi2 = 0

        for id, track in self.tracks_collection.items():

            if track.roi1_inside:
                
                bb = track.to_ltrb()
                x_min, y_min, x_max, y_max = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]) # sono top left e bottom right. Con il sistema di riferimento al contrario le coordinate di bottom right sono piu grandi

                cv2.rectangle(frame_to_show, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                # Calcola le dimensioni per il rettangolo bianco
                text_width, text_height = cv2.getTextSize(f"{track.track_id}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

                # Disegna il rettangolo bianco
                cv2.rectangle(frame_to_show, (x_min , y_min), (x_min + text_width, y_min + text_height), (255, 255, 255), -1)
                cv2.putText(frame_to_show, f"{track.track_id}", (x_min+1, y_min+11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                in_roi1 += 1
                
                
            
            elif track.roi2_inside:

                bb = track.to_ltrb()
                x_min, y_min, x_max, y_max = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]) # sono top left e bottom right. Con il sistema di riferimento al contrario le coordinate di bottom right sono piu grandi

                cv2.rectangle(frame_to_show, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Calcola le dimensioni per il rettangolo bianco
                text_width, text_height = cv2.getTextSize(f"{track.track_id}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

                # Disegna il rettangolo bianco
                cv2.rectangle(frame_to_show, (x_min , y_min), (x_min + text_width, y_min + text_height), (255, 255, 255), -1)
                cv2.putText(frame_to_show, f"{track.track_id}", (x_min+1, y_min+11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                in_roi2 += 1
                
                

            if track.roi1_inside == False and track.roi2_inside == False:

                bb = track.to_ltrb()
                x_min, y_min, x_max, y_max = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]) # sono top left e bottom right. Con il sistema di riferimento al contrario le coordinate di bottom right sono piu grandi

                cv2.rectangle(frame_to_show, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                # Calcola le dimensioni per il rettangolo bianco
                text_width, text_height = cv2.getTextSize(f"{track.track_id}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

                # Disegna il rettangolo bianco
                cv2.rectangle(frame_to_show, (x_min , y_min), (x_min + text_width, y_min + text_height), (255, 255, 255), -1)
                cv2.putText(frame_to_show, f"{track.track_id}", (x_min+1, y_min+11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                if not (x_min >= WIDTH or x_max <= 0 or y_min >= HEIGHT or y_max <= 0): #se persona esce dalla scena non viene contata, questo perché se persona esce dalla scena non perde subito il tracking (non si sa il motivo)                    
                    outside_roi += 1
        
            # alcola i passaggi totali nelle roi
            passages_roi1 += track.roi1_transit
            passages_roi2 += track.roi2_transit

            # Calcola le dimensioni del testo per ogni riga
            text_line1 = f"People in ROI: {in_roi1 + in_roi2}"
            text_line2 = f"Total Person: {in_roi1 + in_roi2 + outside_roi}"
            text_line3 = f"Passages in ROI 1: {passages_roi1}"
            text_line4 = f"Passages in ROI 2: {passages_roi2}"

            # Disegna il rettangolo bianco con testo nero
            cv2.rectangle(frame_to_show, (0, 0), (200, 80), (255, 255, 255), -1)

            # Disegna le tre righe di testo
            cv2.putText(frame_to_show, text_line1, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(frame_to_show, text_line2, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(frame_to_show, text_line3, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(frame_to_show, text_line4, (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


            

            # Stampa info persone
            cv2.rectangle(frame_to_show, (x_min - 30, y_max), (x_max + 30, y_max + 40), (255, 255, 255), -1)
            cv2.putText(frame_to_show, f"Gender: {track.gender}", (x_min - 25, y_max + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            if track.bag == True and track.hat == False:
                cv2.putText(frame_to_show, "Bag", (x_min - 25, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            elif track.bag == False and track.hat == True:
                cv2.putText(frame_to_show, "Hat", (x_min - 25, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            elif track.bag == True and track.hat == True:
                cv2.putText(frame_to_show, "Bag Hat", (x_min - 25, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            else:
                 cv2.putText(frame_to_show, "No Bag No Hat", (x_min - 25, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)   
            cv2.putText(frame_to_show, f"U-L: {track.upper} - {track.lower}", (x_min - 25, y_max + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
 

        cv2.imshow('Stampa_Contest', frame_to_show)
        if cv2.waitKey(1) & 0xFF:
            pass