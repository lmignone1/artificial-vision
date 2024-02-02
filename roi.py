from system import *
import os
import time
import cv2
import torch
import logging
import numpy as np

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)

logging.basicConfig()  # livello globale

PATH = os.path.dirname(__file__)

VIDEO = 2

SRC = 0  # 0 = video, 1 = webcam
SAMPLE_TIME = 1/15  # 1/fps dove fps = #immagini/secondi
SHOW = True

# Definizione delle coordinate e dimensioni dei rettangoli
#HEIGHT = 576 # se metto 640 abbiamo uguali performance 
#WIDTH = 352
roi1 = (int(WIDTH * 0.1), int(HEIGHT * 0.2), int(WIDTH * 0.4), int(HEIGHT * 0.4))
roi2 = (int(WIDTH * 0.5), int(HEIGHT * 0.7), int(WIDTH * 0.5), int(HEIGHT * 0.3))
roi1_x, roi1_y, roi1_w, roi1_h = roi1
roi2_x, roi2_y, roi2_w, roi2_h = roi2


system = System()

if SRC == 0:
    video = cv2.VideoCapture(os.path.join(PATH, 'video', f'video{VIDEO}.mp4'))
    fps = video.get(cv2.CAP_PROP_FPS)  # frames per second
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)  # total number of frames
    duration_seconds = total_frames / fps

    logger.info("Video duration: %s s", str(duration_seconds))

elif SRC == 1:
    video = cv2.VideoCapture(0)
    duration_seconds = None

def crop_image(detections):
    for detection in detections:
    # Assicurati che la struttura delle detection sia corretta
        if isinstance(detection, list) and len(detection[0]) >= 4:
            x, y, w, h = detection[0][:4]  # Le prime quattro posizioni sono considerate come bounding box
            cropped_img = frame[y:y+h, x:x+w]  # Effettua la crop dell'immagine

            # Stampa delle informazioni di debug
            logger.debug(f"Bounding box coordinates: x={x}, y={y}, w={w}, h={h}")
            logger.debug(f"Cropped image shape: {cropped_img.shape if cropped_img is not None else None}")
            
            # Centro dei bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            point_color = (0, 255, 0)  # Colore verde
            point_radius = 5  # Raggio del punto
            #cv2.circle(frame, (int(center_x), int(center_y)), point_radius, point_color, -1)  # -1 riempie il cerchio
            

            # Visualizza l'immagine solo se la bounding box ha dimensioni valide
            if w > 0 and h > 0:
                cv2.imshow("Cropped Image", cropped_img)
                #cv2.waitKey(0)  # Aspetta un tasto per chiudere l'immagine
            
            
    return cropped_img, center_x, center_y

def tracking():

    # Inizializza un dizionario per tracciare il tempo totale di permanenza e il numero di entrate per ciascun ID nelle ROI
    

    tracks = system.update_tracks(detections, frame=frame, show=SHOW)

    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Verifica se l'ID è già presente nel dizionario
            if track_id not in total_info:
                total_info[track_id] = {"ROI1": {"time": 0, "entrances": 0, "inside": False},
                                        "ROI2": {"time": 0, "entrances": 0, "inside": False}}

            # Ottenere le coordinate del bounding box
            bb = track.to_ltrb()
            x_min, y_min, x_max, y_max = map(int, bb)

            # Centro dei bounding box
            center_x1 = (x_min + x_max) // 2
            center_y1 = (y_min + y_max) // 2

            # Verifica se il punto si trova all'interno di ROI1
            if roi1_x <= center_x1 <= (roi1_x + roi1_w) and roi1_y <= center_y1 <= (roi1_y + roi1_h):
                if not total_info[track_id]["ROI1"]["inside"]:
                    total_info[track_id]["ROI1"]["entrances"] += 1
                    total_info[track_id]["ROI1"]["inside"] = True

                # Aggiorna il tempo totale di permanenza nella ROI1 per l'ID
                total_info[track_id]["ROI1"]["time"] += SAMPLE_TIME

            else:
                total_info[track_id]["ROI1"]["inside"] = False

            # Verifica se il punto si trova all'interno di ROI2
            if roi2_x <= center_x1 <= roi2_x + roi2_w and roi2_y <= center_y1 <= roi2_y + roi2_h:
                if not total_info[track_id]["ROI2"]["inside"]:
                    total_info[track_id]["ROI2"]["entrances"] += 1
                    total_info[track_id]["ROI2"]["inside"] = True

                # Aggiorna il tempo totale di permanenza nella ROI2 per l'ID
                total_info[track_id]["ROI2"]["time"] += SAMPLE_TIME

            else:
                total_info[track_id]["ROI2"]["inside"] = False

    # Stampa la lista di tempi di permanenza e entrate
    print("List with residence time and entries in the ROIs:")
    for track_id, info in total_info.items():
        print(f"ID {track_id}: ROI1 - Time: {info['ROI1']['time']}, Entrances: {info['ROI1']['entrances']}")
        print(f"ID {track_id}: ROI2 - Time: {info['ROI2']['time']}, Entrances: {info['ROI2']['entrances']}")










sec = 0

while True:
    video.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, frame = video.read()
    sec += SAMPLE_TIME

    if duration_seconds is not None and (sec > duration_seconds or not hasFrames):
        logger.info('Video ended')
        break

    sec = round(sec, 2)

    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    logger.debug('Frame shape: %s', str(frame.shape))

    # Disegno dei rettangoli sul frame del video
    color_roi1 = (0, 255, 0)  # Colore verde per il primo rettangolo
    color_roi2 = (0, 0, 255)  # Colore rosso per il secondo rettangolo
    thickness = 2  # Spessore della linea

    cv2.rectangle(frame, (roi1[0], roi1[1]), (roi1[0] + roi1[2], roi1[1] + roi1[3]), color_roi1, thickness)
    cv2.rectangle(frame, (roi2[0], roi2[1]), (roi2[0] + roi2[2], roi2[1] + roi2[3]), color_roi2, thickness)

    cv2.putText(frame, "ROI_1", (roi1[0] + 5, roi1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "ROI_2", (roi2[0] + 5, roi2[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    
    

    detections = system.predict(frame, show=SHOW)
    #crop, center_x, center_y = crop_image(detections)
    #cv2.imshow("Cropped", crop)
    ################################
    ############################### NON SI TROVA CON LA CORDINATA X
    #print(center_x, '----', roi1_x, '----', roi1_x+roi1_w)
    #print(center_y)
    #print('---------------')
    #print(f"center_x: {center_x}, roi1_x: {roi1_x}, roi1_w: {roi1_w}")
    
    tracking()
    

    time.sleep(SAMPLE_TIME)

video.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()



