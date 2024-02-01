from system import *
import os
import time
import cv2
import torch
import logging

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)

logging.basicConfig()  # livello globale

PATH = os.path.dirname(__file__)

VIDEO = 7

SRC = 0  # 0 = video, 1 = webcam
SAMPLE_TIME = 1/15  # 1/fps dove fps = #immagini/secondi
SHOW = True

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

            # Visualizza l'immagine solo se la bounding box ha dimensioni valide
            if w > 0 and h > 0:
                cv2.imshow("Cropped Image", cropped_img)
                #cv2.waitKey(0)  # Aspetta un tasto per chiudere l'immagine

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

    detections = system.predict(frame, show=SHOW)
    crop = crop_image(detections)
    tracks = system.update_tracks(detections, frame=frame, show=SHOW)
    
    #### Se crop lo metti qui, sembra metterci tantissimo
    

    time.sleep(SAMPLE_TIME)

video.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()
