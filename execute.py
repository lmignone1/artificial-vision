from system import *
import os, time
from tracks import *

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)

logging.basicConfig(level=logging.DEBUG) # livello globale

PATH = os.path.dirname(__file__)

VIDEO = 7

SRC = 0 # 0 = video, 1 = webcam

SHOW_DETECTOR = False
SHOW_TRACKER = True

system = System()

if SRC == 0:
    video = cv2.VideoCapture(os.path.join(PATH, 'video', f'video{VIDEO}.mp4'))
    fps = video.get(cv2.CAP_PROP_FPS) # frames per second
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT) # total number of frames
    duration_seconds = total_frames / fps
    
    logger.info("Video duration: %s s", str(duration_seconds))

elif SRC == 1:
    video = cv2.VideoCapture(0)
    duration_seconds = None

sec = 0
while True:
    video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames, frame = video.read()
    sec += SAMPLE_TIME
    
    if duration_seconds is not None and (sec > duration_seconds or not hasFrames):
        logger.info('Video ended')
        break

    sec = round(sec, 2)
    
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    logger.debug('Frame shape: %s', str(frame.shape))
    
    system.print_roi(frame)
    detections = system.predict(frame, show=SHOW_DETECTOR)
    tracks = system.update_tracks(detections, frame=frame, show=SHOW_TRACKER)

    track : CustomTrack
    for track in tracks:
        
        if track.is_tentative():
            tracker_logger.debug('track is tentative %s', track.track_id)
            continue

        if track.is_confirmed():
            tracker_logger.debug('track is confirmed %s', track.track_id)
            system.update_roi(track)

            # se io counter par < N allora classificazione altrimenti stop.Quindi procedo con il massimo . 
            # alle prossime iterazioni essendo confermata la classificazione non faremo piu calcoli per quel track

        if not track.is_confirmed() or track.is_deleted(): 
            continue
    
    time.sleep(SAMPLE_TIME)
    
video.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()