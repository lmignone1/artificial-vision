from system import *
import os, time

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)

logging.basicConfig() # livello globale

PATH = os.path.dirname(__file__)

NUMBER_OF_VIDEO = 7

SRC = 0 # 0 = video, 1 = webcam
VIDEO_NAME =  os.path.join(PATH, 'video', f'video{NUMBER_OF_VIDEO}.mp4')
SAMPLE_TIME = 1/15   # 1/fps dove fps = #immagini/secondi

SHOW = True

system = System()

if SRC == 0:
    video = cv2.VideoCapture(VIDEO_NAME)
    fps = video.get(cv2.CAP_PROP_FPS) # frames per second
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT) # total number of frames
    duration_seconds = total_frames / fps
    
    logger.info("Video duration: %s s", str(duration_seconds))

elif SRC == 1:
    video = cv2.VideoCapture(0)

sec = 0
while True:
    video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames, frame = video.read()
    sec += SAMPLE_TIME
    
    if sec > duration_seconds or not hasFrames:
        logger.info('Video ended')
        break

    sec = round(sec, 2)
    
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    logger.debug('Frame shape: %s', str(frame.shape))
    
    detections = system.predict(frame, show=SHOW)

    tracks = system.update_tracks(detections, frame=frame, show=SHOW)
    
    time.sleep(SAMPLE_TIME)
    
video.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()