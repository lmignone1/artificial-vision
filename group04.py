from system import *
import os, time
from tracks import *
import argparse as ap

# logger = logging.getLogger(__file__)
# logger.setLevel(logging.INFO)

# logging.basicConfig(level=logging.INFO)

PATH = os.path.dirname(__file__)

SHOW_DETECTOR = True
SHOW_TRACKER = True


def get_args():
    parser = ap.ArgumentParser(description='Detection and tracking for PAR')
    parser.add_argument('-v', '--video', help='name of the video to run', type=str, required=True)
    parser.add_argument('-c', '--configuration', help='configuration file for setting the two rois', default='config.txt', type=str)
    parser.add_argument('-r', '--results', help='result file for writing all tracks', default='results.txt', type=str)
    return parser.parse_args()

args = get_args()

system = System(args.configuration)

video = cv2.VideoCapture(os.path.join(PATH, args.video))
fps = video.get(cv2.CAP_PROP_FPS) # frames per second
total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT) # total number of frames
duration_seconds = total_frames / fps

# logger.info("Video duration: %s s", str(duration_seconds))

sec = 0
while True:
    video.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames, frame = video.read()
    sec += SAMPLE_TIME
    
    if sec > duration_seconds or not hasFrames:
        # logger.info('Video ended')
        break

    sec = round(sec, 2)
    
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    # logger.debug('Frame shape: %s', str(frame.shape))
    
 
    detections = system.predict(frame, confidence=CONFIDENCE, show=SHOW_DETECTOR)
    tracks = system.update_tracks(detections, frame=frame, show=SHOW_TRACKER)

    track : CustomTrack
    for track in tracks:
        
        if track.is_tentative():
            # tracker_logger.debug('track is tentative %s', track.track_id)
            continue
        
        if not track.is_confirmed() or track.is_deleted(): 
            continue

        if track.is_confirmed():
            # tracker_logger.debug('track is confirmed %s', track.track_id)
            
            if not system.is_observed(track):
                system.add_track(track)

            system.update_roi(track)
            system.update_par(track, frame)
    
    system.print_scene(frame)
    # time.sleep(SAMPLE_TIME)

system.write_par(args.results)

video.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()