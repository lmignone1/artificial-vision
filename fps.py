import cv2
import os


PATH = os.path.dirname(__file__)
FILE_NAME =  os.path.join(PATH, 'video', 'video7.mp4')

count=1
vidcap = cv2.VideoCapture(FILE_NAME)
fps = vidcap.get(cv2.CAP_PROP_FPS) # frames per second
totalNoFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT) # total number of frames
durationInSeconds = totalNoFrames / fps
print("durationInSeconds:", durationInSeconds, "s")

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    height, width = image.shape[:2]
    if hasFrames:
        print("frame trovato al secondo: ", sec)
        cv2.imshow('frame', cv2.resize(image, (int(width/2), int(height/2))))
        if cv2.waitKey(1) & 0xFF:
            pass
    return hasFrames
sec = 0
frameRate = 1/(10) # Change this number to 1 for each 1 second fps = #immagini/secondi
    

while True:
    success = getFrame(sec)
    count = count + 1
    sec = sec + frameRate
    if sec > durationInSeconds or not success:
        break
    sec = round(sec, 2)

    import time
    time.sleep(frameRate)








# import cv2, os, argparse, glob, PIL, tqdm

# def extract_frames(video):
#     # Process the video
#     ret = True
#     cap = cv2.VideoCapture(video) #legge il video
#     f = 0
#     while ret:
#         ret, img = cap.read() # legge il frame successivo, ritorna True se letto correttamente e il frame
#         if ret:
#             f += 1
#             img = cv2.resize(img, (224,224))
#             PIL.Image.fromarray(img).save(os.path.join(frames_path, video, "{:05d}.jpg".format(f)))
#     cap.release()

# # For all the videos
# file_list = [path for path in glob.glob(os.path.join(videos_path,"**"), recursive=True)
#              if os.path.isfile(path)] #glob mi da tutti i path in quella directory
# print(file_list)
# for video in tqdm.tqdm(file_list):
#   if os.path.isdir(os.path.join(frames_path, video)):
#     continue

#   os.makedirs(os.path.join(frames_path, video))
#   #extract_frames(video)
#   os.system("ffmpeg -i {} -r 1/1 {}/{}/$Frame{}.jpg".format(video, frames_path, video, "%05d"))