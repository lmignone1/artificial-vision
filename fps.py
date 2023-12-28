import cv2, os, argparse, glob, PIL, tqdm

def extract_frames(video):
    # Process the video
    ret = True
    cap = cv2.VideoCapture(video) #legge il video
    f = 0
    while ret:
        ret, img = cap.read() # legge il frame successivo, ritorna True se letto correttamente e il frame
        if ret:
            f += 1
            img = cv2.resize(img, (224,224))
            PIL.Image.fromarray(img).save(os.path.join(frames_path, video, "{:05d}.jpg".format(f)))
    cap.release()

# For all the videos
file_list = [path for path in glob.glob(os.path.join(videos_path,"**"), recursive=True)
             if os.path.isfile(path)] #glob mi da tutti i path in quella directory
print(file_list)
for video in tqdm.tqdm(file_list):
  if os.path.isdir(os.path.join(frames_path, video)):
    continue

  os.makedirs(os.path.join(frames_path, video))
  #extract_frames(video)
  os.system("ffmpeg -i {} -r 1/1 {}/{}/$Frame{}.jpg".format(video, frames_path, video, "%05d"))