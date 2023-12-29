import os

PATH = os.path.dirname(__file__)
FILE_NAME =  os.path.join('video', 'video0.mp4')
FRAMES_DIR = 'frames'
FPS = 1/5   # 1/T

os.chdir(PATH)
os.system("rm -rf frames")
os.makedirs(FRAMES_DIR, exist_ok=True)
os.system("ffmpeg -i {} -r {} {}/$Frame{}.jpg".format(FILE_NAME, FPS, FRAMES_DIR, "%04d"))






