# artificial-vision

This project enables the detection and tracking of pedestrians for attribute recognition, including gender, upper color, lower color, presence of a bag, and presence of a hat. Additionally, the project counts the number of passages and time for each tracked individual within two possible regions of interest specified in `config.txt`. All tracked people are displayed in the `results.txt` . The file contains information about each tracked individual, including their id. The detector used is Yolo8s, while the tracker used is Deep Sort.

# Dependencies

Before starting to use the project, ensure that your system satisfies the following dependencies by executing the following command in your terminal:

```bash
pip3 install -r requirements.txt
```

# Run project

Execute the project by running this command. You must specify your video, your own  `config.txt` for roi coords and eventually the `results.txt` . 

```bash
python3 group04.py --video video.mp4 -c config.txt -r results.txt
```
