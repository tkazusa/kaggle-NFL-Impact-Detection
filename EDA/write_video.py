import os
import subprocess

import cv2
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
colorpal = sns.color_palette("husl", 9)

import warnings

warnings.filterwarnings("ignore")

DATA_DIR = '../data/input/'
OUTPUT_DIR = 'output/'
VIDEO_DIR = OUTPUT_DIR + 'video'


# Modified function from to take single frame.
# https://www.kaggle.com/samhuddleston/nfl-1st-and-future-getting-started
def annotate_frame(video_path: str, video_labels: pd.DataFrame, stop_frame: int, output_dir: str) -> str:
    VIDEO_CODEC = "MP4V"
    HELMET_COLOR = (0, 0, 0)    # Black
    IMPACT_COLOR = (0, 0, 255)  # Red
    video_name = os.path.basename(video_path)
    
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = output_dir + "labeled_" + video_name
    tmp_output_path = output_dir + "tmp_labeled" + video_name
    output_video = cv2.VideoWriter(tmp_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps, (width, height))
    frame = 0
    while True:
        print(frame)
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        
        # We need to add 1 to the frame count to match the label frame index that starts at 1
        frame += 1
        if frame == stop_frame:
             break
        
        # Let's add a frame index to the video so we can track where we are
        img_name = f"{video_name}_frame{frame}"
        cv2.putText(img, img_name, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, HELMET_COLOR, thickness=2)
    
        # Now, add the boxes
        boxes = video_labels.query("video == @video_name and frame == @frame")
        for box in boxes.itertuples(index=False):
            if box.impact == 1 and box.confidence > 1 and box.visibility > 0:    # Filter for definitive head impacts and turn labels red
                color, thickness = IMPACT_COLOR, 2
            else:
                color, thickness = HELMET_COLOR, 1
            # Add a box around the helmet
            cv2.rectangle(img, (box.left, box.top), (box.left + box.width, box.top + box.height), color, thickness=thickness)
            cv2.putText(img, box.label, (box.left, max(0, box.top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=1)
        output_video.write(img)
    output_video.release()
    
    # Not all browsers support the codec, we will re-load the file at tmp_output_path and convert to a codec that is more broadly readable using ffmpeg
    if os.path.exists(output_path):
        os.remove(output_path)
    subprocess.run(["ffmpeg", "-i", tmp_output_path, "-crf", "18", "-preset", "veryfast", "-vcodec", "libx264", output_path])
    os.remove(tmp_output_path)
    
    return output_path


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR+'video', exist_ok=True)

    tr_labels = pd.read_csv(DATA_DIR + 'train_labels.csv')

    videos = tr_labels.dropna().reset_index()
    video_name = videos.iloc[0]['video']
    all_frame = videos[videos['video']==video_name]
    video_path = f"{DATA_DIR}train/{video_name}"
    stop_frame = videos.iloc[-1]['frame']
    print(stop_frame)

    output_path = annotate_frame(
                video_path=video_path,
                video_labels=tr_labels,
                stop_frame=stop_frame,
                output_dir=OUTPUT_DIR
                )
    print(output_path)