import cv2
import argparse
import numpy as np
import os

def parser():
    # parse all args
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='Path to source video')
    parser.add_argument('dest_folder', type=str, help='Path to destination folder')
    args = parser.parse_args()
    return args

def cap_frame(cap, frame_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    return ret, frame

def save_frame(save_path, frame):
    cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_WEBP_QUALITY), 80])


def split_frames(source, save_dir=None, is_save=False):
    """
        Spliting video into frame

        Parameters:
        -----------
        - source: mp4 video path
        - save_dir: directory where frames is saved
    """
    #-- Capture video
    cap = cv2.VideoCapture(source)
    if not os.path.exists(save_dir):
        print("Create save directory")
        os.mkdir(save_dir)

    #-- Setup config
    interval_sec = 2
    fps = cap.get(cv2.CAP_PROP_FPS) # frame per second
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    num_skip_frames = interval_sec * fps

    #-- Split frame
    frame_id = 0
    saved_count = 0
    if cap.isOpened() == False:
        print('Cap is not open')

    # print(f"Start splitting - Total frames: {total_frames}")
    while(cap.isOpened()):
        ret, frame = cap_frame(cap, frame_id=frame_id)
        # print(f"Frame {frame_id} - Type: {type(frame)}")
        #~ Save frame
        if is_save and frame is not None:
            save_path = os.path.join(save_dir, f'frame_{saved_count:04d}.webp')
            save_frame(save_path, frame)

        #~ Yield each frame
        yield frame
        if not ret:
            #~~ Save last frame
            if frame_id - num_skip_frames < total_frames:
                # print("Split last frame")
                ret, frame = cap_frame(cap, frame_id=total_frames - 1)
                if is_save and frame is not None:
                    save_path = os.path.join(save_dir, f'frame_{saved_count:04d}.webp')
                    save_frame(save_path, frame)
                saved_count += 1
                yield frame
            break
        frame_id += num_skip_frames
        saved_count += 1

    # Release capture 
    cap.release()
    # print(f'Finish splitting {saved_count} frames')


if __name__=="__main__":
    print("Running split frames")
    source = "data/video.mp4"
    save_dir = "save"

    iterations = split_frames(
        source=source,
        save_dir=save_dir,
        is_save=True
    )
    for id, frame in enumerate(iterations):
        None
