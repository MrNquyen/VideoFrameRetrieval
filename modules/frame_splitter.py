import cv2
import argparse
import numpy as np
import os

from utils.registry import registry

class FrameSplitter:
    def __init__(self, interval: int):
        self.writer = registry.get_writer("common")
        self.interval = interval

    def split_frames(
            self,
            source  : str,
            save_dir: str = None, 
            is_saved: bool = False
        ):
        """
            Spliting video into frame

            Parameters:
            -----------
            - source: mp4 video path
            - save_dir: directory where frames is saved
        """
        cap = cv2.VideoCapture(source)
        if is_saved:
            if save_dir==None:
                self.writer.LOG_ERROR("Please provide valid save dir")
                raise ValueError
                if not os.path.exists(save_dir):
                    self.writer.LOG_INFO("Create save directory")
                    os.mkdir(save_dir)

        #-- Setup config
        interval_sec = 2
        fps = cap.get(cv2.CAP_PROP_FPS) # frame per second
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        num_skip_frames = interval_sec * fps

        #-- Split frame
        frames = []
        frame_id = 0
        saved_count = 0
        if cap.isOpened() == False:
            self.writer.LOG_INFO('Cap is not open')

        # print(f"Start splitting - Total frames: {total_frames}")
        while(cap.isOpened()):
            ret, frame = self.cap_frame(cap, frame_id=frame_id)
            # print(f"Frame {frame_id} - Type: {type(frame)}")
            #~ Save frame
            if is_saved and frame is not None:
                save_path = os.path.join(save_dir, f'frame_{saved_count:04d}.webp')
                self.save_frame(save_path, frame)

            #~ Yield each frame
            if frame is not None: frames.append(frame)
            if not ret:
                #~~ Save last frame
                if frame_id - num_skip_frames < total_frames:
                    ret, frame = self.cap_frame(cap, frame_id=total_frames - 1)
                    if is_saved and frame is not None:
                        save_path = os.path.join(save_dir, f'frame_{saved_count:04d}.webp')
                        self.save_frame(save_path, frame)
                    saved_count += 1
                    if frame is not None: frames.append(frame)
                break
            frame_id += num_skip_frames
            saved_count += 1

        self.writer.LOG_INFO(f"Splitting {saved_count} frames from video")
        cap.release()

        frames = [for frame in frame if frame is not None]
        return frames

    def cap_frame(self, cap, frame_id):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        return ret, frame

    def save_frame(self, save_path, frame):
        cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_WEBP_QUALITY), 80])

