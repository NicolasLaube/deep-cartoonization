import os
from moviepy.editor import VideoFileClip
import numpy as np
import os
from datetime import timedelta
from tqdm import tqdm


def format_timedelta(td):
    """
    Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds
    """
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def extract_frames_from_movie(movie_path: str, folder: str, nb_images: int = 1500):
    """
    Extracts frames from movie

    :param movie_path: complete path to movie
    :param folder: folder path where to extract frames 
    :param frequency: number of frames to extract from movie
    """
    # assert ".mp4" in movie_path
    # load the video clip
    video_capture = VideoFileClip(movie_path)
    duration = video_capture.duration
    step = duration / nb_images

    # iterate over each possible frame
    for current_time in tqdm(np.arange(0, duration, step)):
        # format the file name and save it
        frame_duration_formatted = format_timedelta(timedelta(seconds=current_time)).replace(":", "-")
        frame_filename = os.path.join(folder, f"frame{frame_duration_formatted}.jpg")
        # save the frame with the current duration
        video_capture.save_frame(frame_filename, current_time)
