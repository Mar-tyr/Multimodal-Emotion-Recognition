import json
import os
import os.path as osp
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import scipy.misc
from moviepy.editor import VideoFileClip, AudioFileClip
from tqdm import trange, tqdm
from skimage import color

portion_to_id = dict(
    train=[17, 43, 30, 28, 46, 19, 41, 26, 62, 39, 25, 56],  # 25
    valid=[64, 48, 16, 58, 34, 45, 23],
    test=[42, 65, 21, 37]  # 54, 53
)


def get_sample(video_dir, subject_id, outdir):
    mp4_path = video_dir / 'P{}.mp4'.format(subject_id)
    clip = VideoFileClip(str(mp4_path))
    subsampled_audio: AudioFileClip = clip.audio.set_fps(16000)
    subject_audiodir = outdir / 'Audio'
    subject_videodir = outdir / 'Video'
    if not osp.isdir(str(subject_audiodir)):
        os.makedirs(str(subject_audiodir))
    if not osp.isdir(str(subject_videodir)):
        os.makedirs(str(subject_videodir))
    audio_frames = []
    for i in trange(1, 7501):
        start_time = 0.04 * (i - 1)
        end_time = 0.04 * i
        audio = np.array(list(subsampled_audio.subclip(start_time, end_time).iter_frames()))
        audio = audio.mean(1)[:640]
        audio_frames.append(audio)
    audio_frames = np.vstack(audio_frames)
    print('Audio Frames Shape:', audio_frames.shape)
    np.save(str(subject_audiodir / '{}.npy'.format(subject_id)), audio_frames)
    clip.close()

    face_mp4_path = outdir / 'Face' / 'P{}.mp4'.format(subject_id)
    face_clip = VideoFileClip(str(face_mp4_path))
    print('Face Video Fps:', face_clip.fps)

    video_frames = []
    for i, frame in tqdm(enumerate(face_clip.iter_frames())):
        if i == 7500:
            break
        video_frames.append(color.rgb2gray(frame).reshape((1, 48, 48)))
    video_frames = np.vstack(video_frames)
    print("Video Frames Shape:", video_frames.shape)
    np.save(str(subject_videodir / '{}.npy'.format(subject_id)), video_frames)
    face_clip.close()


if __name__ == '__main__':
    with open('config/path.json', 'r') as fp:
        path_config = json.load(fp)
    root_dir = Path(path_config['gpu_025'])
    video_dir = root_dir / 'RECOLA-Video-recordings'
    anno_dir = root_dir / 'RECOLA-Annotation/emotional_behaviour'
    outdir = root_dir / 'recola_out'

    all_subject_ids = set()
    for ids in portion_to_id.values():
        all_subject_ids.update(ids)
    for subject_id in all_subject_ids:
        print(subject_id)
        get_sample(video_dir, subject_id, outdir)
