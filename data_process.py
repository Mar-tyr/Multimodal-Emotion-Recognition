import json

import numpy as np
import os

import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path
from moviepy.editor import VideoFileClip, AudioFileClip
import matplotlib.pyplot as plt
import scipy.misc
from tqdm import trange, tqdm
import cv2
import os.path as osp
import multiprocessing

portion_to_id = dict(
    train=[17, 43, 30, 28, 46, 19, 41, 26, 62, 39, 25, 56],  # 25
    valid=[64, 48, 16, 58, 34, 45, 23],
    test=[42, 65, 21, 37]  # 54, 53
)


def detect(image, cascade_file="haarcascade_frontalface_alt.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.equalizeHist(image)
    face_coords = cascade.detectMultiScale(image,
                                           # detector options
                                           scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=(48, 48))

    if len(face_coords) == 0:
        return None
    else:
        face_coord = face_coords[0]
        x, y, w, h = face_coord
        face = gray[y: y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        return face, face_coord


def get_sample(video_dir, subject_id, outdir):
    coord_df = pd.DataFrame(columns=['x', 'y', 'w', 'h'])
    mp4_path = video_dir / 'P{}.mp4'.format(subject_id)
    clip = VideoFileClip(str(mp4_path))
    subsampled_audio: AudioFileClip = clip.audio.set_fps(16000)
    subject_videodir = outdir / 'Video/{}'.format(subject_id)
    subject_coorddir = subject_videodir / 'coords'
    subject_fullimgdir = subject_videodir / 'full_img'
    subject_audiodir = outdir / 'Audio/{}'.format(subject_id)
    if not osp.isdir(str(subject_fullimgdir)):
        os.makedirs(str(subject_fullimgdir))
    if not osp.isdir(str(subject_coorddir)):
        os.makedirs(str(subject_coorddir))
    if not osp.isdir(str(subject_audiodir)):
        os.makedirs(str(subject_audiodir))
    audio_frames = []
    for i in trange(1, 7501):
        start_time = 0.04 * (i - 1)
        end_time = 0.04 * i
        audio = np.array(list(subsampled_audio.subclip(start_time, end_time).iter_frames()))
        audio = audio.mean(1)[:640]
        audio_frames.append(audio)
        frame = np.array(list(clip.subclip(start_time, end_time).iter_frames()))[0]
        detect_res = detect(frame)
        if detect_res:
            face, face_coord = detect_res
            face_path = subject_videodir / '{}_{}.jpg'.format(subject_id, i)
            scipy.misc.imsave(face_path, face)
            coord_df.loc[i] = face_coord
        else:
            fullimg_path = subject_fullimgdir / '{}_{}.jpg'.format(subject_id, i)
            scipy.misc.imsave(fullimg_path, cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    audio_frames = np.vstack(audio_frames)
    np.save(str(subject_audiodir / '{}.npy'.format(subject_id)), audio_frames)
    coord_df.to_pickle(str(subject_coorddir / 'coords.pkl'))


if __name__ == '__main__':
    with open('config/path.json', 'r') as fp:
        path_config = json.load(fp)
    root_dir = path_config['gpu_025']
    video_dir = root_dir / 'RECOLA-Video-recordings'
    anno_dir = root_dir / 'RECOLA-Annotation/emotional_behaviour'
    outdir = root_dir / 'recola_out'

    all_subject_ids = set()
    for ids in portion_to_id.values():
        all_subject_ids.update(ids)
    all_subject_ids.remove(17)
    for subject_id in all_subject_ids:
        print(subject_id)
        get_sample(video_dir, subject_id, outdir)
