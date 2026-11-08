import os
import glob
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
from scipy.signal import stft
import cv2

from pathlib import Path

CLASSES = [
    "female speech", "male speech", "clapping", "telephone", "laughter",
    "domestic sounds", "walk footsteps", "door", "music", "musical instrument",
    "water tap", "bell", "knock",
]


def resolve_starss_path(root=None):
    """Dynamically resolve STARSS23 dataset path without machine-specific hardcoding."""
    if root is not None and os.path.exists(root):
        return root

    env_path = os.environ.get("STARSS_DATA_DIR")
    if env_path and os.path.exists(env_path):
        return env_path

    current_file = Path(__file__).resolve()
    candidate_paths = [
        current_file.parents[3] / "Data" / "starss23",  # e.g. SLAM/Data/starss23
        current_file.parents[4] / "Data" / "starss23",  # e.g. LAM/Data/starss23
        Path.cwd() / "Data" / "starss23",
        Path.cwd().parent / "Data" / "starss23",
        Path("Data/starss23"),
    ]

    for p in candidate_paths:
        if p.exists():
            return str(p)

    return str(candidate_paths[0])


class STARSSDataset(Dataset):
    """
    Dataset loader for STARSS23 (Sony-TAUS Spatial Sound Events 2023).
    Loads synchronized 4-channel microphone audio, computes covariance C_obs,
    loads 360-degree video frames, and parses ground truth labels and DOAs.
    """
    def __init__(self, root=None, split="dev-train-sony", window_s=1.0,
                 band=(4000.0, 5000.0), img_size=(128, 256), active_only=False):
        self.root = resolve_starss_path(root)
        self.window_s = window_s
        self.band = band
        self.img_h, self.img_w = img_size
        self.active_only = active_only

        self.wav_dir = os.path.join(self.root, "mic_dev", split)
        self.mp4_dir = os.path.join(self.root, "video_dev", split)
        self.csv_dir = os.path.join(self.root, "metadata_dev", split)

        wavs = sorted(glob.glob(os.path.join(self.wav_dir, "*.wav")))
        self.clips = []
        for w in wavs:
            base = os.path.splitext(os.path.basename(w))[0]
            mp4 = os.path.join(self.mp4_dir, base + ".mp4")
            csvf = os.path.join(self.csv_dir, base + ".csv")
            if os.path.exists(mp4) and os.path.exists(csvf):
                self.clips.append((base, w, mp4, csvf))

        self.windows = []
        self.sr = 24000
        for ci, (base, w, mp4, csvf) in enumerate(self.clips):
            try:
                sr, data = wavfile.read(w)
                self.sr = sr
                n_samples = data.shape[0]
                win_samples = int(window_s * sr)
                n_windows = n_samples // win_samples
                events = self._read_events(csvf)
                for wi in range(n_windows):
                    if self.active_only:
                        t0_frame = int(wi * window_s * 10)
                        t1_frame = t0_frame + int(window_s * 10)
                        if not any(t0_frame <= f < t1_frame for f in events):
                            continue
                    self.windows.append((ci, wi))
            except Exception as e:
                print(f"[STARSSDataset] Warning loading {w}: {e}")

        self._events_cache = {}

    def _read_events(self, csvf):
        events = {}
        if not os.path.exists(csvf):
            return events
        with open(csvf) as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                try:
                    frame, cls, src, az, el, dist = [int(x) for x in row[:6]]
                    events.setdefault(frame, []).append(
                        (cls, src, np.deg2rad(az), np.deg2rad(el), dist)
                    )
                except ValueError:
                    continue
        return events

    def _clip_events(self, ci):
        if ci not in self._events_cache:
            self._events_cache[ci] = self._read_events(self.clips[ci][3])
        return self._events_cache[ci]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ci, wi = self.windows[idx]
        base, wav_path, mp4_path, csv_path = self.clips[ci]

        t0 = wi * self.window_s
        t_center = t0 + self.window_s / 2.0

        sr, data = wavfile.read(wav_path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        else:
            data = data.astype(np.float32)

        win_samples = int(self.window_s * sr)
        seg = data[wi * win_samples:(wi + 1) * win_samples]

        # STFT computation for narrowband microphone covariance
        f, _, Z = stft(seg.T, fs=sr, window="hann", nperseg=1024,
                       noverlap=1024 - 480, boundary=None, padded=False)
        band_mask = (f >= self.band[0]) & (f <= self.band[1])
        X = Z[:, band_mask, :]
        n_ch, n_bins, n_frames = X.shape
        flat = X.reshape(n_ch, -1)
        C = (flat @ flat.conj().T) / max(flat.shape[1], 1)
        C = 0.5 * (C + C.conj().T)
        freq = 0.5 * (self.band[0] + self.band[1])

        C_obs = torch.stack([
            torch.from_numpy(np.ascontiguousarray(C.real)),
            torch.from_numpy(np.ascontiguousarray(C.imag)),
        ]).float()

        # Video frame extraction
        cap = cv2.VideoCapture(mp4_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_center * 1000.0)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            frame = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_w, self.img_h))
        image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # Ground truth events
        events = self._clip_events(ci)
        f0 = int(t0 * 10)
        f1 = f0 + int(self.window_s * 10)
        active = {}
        for fr in range(f0, f1):
            for (cls, src, az, el, dist) in events.get(fr, []):
                active[(cls, src)] = (cls, src, az, el, dist)

        return {
            "clip": base,
            "t": t_center,
            "C_obs": C_obs,
            "freq": freq,
            "image": image,
            "events": list(active.values()),
        }
