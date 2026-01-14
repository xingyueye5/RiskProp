import os
import csv
import json
from typing import Optional, List, Dict, Any
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DashcamVideoDataset(Dataset):
    """
    Expects:
      data_root/<video_id>/frame_00001.jpg ...
    ann_file CSV rows: video_id,num_frames,collision_frame
      collision_frame: 1-indexed frame number of collision, or -1 if none

    Produces sliding windows of length seq_len. For each window, returns
    - frames tensor: [seq_len, 3, H, W]
    - labels tensor: [seq_len]  (0/1) or risk scalar per frame
    - meta: dict(video_id=..., start_frame=1-indexed)

    This is a flexible template — adapt to your dataset labeling conventions.
    """

    def __init__(self, data_root, ann_file, seq_len=16, transform=None, step=1, mode='train'):
        self.data_root = data_root
        self.seq_len = seq_len
        self.transform = transform
        self.mode = mode
        self.step = step

        # read annotations
        self.anns = {}
        with open(ann_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 0: continue
                video_id = row[0]
                num_frames = int(row[1])
                coll = int(row[2])
                self.anns[video_id] = {'num_frames': num_frames, 'collision': coll}

        # build windows
        self.items = []  # tuples (video_id, start_frame)
        for vid, info in self.anns.items():
            n = info['num_frames']
            if n < seq_len:
                continue
            # slide windows
            for start in range(1, n - seq_len + 2, self.step):
                self.items.append((vid, start))

        # default transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.items)

    def _load_frame(self, video_id, frame_idx):
        # frame_idx is 1-indexed
        frame_path = os.path.join(self.data_root, video_id, f'frame_{frame_idx:05d}.jpg')
        img = Image.open(frame_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        video_id, start = self.items[index]
        seq = []
        for i in range(start, start + self.seq_len):
            seq.append(self._load_frame(video_id, i))
        frames = torch.stack(seq, dim=0)
        # labels
        coll = self.anns[video_id]['collision']
        labels = torch.zeros(self.seq_len, dtype=torch.float32)
        if coll > 0:
            # mark collision frame in sequence if it lies within window
            for i in range(self.seq_len):
                frame_idx = start + i
                if frame_idx >= coll:
                    # Simple labeling strategy: frames at and after collision labeled 1
                    # You may adapt to paper's labeling (e.g., only collision frame=1)
                    labels[i] = 1.0
        meta = {'video_id': video_id, 'start_frame': start}
        return frames, labels, meta


class MetaTableDataset(Dataset):
    """
    Dataset that mimics the mmaction data dict for MM-AU and Nexar without using mmaction.

    Expects a CSV file with headers matching keys used in taa/datasets.py outputs:
      dataset,video_id,frame_dir,filename_tmpl,start_index,target,abnormal_start_frame,
      accident_frame,total_frames,fps,is_val,is_test

    - frame_dir: absolute or relative path to directory containing frames
    - filename_tmpl: e.g., "img_{:05}.jpg" or "frame_{:05d}.jpg"
    - start_index: usually 0 or 1 depending on filename indexing convention
    - target: "True"/"False"/"None" (for test where labels are unknown)
    - abnormal_start_frame, accident_frame: integers or empty for non-target/test
    - fps: 10/20/30 per dataset conventions

    This dataset will produce a fixed number of clips per sample similar to SampleFramesBeforeAccident:
      - seq_len: clip_len, repeated num_clips times into a contiguous sequence [num_clips*clip_len]
      - For target=True (positive), the last frame aligns with accident_frame during test mode.
      - For target=False, windows are placed before the accident when provided; otherwise from start.
    """

    def __init__(self,
                 meta_csv: str,
                 clip_len: int = 1,
                 num_clips: int = 50,
                 resize_hw: tuple = (224, 224),
                 mode: str = 'train') -> None:
        self.meta: List[Dict[str, Any]] = []
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.mode = mode
        self.resize_hw = resize_hw

        with open(meta_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # normalize and cast types
                row['start_index'] = int(row.get('start_index', 0))
                row['total_frames'] = int(row['total_frames'])
                row['fps'] = int(row.get('fps', 10))
                row['is_val'] = row.get('is_val', 'False') in ['True', 'true', '1', 'yes']
                row['is_test'] = row.get('is_test', 'False') in ['True', 'true', '1', 'yes']
                t_str = row.get('target', '')
                if t_str in ['True', 'true', '1', 'yes']:
                    row['target'] = True
                elif t_str in ['False', 'false', '0', 'no']:
                    row['target'] = False
                else:
                    row['target'] = None
                for key in ['abnormal_start_frame', 'accident_frame']:
                    val = row.get(key, '')
                    row[key] = int(val) if str(val).strip() != '' else None
                self.meta.append(row)

        self.transform = transforms.Compose([
            transforms.Resize(resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # build index items (one item per meta row)
        self.items = list(range(len(self.meta)))

    def __len__(self):
        return len(self.items)

    def _frame_path(self, frame_dir: str, filename_tmpl: str, idx: int) -> str:
        # Support both {:05} and %05d styles
        try:
            filename = filename_tmpl.format(idx)
        except Exception:
            if '{' not in filename_tmpl and '%' in filename_tmpl:
                filename = filename_tmpl % idx
            else:
                # common defaults
                filename = f'frame_{idx:05d}.jpg'
        return os.path.join(frame_dir, filename)

    def _load_frames_sequence(self, info: Dict[str, Any]) -> torch.Tensor:
        total = info['total_frames']
        start_index = info['start_index']
        accident_frame = info.get('accident_frame', None)
        target = info.get('target', None)
        fps = info['fps']
        frame_interval = max(fps // 10, 1)  # align to 10 Hz if possible

        # Build clip indices like SampleFramesBeforeAccident
        clip_inds = np.concatenate(np.arange(self.num_clips)[:, None] + np.arange(self.clip_len)[None, :])
        clip_inds = clip_inds * frame_interval
        clip_inds_max = clip_inds[-1]

        if self.mode in ['val', 'test']:
            if target is True and accident_frame is not None:
                base = accident_frame - start_index - clip_inds_max
            elif target is False:
                if accident_frame is None:
                    base = max(total - 1 - clip_inds_max, 0)
                else:
                    base = max(accident_frame - start_index - frame_interval * 30 - clip_inds_max, 0)
            else:
                base = max(total - 1 - clip_inds_max, 0)
        else:
            if target is True and accident_frame is not None:
                # jitter around accident frame within one interval
                accident_ind = accident_frame - start_index + int(np.random.randint(0, frame_interval))
                base = min(accident_ind, total - 1) - clip_inds_max
            elif target is False:
                if accident_frame is None:
                    base = int(np.random.randint(0, max(total - clip_inds_max, 1)))
                else:
                    hi = max(accident_frame - start_index - frame_interval * 30 - clip_inds_max, 0)
                    base = int(np.random.randint(0, hi + 1)) if hi > 0 else 0
            else:
                base = max(total - 1 - clip_inds_max, 0)

        clip_inds = clip_inds + base
        frame_inds = (clip_inds + start_index).astype('int32')
        frame_inds = frame_inds.clip(min=start_index, max=start_index + total - 1)

        imgs = []
        for idx in frame_inds.reshape(-1):
            path = self._frame_path(info['frame_dir'], info['filename_tmpl'], idx)
            img = Image.open(path).convert('RGB')
            imgs.append(self.transform(img))
        frames = torch.stack(imgs, dim=0)  # [num_clips*clip_len, 3, H, W]
        return frames, frame_inds

    def __getitem__(self, index):
        info = self.meta[self.items[index]]
        frames, frame_inds = self._load_frames_sequence(info)
        # Build labels like earlier: frames >= accident are positive
        labels = torch.zeros(frames.shape[0], dtype=torch.float32)
        coll = info.get('accident_frame', None)
        if coll is not None and info.get('target', None) is True:
            for i in range(labels.numel()):
                frame_idx = frame_inds.reshape(-1)[i]
                if int(frame_idx) >= int(coll):
                    labels[i] = 1.0
        meta = {
            'video_id': info['video_id'],
            'dataset': info.get('dataset', ''),
            'frame_dir': info['frame_dir'],
            'filename_tmpl': info['filename_tmpl'],
            'type': info.get('type', None),
            'frame_inds': frame_inds,
            'start_index': info['start_index'],
            'total_frames': info['total_frames'],
            'abnormal_start_frame': info.get('abnormal_start_frame', None),
            'accident_frame': info.get('accident_frame', None),
            'is_val': info.get('is_val', False),
            'is_test': info.get('is_test', False),
            'target': info.get('target', None),
            'fps': info.get('fps', 10),
        }
        return frames, labels, meta