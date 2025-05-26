# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SampleFramesBeforeAccident(BaseTransform):
    """Sample frames for anticipation task.

    Required Keys:

        - total_frames
        - start_index

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(self, clip_len: int = 1, num_clips: int = 50, test_mode: bool = False, **kwargs) -> None:
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

    def transform(self, results: dict) -> dict:
        """Perform the SampleFramesBeforeAccident loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results["total_frames"]
        assert results["fps"] in [30, 20, 10]
        frame_interval = results["fps"] // 10

        abnormal_start_frame = results["abnormal_start_frame"]
        accident_frame = results["accident_frame"]
        start_index = results["start_index"]

        clip_inds = np.concatenate(np.arange(self.num_clips)[:, None] + np.arange(self.clip_len)[None, :])
        clip_inds *= frame_interval
        clip_inds_max = clip_inds[-1]

        if self.test_mode:
            if results["target"] is True:
                clip_inds += accident_frame - start_index - clip_inds_max
            elif results["target"] is False:
                if abnormal_start_frame is None:
                    if total_frames > clip_inds_max:
                        clip_inds += 0
                    else:
                        clip_inds += total_frames - 1 - clip_inds_max
                else:
                    if abnormal_start_frame - start_index > clip_inds_max:
                        clip_inds += 0
                    else:
                        clip_inds += abnormal_start_frame - start_index - 1 - clip_inds_max
            elif results["target"] is None:
                clip_inds += total_frames - 1 - clip_inds_max
        else:
            if results["target"] is True:
                accident_ind = accident_frame - start_index + np.random.randint(0, frame_interval)
                clip_inds += min(accident_ind, total_frames - 1) - clip_inds_max
            elif results["target"] is False:
                if abnormal_start_frame is None:
                    if total_frames > clip_inds_max:
                        clip_inds += np.random.randint(0, total_frames - clip_inds_max)
                    else:
                        clip_inds += total_frames - 1 - clip_inds_max
                else:
                    if abnormal_start_frame - start_index > clip_inds_max:
                        clip_inds += np.random.randint(0, abnormal_start_frame - start_index - clip_inds_max)
                    else:
                        clip_inds += abnormal_start_frame - start_index - 1 - clip_inds_max

        frame_inds = np.maximum(clip_inds, 0) + start_index

        results["frame_inds"] = frame_inds.astype(np.int32)
        results["frame_interval"] = frame_interval
        results["clip_len"] = self.clip_len
        results["num_clips"] = self.num_clips
        return results


@TRANSFORMS.register_module()
class Flow(BaseTransform):
    def __init__(self, modality: str = "rgb") -> None:
        self.modality = modality
        assert self.modality in ["rgb", "flow"]

    def transform(self, results: dict) -> dict:
        if self.modality == "flow":
            imgs = []
            for i in range(results["num_clips"]):
                frames = results["imgs"][i * results["clip_len"] : (i + 1) * results["clip_len"]]
                if results["clip_len"] == 1:
                    prev_frame = results["imgs"][max(i - 1, 0)]
                    flows = calculate_optical_flow(frames, prev_frame)
                else:
                    flows = calculate_optical_flow(frames)
                imgs += flows
            results["imgs"] = imgs
        return results


def calculate_optical_flow(frames, prev_frame=None):
    """Calculate dense optical flow between consecutive frames"""
    flows = []
    if prev_frame is None:
        prev_frame = frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(np.dstack([flow, np.linalg.norm(flow, axis=2)]))
        prev_gray = gray

    return flows
