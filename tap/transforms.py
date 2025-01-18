# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os.path as osp
from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS

from .utils import visualize_tensor_as_videos


@TRANSFORMS.register_module()
class TrimPosNegSegment(BaseTransform):
    """Trim positive and negative segments from the video.

    Args:
        clip_len (int): The length of the clip.
        p_pos (float): The probability of trimming a positive segment.
            Defaults to 0.5.
    """

    def __init__(self, clip_len: int, p_pos: float = 0.5) -> None:
        self.clip_len = clip_len
        self.p_pos = p_pos

    def transform(self, results: dict) -> dict:
        if np.random.rand() < self.p_pos:
            results["label"] = 1
            results["start_index"] = max(results["accident_frame"] - self.clip_len + 1, 1)
            results["total_frames"] = min(
                2 * self.clip_len - 1 - 3, results["total_frames"] - results["start_index"] + 1
            )
        else:
            results["label"] = 0
            results["total_frames"] = results["abnormal_start_frame"] - 1
        return results


@TRANSFORMS.register_module()
class CustomSampleFrames(BaseTransform):
    """Sample frames from the video.

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
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Defaults to 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(
        self, clip_len: int, num_clips: int = 1, out_of_bound_opt: str = "loop", test_mode: bool = False, **kwargs
    ) -> None:
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ["loop", "repeat_last"]

    def transform(self, results: dict) -> dict:
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results["total_frames"]
        assert results["fps"] in [30, 20, 10]
        frame_interval = results["fps"] // 10

        if self.clip_len == -1:
            # use full video
            assert self.num_clips == 1
            frame_inds = np.arange(0, total_frames, step=frame_interval)[None, :]
        else:
            clip_offsets = np.random.randint(
                0, max(total_frames - (self.clip_len - 1) * frame_interval, 1), size=self.num_clips
            )
            frame_inds = clip_offsets[:, None] + np.arange(self.clip_len)[None, :] * frame_interval
            if self.out_of_bound_opt == "loop":
                frame_inds = np.mod(frame_inds, total_frames)
            elif self.out_of_bound_opt == "repeat_last":
                safe_inds = frame_inds < total_frames
                unsafe_inds = 1 - safe_inds
                last_ind = np.max(safe_inds * frame_inds, axis=1)
                new_inds = safe_inds * frame_inds + (unsafe_inds.T * last_ind).T
                frame_inds = new_inds
            else:
                raise ValueError("Illegal out_of_bound option.")

        start_index = results["start_index"]
        frame_inds = np.concatenate(frame_inds) + start_index
        results["frame_inds"] = frame_inds.astype(np.int32)
        results["clip_len"] = self.clip_len
        results["frame_interval"] = frame_interval
        results["num_clips"] = self.num_clips
        return results


@TRANSFORMS.register_module()
class VisualizeInputsAsVideos(BaseTransform):
    """Visualize inputs as videos and save them to the specified directory.

    Args:
        output_dir (str): The directory to save the videos.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def transform(self, results: dict) -> dict:
        inputs = results["inputs"]
        data_samples = results["data_samples"]
        type = data_samples.type
        video_id = data_samples.video_id
        visualize_tensor_as_videos(inputs, osp.join(self.output_dir, str(type), video_id))
        return results
