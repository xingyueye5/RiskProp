# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
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
        self,
        clip_len: Optional[int] = None,
        num_clips: int = 1,
        out_of_bound_opt: str = "loop",
        test_mode: bool = False,
        **kwargs
    ) -> None:
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ["loop", "repeat_last"]

    def transform(self, results: dict) -> dict:
        """Perform the CustomSampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results["total_frames"]
        assert results["fps"] in [30, 20, 10]
        frame_interval = results["fps"] // 10

        if self.test_mode:
            # use full video
            assert self.clip_len is None
            assert self.num_clips == 1
            clip_offsets = np.array([0])
            clip_inds = np.arange(0, total_frames, step=frame_interval)
        else:
            if isinstance(self.clip_len, int):
                assert self.clip_len >= 1
                clip_offset_max = total_frames - (self.clip_len - 1) * frame_interval
                clip_offset_max = max(clip_offset_max, 1)
                assert clip_offset_max > 0
                clip_offsets = np.random.randint(0, clip_offset_max, size=self.num_clips)
                clip_inds = np.arange(self.clip_len) * frame_interval
            else:
                raise ValueError("Illegal clip_len option.")

        frame_inds = np.concatenate(clip_offsets[:, None] + clip_inds[None, :])
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

        frame_inds = frame_inds + results["start_index"]
        accident_frame = results["accident_frame"]

        frame_labels = np.where(np.abs(frame_inds - accident_frame + 0.25) < frame_interval / 2, 1, 0)

        results["frame_inds"] = frame_inds.astype(np.int32)
        results["label"] = frame_labels
        results["clip_len"] = len(frame_inds)
        results["frame_interval"] = frame_interval
        results["num_clips"] = self.num_clips
        return results


@TRANSFORMS.register_module()
class CustomSampleSnippets(BaseTransform):
    """Sample snippets from the video.

    Required Keys:

        - total_frames
        - start_index

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips

    Args:
        snippet_len (int): Frames of each sampled output snippet.
        num_snippets (int): Number of snippets to be sampled. Default: 1.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
    """

    def __init__(
        self, snippet_len: int = 5, num_snippets: Optional[int] = None, test_mode: bool = False, **kwargs
    ) -> None:
        self.snippet_len = snippet_len
        self.num_snippets = num_snippets
        self.test_mode = test_mode

    def transform(self, results: dict) -> dict:
        """Perform the CustomSampleSnippets loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results["total_frames"]
        assert results["fps"] in [30, 20, 10]
        frame_interval = results["fps"] // 10

        accident_frame = results["accident_frame"]
        start_index = results["start_index"]
        accident_ind = accident_frame - start_index

        snippet_inds = np.arange(self.snippet_len) * frame_interval
        snippet_offset_max = total_frames - (self.snippet_len - 1) * frame_interval
        assert snippet_offset_max > 0

        if self.test_mode:
            # dense sample from the full video
            assert self.num_snippets is None
            num_snippets_max = int(1000 / self.snippet_len)
            if snippet_offset_max <= frame_interval * num_snippets_max:
                snippet_offsets = np.arange(0, snippet_offset_max, step=frame_interval)
            else:
                # Prevent CUDA out of memory
                snippet_offsets = np.arange(
                    max(accident_ind - frame_interval * (num_snippets_max - int(num_snippets_max / 2)), 0),
                    min(accident_ind + frame_interval * int(num_snippets_max / 2), snippet_offset_max),
                    step=frame_interval,
                )
        else:
            if isinstance(self.num_snippets, int):
                assert self.num_snippets >= 1
                snippet_offsets = np.random.randint(0, snippet_offset_max, size=self.num_snippets)
            else:
                raise ValueError("Illegal num_snippets option.")

            snippet_offset_accident = accident_ind - (self.snippet_len - 1) * frame_interval
            snippet_offsets[0] = snippet_offset_accident

        snippet_labels = (snippet_offsets + (self.snippet_len - 1) * frame_interval >= accident_ind) & (
            snippet_offsets + (self.snippet_len - 1) * frame_interval < accident_ind + frame_interval
        )

        frame_inds = np.concatenate(snippet_offsets[:, None] + snippet_inds[None, :])
        frame_inds = np.minimum(frame_inds, total_frames - 1) + start_index
        results["frame_inds"] = frame_inds.astype(np.int32)
        results["label"] = snippet_labels
        results["clip_len"] = self.snippet_len
        results["frame_interval"] = frame_interval
        results["num_clips"] = len(snippet_offsets)
        return results


@TRANSFORMS.register_module()
class SampleFramesForAnticipation(BaseTransform):
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

    def __init__(self, clip_len: Optional[int] = None, num_clips: int = 1, test_mode: bool = False, **kwargs) -> None:
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

        accident_frame = results["accident_frame"]
        start_index = results["start_index"]

        if self.test_mode:
            # sample all frames
            assert self.clip_len is None
            assert self.num_clips == 1
            clip_inds = np.arange(total_frames, step=frame_interval)
            clip_offsets = np.full(self.num_clips, start_index)
        else:
            assert isinstance(self.clip_len, int)
            assert self.clip_len >= 1
            clip_inds = np.arange(self.clip_len) * frame_interval
            clip_offsets = np.full(
                self.num_clips, min(accident_frame + 5 * frame_interval, total_frames - 1 + start_index) - clip_inds[-1]
            )

        frame_inds = np.concatenate(clip_offsets[:, None] + clip_inds[None, :])
        frame_inds = np.maximum(frame_inds, start_index)
        frame_labels = (frame_inds >= accident_frame).astype(int)

        results["frame_inds"] = frame_inds.astype(np.int32)
        results["label"] = frame_labels
        results["clip_len"] = len(clip_inds)
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
