# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

from mmengine.fileio import exists

from mmaction.registry import DATASETS
from mmengine.dataset import Compose, BaseDataset

import pandas as pd
import numpy as np

from .utils import get_fps
from .splits import cap_test, dada_test, nexar_val


@DATASETS.register_module()
class MultiDataset(BaseDataset):
    def __init__(
        self,
        cap=None,
        dada=None,
        d2city=None,
        nexar=None,
        pipeline_video=None,
        pipeline_frame=None,
        modality="rgb",
        test_mode=False,
        train_with_val=False,
        val_train=False,
        indices=None,
    ) -> None:
        self.cap = cap
        self.dada = dada
        self.d2city = d2city
        self.nexar = nexar

        self.modality = modality
        assert self.modality in ["rgb", "flow", "both", "two_stream"], f"modality {self.modality} is not supported"
        self.test_mode = test_mode
        self.train_with_val = train_with_val
        self.val_train = val_train
        self._indices = indices
        self._metainfo = self._load_metainfo(None)
        self.serialize_data = True
        self.max_refetch = 1000

        self.pipeline_video = Compose(pipeline_video)
        self.pipeline_frame = Compose(pipeline_frame)

        self.full_init()

    def load_data_list(self):
        data_list = []
        if self.cap:
            fin = pd.read_excel(osp.join(self.cap["data_root"], self.cap["ann_file"]), sheet_name=1).values.tolist()
            for line in fin:
                video_id = str(line[0]).zfill(6)
                if 1 <= line[5] <= 10:
                    frame_dir = "1-10"
                elif line[5] == 11:
                    frame_dir = "11"
                elif 12 <= line[5] <= 42:
                    frame_dir = "12-42"
                elif line[5] == 43:
                    frame_dir = "43"
                elif 44 <= line[5] <= 62:
                    frame_dir = "44-62"
                frame_dir = osp.join(self.cap["data_root"], frame_dir, str(line[5]), video_id, "images")
                fps = get_fps(video_id, dataset_type="cap")

                # skip the broken videos
                if video_id in ["011665", "008728", "004928", "007155"]:
                    continue

                # only for ego-car accidents
                if not 1 <= line[5] <= 18:
                    continue

                # keep the train videos
                if not self.test_mode and not self.train_with_val and video_id in cap_test.keys():
                    continue

                # keep the test videos
                if self.test_mode and not self.val_train and video_id not in cap_test.keys():
                    continue

                correct_total_frames = {
                    "005722": 78,
                    "009073": 211,
                    "009074": 220,
                    "009480": 586,
                    "010087": 301,
                    "009541": 453,
                    "009222": 303,
                    "009255": 217,
                    "008589": 106,
                }

                # fix the total frames
                if video_id in correct_total_frames.keys():
                    line[10] = correct_total_frames[video_id]

                if line[9] - self.cap["start_index"] >= fps * 2:
                    data_list.append(
                        dict(
                            dataset="cap",
                            filename=None,
                            frame_dir=frame_dir,
                            filename_tmpl=self.cap["filename_tmpl"],
                            start_index=self.cap["start_index"],
                            video_id=video_id,
                            type=line[5],
                            target=True,
                            abnormal_start_frame=line[7],
                            accident_frame=line[9],
                            total_frames=line[10],
                            fps=fps,
                            is_val=video_id in cap_test.keys(),
                            is_test=False,
                        )
                    )

                if line[9] - self.cap["start_index"] >= fps * 3.5:
                    data_list.append(
                        dict(
                            dataset="cap",
                            filename=None,
                            frame_dir=frame_dir,
                            filename_tmpl=self.cap["filename_tmpl"],
                            start_index=self.cap["start_index"],
                            video_id=video_id,
                            type=line[5],
                            target=False,
                            abnormal_start_frame=line[7],
                            accident_frame=line[9],
                            total_frames=line[10],
                            fps=fps,
                            is_val=video_id in cap_test.keys(),
                            is_test=False,
                        )
                    )

        if self.dada:
            fin = pd.read_excel(osp.join(self.dada["data_root"], self.dada["ann_file"]), sheet_name=1).values.tolist()
            for line in fin:
                video_id = str(line[0]).zfill(3)
                frame_dir = osp.join(self.dada["data_root"], str(line[5]), video_id, "images")
                video_id = f"{line[5]}_{video_id}"
                fps = get_fps(video_id, dataset_type="dada")

                # skip the videos without accidents
                if line[8] == -1:
                    continue

                # only for ego-car accidents
                if not 1 <= line[5] <= 18:
                    continue

                # keep the train videos
                if not self.test_mode and not self.train_with_val and video_id in dada_test.keys():
                    continue

                # keep the test videos
                if self.test_mode and not self.val_train and video_id not in dada_test.keys():
                    continue

                correct_total_frames = {
                    "5_040": 382,
                    "5_049": 204,
                    "6_009": 325,
                    "10_169": 320,
                    "11_076": 666,
                    "11_139": 220,
                    "36_002": 342,
                    "37_003": 695,
                    "43_080": 338,
                    "43_188": 220,
                    "48_065": 330,
                    "50_136": 422,
                    "56_008": 433,
                }

                # fix the total frames
                if video_id in correct_total_frames.keys():
                    line[10] = correct_total_frames[video_id]

                if line[8] - self.dada["start_index"] >= fps * 2:
                    data_list.append(
                        dict(
                            dataset="dada",
                            filename=None,
                            frame_dir=frame_dir,
                            filename_tmpl=self.dada["filename_tmpl"],
                            start_index=self.dada["start_index"],
                            video_id=video_id,
                            type=line[5],
                            target=True,
                            abnormal_start_frame=line[7],
                            accident_frame=line[8],
                            total_frames=line[10],
                            fps=fps,
                            is_val=video_id in dada_test.keys(),
                            is_test=False,
                        )
                    )

                if line[8] - self.dada["start_index"] >= fps * 3.5:
                    data_list.append(
                        dict(
                            dataset="dada",
                            filename=None,
                            frame_dir=frame_dir,
                            filename_tmpl=self.dada["filename_tmpl"],
                            start_index=self.dada["start_index"],
                            video_id=video_id,
                            type=line[5],
                            target=False,
                            abnormal_start_frame=line[7],
                            accident_frame=line[8],
                            total_frames=line[10],
                            fps=fps,
                            is_val=video_id in dada_test.keys(),
                            is_test=False,
                        )
                    )

        if self.d2city:
            fin = pd.read_csv(osp.join(self.d2city["data_root"], self.d2city["ann_file"])).values.tolist()
            for line in fin:
                video_id = line[0]
                filename = osp.join(self.d2city["data_root"], "raw", str(int(line[1])).zfill(4), video_id + ".mp4")
                fps = 20 if np.random.rand() < 0.5 else 30

                # keep the train videos
                if self.test_mode:
                    continue

                if np.random.rand() < 0.8:
                    continue

                data_list.append(
                    dict(
                        dataset="d2city",
                        filename=filename,
                        frame_dir=None,
                        filename_tmpl=None,
                        start_index=0,
                        video_id=video_id,
                        type=None,
                        target=False,
                        abnormal_start_frame=None,
                        accident_frame=None,
                        total_frames=int(line[2]),
                        fps=fps,
                        is_val=False,
                        is_test=False,
                    )
                )

        if self.nexar:
            fin = pd.read_csv(osp.join(self.nexar["data_root"], self.nexar["ann_file"])).values.tolist()
            for line in fin:
                video_id = str(int(line[0])).zfill(5)
                is_test = bool(line[1])
                target = bool(line[6]) if not is_test else None
                if not is_test:
                    filename = "train"
                    frame_dir = "train_raw_frames"
                else:
                    filename = "test"
                    frame_dir = "test_raw_frames"
                filename = osp.join(self.nexar["data_root"], filename, video_id + ".mp4")
                frame_dir = osp.join(self.nexar["data_root"], frame_dir, video_id)
                fps = 30

                # keep the train videos
                if not self.test_mode and is_test:
                    continue

                if not self.test_mode and not self.train_with_val and video_id in nexar_val:
                    continue

                # keep the test videos
                if self.test_mode and not self.val_train and video_id not in nexar_val and not is_test:
                    continue

                data_list.append(
                    dict(
                        dataset="nexar",
                        filename=filename,
                        frame_dir=frame_dir,
                        filename_tmpl=self.nexar["filename_tmpl"],
                        start_index=self.nexar["start_index"],
                        video_id=video_id,
                        type=None,
                        target=target,
                        abnormal_start_frame=int(line[4]) if not is_test and target else None,
                        accident_frame=int(line[5]) if not is_test and target else None,
                        total_frames=int(line[2]),
                        fps=fps,
                        is_val=video_id in nexar_val,
                        is_test=is_test,
                    )
                )
        return data_list

    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        if data_info["dataset"] in ["d2city"]:
            pipeline = self.pipeline_video
        else:
            pipeline = self.pipeline_frame
        if self.modality == "two_stream":
            data_info["flow"] = False
            data_info_flow = None
            for t in pipeline.transforms:
                if t.__class__.__name__ in ["RandomResizedCrop", "Resize", "Flip", "Flow"]:
                    if data_info_flow is None:
                        data_info_flow = copy.deepcopy(data_info)
                        data_info_flow["flow"] = True
                    data_info = t(data_info)
                    data_info_flow = t(data_info_flow)
                    if t.__class__.__name__ == "Flow":
                        data_info["imgs"] = [
                            np.concatenate([frame, flow], axis=-2) for frame, flow in zip(data_info["imgs"], data_info_flow["imgs"])
                        ]
                        del data_info_flow
                else:
                    data_info = t(data_info)
            return data_info
        else:
            return pipeline(data_info)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info["modality"] = "RGB"
        return data_info
