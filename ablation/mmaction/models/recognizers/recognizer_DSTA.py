from typing import Optional, Union


from mmaction.registry import MODELS
from .base import BaseRecognizer
from mmdet.apis import init_detector, inference_detector
import mmcv

import numpy as np
import cv2
import os, sys
import os.path as osp
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

@MODELS.register_module()
class DSTARecognizer(BaseRecognizer):
    def __init__(self,
                 backbone,
                 data_preprocessor=None,
                 args=None,
                 ):
        super().__init__(backbone=backbone, data_preprocessor=data_preprocessor)
        currentDirectory = os.getcwd()
        self.device = torch.device('cuda:0')
        self.cfg_file = osp.join("/root/autodl-fs/traffic-accident-anticipation/DSTA/lib/mmdetection/", "configs/cascade_rcnn_x101_64x4d_fpn_1x_kitti2d.py")
        self.model_file = osp.join("/root/autodl-fs/traffic-accident-anticipation/DSTA/lib/mmdetection/", "work_dirs/cascade_rcnn_x101_64x4d_fpn_1x_kitti2d/latest.pth")
        self.detector = init_detector(self.cfg_file, self.model_file, device=self.device)
        # init feature extractor
        self.feat_extractor = init_feature_extractor(backbone='vgg16', device=self.device)

        self.DSTA = None


    def forward(self,
                inputs: torch.Tensor,       # [B, T, H, W, C]
                data_samples: Optional[list] = None,
                mode: str = 'tensor',
                **kwargs):

        detections, features = self.extract_feats(inputs)
        losses, all_outputs, all_hidden, all_alphas = self.DSTA(features, None, None)


        # if mode == 'predict':
        #     return self._format_predictions(score_pred, data_samples)
        # else:
        #     return score_pred

    def extract_feat(self, inputs: torch.Tensor, **kwargs):
        n_frames = inputs.shape[0]
        transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.Resize(512),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
        )
        features = np.zeros((n_frames, 20, self.feat_extractor.dim_feat), dtype=np.float32)
        detections = np.zeros((n_frames, 19, 6))  # (50 x 19 x 6)
        frame_prev = None
        for idx in range(n_frames):
            frame = inputs[idx,...]
            bbox_result = inference_detector(self.detector, frame)
            bboxes = bbox_sampling(bbox_result, nbox=19, imsize=frame.shape[:2])
            detections[idx, :, :] = bboxes
            with torch.no_grad():
                # bboxes to roi feature
                ims_roi = bbox_to_imroi(transform, bboxes, frame)
                ims_roi = ims_roi.float().to(device=self.device)
                feature_roi = self.feat_extractor(ims_roi)
                # extract image feature
                ims_frame = transform(Image.fromarray(frame))
                ims_frame = torch.unsqueeze(ims_frame, dim=0).float().to(device=self.device)
                feature_frame = self.feat_extractor(ims_frame)
                features[idx, 0, :] = np.squeeze(feature_frame.cpu().numpy()) if feature_frame.is_cuda else np.squeeze(
                    feature_frame.detach().numpy())
                features[idx, 1:, :] = np.squeeze(feature_roi.cpu().numpy()) if feature_roi.is_cuda else np.squeeze(
                    feature_roi.detach().numpy())

        return detections, features


    def _format_predictions(self, outputs, data_samples):
        """ 将输出转换为 MMAction2 的 ActionDataSample 格式 """
        data_samples[0].set_pred_score(outputs.squeeze())

        # 构造ActionDataSample
        return data_samples


def init_feature_extractor(backbone='vgg16', device=torch.device('cuda')):
    feat_extractor = None
    if backbone == 'vgg16':
        feat_extractor = VGG16()
        feat_extractor = feat_extractor.to(device=device)
        feat_extractor.eval()
    else:
        raise NotImplementedError
    return feat_extractor

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        VGG = models.vgg16(pretrained=True)
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
        self.dim_feat = 4096

    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

def bbox_sampling(bbox_result, nbox=19, imsize=None, topN=5):
    """
    imsize[0]: height
    imsize[1]: width
    """
    assert not isinstance(bbox_result, tuple)

    bboxes = np.vstack(bbox_result)  # n x 5

    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]

    labels = np.concatenate(labels)  # n

    ndet = bboxes.shape[0]

    # fix bbox
    new_boxes = []
    for box, label in zip(bboxes, labels):
        x1 = min(max(0, int(box[0])), imsize[1])
        y1 = min(max(0, int(box[1])), imsize[0])
        x2 = min(max(x1 + 1, int(box[2])), imsize[1])
        y2 = min(max(y1 + 1, int(box[3])), imsize[0])
        if (y2 - y1 + 1 > 2) and (x2 - x1 + 1 > 2):
            new_boxes.append([x1, y1, x2, y2, box[4], label])

    if len(new_boxes) == 0:  # no bboxes
        new_boxes.append([0, 0, imsize[1]-1, imsize[0]-1, 1.0, 0])
    new_boxes = np.array(new_boxes, dtype=int)

    # sampling
    n_candidate = min(topN, len(new_boxes))
    if len(new_boxes) <= nbox - n_candidate:
        indices = np.random.choice(n_candidate, nbox - len(new_boxes), replace=True)
        sampled_boxes = np.vstack((new_boxes, new_boxes[indices]))
    elif len(new_boxes) > nbox - n_candidate and len(new_boxes) <= nbox:
        indices = np.random.choice(n_candidate, nbox - len(new_boxes), replace=False)
        sampled_boxes = np.vstack((new_boxes, new_boxes[indices]))
    else:
        sampled_boxes = new_boxes[:nbox]

    return sampled_boxes

def bbox_to_imroi(transform, bboxes, image):
    imroi_data = []
    for bbox in bboxes:
        imroi = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        imroi = transform(Image.fromarray(imroi))  # (3, 224, 224), torch.Tensor
        imroi_data.append(imroi)
    imroi_data = torch.stack(imroi_data)
    return imroi_data