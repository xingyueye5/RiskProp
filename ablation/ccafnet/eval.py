import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets import DashcamVideoDataset, MetaTableDataset
from model import CCAFNetBaseline
from util import load_checkpoint, compute_ap, compute_mTTA, eval_suite_from_sequences, write_inference_csv


def evaluate(ckpt, data_root, ann_file, seq_len=16, batch_size=4, device='cpu', threshold=0.5,
             feat_dim=512, rnn_hidden=512, num_layers=1, cascade_stages=2, meta_csv=None, out_csv=None):
    if meta_csv:
        ds = MetaTableDataset(meta_csv, clip_len=1, num_clips=seq_len, mode='test')
    else:
        ds = DashcamVideoDataset(data_root, ann_file, seq_len=seq_len, step=1, mode='test')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    model = CCAFNetBaseline(pretrained=False,
                            feat_dim=feat_dim,
                            rnn_hidden=rnn_hidden,
                            num_layers=num_layers,
                            cascade_stages=cascade_stages)
    model = model.to(device)
    ck = load_checkpoint(ckpt, map_location=device)
    model.load_state_dict(ck['state_dict'])
    model.eval()

    # aggregate per-video predictions
    preds_by_vid = {}
    anns = {}
    with torch.no_grad():
        for frames, labels, meta in tqdm(loader):
            frames = frames.to(device)
            outputs = model(frames)  # [B,T]
            outputs = outputs.cpu().numpy()
            B, T = outputs.shape
            for i in range(B):
                vid = meta['video_id'][i]
                # Support both dataset types
                if 'start_frame' in meta:
                    start = int(meta['start_frame'][i])
                    num_frames = ds.anns[vid]['num_frames']
                    coll = ds.anns[vid]['collision']
                else:
                    # MetaTableDataset path
                    start_index = int(meta['start_index'][i])
                    total_frames = int(meta['total_frames'][i])
                    num_frames = total_frames
                    coll = int(meta['accident_frame'][i]) if meta['accident_frame'][i] is not None else -1
                    # frame_inds is absolute indices; align start per sequence base
                    start = int(meta['frame_inds'][i][0])
                if vid not in preds_by_vid:
                    preds_by_vid[vid] = [0.0] * num_frames
                    anns[vid] = coll
                for j in range(T):
                    idx = start + j - 1
                    # keep max score if overlapping windows
                    preds_by_vid[vid][idx] = max(preds_by_vid[vid][idx], float(outputs[i, j]))

    # flatten for AP baseline
    all_y_true = []
    all_y_score = []
    for vid, scores in preds_by_vid.items():
        if meta_csv:
            # find collision from meta
            # build a simple lookup
            pass
        coll = anns[vid]
        for i, s in enumerate(scores, start=1):
            label = 1.0 if (coll > 0 and i >= coll) else 0.0
            all_y_true.append(label)
            all_y_score.append(s)

    ap = compute_ap(all_y_true, all_y_score)
    mtta = compute_mTTA(preds_by_vid, anns, threshold=threshold)

    # sequence metrics similar to taa/metrics.py
    preds_list = []
    labels_list = []
    abnormal_inds = []
    accident_inds = []
    for vid, scores in preds_by_vid.items():
        preds_list.append(torch.tensor(scores).numpy())
        coll = anns[vid]
        accident_inds.append(coll - 1 if coll > 0 else 0)
        abnormal_inds.append(0)
        labels_list.append(True if coll > 0 else False)
    suite = eval_suite_from_sequences(preds_list, labels_list, abnormal_inds, accident_inds, fpr_max=0.1)

    if out_csv:
        write_inference_csv(preds_by_vid, anns, out_csv)

    return ap, mtta, suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--ann_file', required=True)
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--meta_csv', type=str, default=None, help='Meta CSV describing MM-AU/Nexar frames')
    parser.add_argument('--out_csv', type=str, default=None, help='Write per-video inference scores')
    parser.add_argument('--feat_dim', type=int, default=512)
    parser.add_argument('--rnn_hidden', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--cascade_stages', type=int, default=2)
    args = parser.parse_args()
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    ap, mtta, suite = evaluate(
        args.ckpt, args.data_root, args.ann_file,
        seq_len=args.seq_len, batch_size=args.batch_size, device=device,
        feat_dim=args.feat_dim, rnn_hidden=args.rnn_hidden,
        num_layers=args.num_layers, cascade_stages=args.cascade_stages,
        meta_csv=args.meta_csv, out_csv=args.out_csv
    )
    print('AP:', ap)
    print('mTTA:', mtta)
    print('Suite:', suite)

if __name__ == '__main__':
    main()
