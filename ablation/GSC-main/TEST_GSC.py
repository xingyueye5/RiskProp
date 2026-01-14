import argparse
import os
import os.path as osp

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmaction.registry import RUNNERS

import argparse
import os
import shutil

from mmcv import Config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler

from msp import LightningModel
from msp.utils import setup_seed, partial_state_dict


def init_accident_model(model_file, dim_feature=4096, hidden_dim=512, latent_dim=256, n_obj=19, n_frames=50, fps=10.0):
    # building model
    model = DSTA(dim_feature, hidden_dim, latent_dim,
        n_layers=1, n_obj=n_obj, n_frames=n_frames, fps=fps, with_saa=True)

    device = torch.device('cuda:0')
    model = model.to(device=device)
    model.eval()
    # load check point
    model, _, _ = load_checkpoint(model, filename=model_file, isTraining=False)
    return model

def load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar', isTraining=True):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if isTraining:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    parser.add_argument("--modelconfig", help="Train config file path.")

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'VisualizationHook is not set in the `default_hooks` field of ' \
            'config. Please set `visualization=dict(type="VisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator)
            cfg.test_evaluator.append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    return cfg


def main():
    args = parse_args()

    # mmaction config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # 在此处进行模型修改

    model_config = Config.fromfile(args.modelconfig)

    model = LightningModel(model_config)
    checkpoint_callback = ModelCheckpoint(
        filepath=f"{cfg.checkpoint_path}/{cfg.name}/{cfg.version}/"
                 f"{cfg.name}_{cfg.version}_{{epoch}}_{{avg_val_loss:.3f}}_{{ade:.3f}}_{{fde:.3f}}_{{fiou:.3f}}",
        save_last=None,
        save_top_k=-1,
        verbose=True,
        monitor='fiou',
        mode='max',
        prefix=''
    )

    lr_logger_callback = LearningRateLogger(logging_interval='step')
    profiler = SimpleProfiler() if cfg.simple_profiler else AdvancedProfiler()
    logger = TensorBoardLogger(save_dir=cfg.log_path, name=cfg.name, version=cfg.version)

    trainer = pl.Trainer(
        gpus=cfg.num_gpus,
        # distributed_backend='dp',
        max_epochs=cfg.max_epochs,
        logger=logger,
        profiler=profiler,
        callbacks=[lr_logger_callback, ],
        # gradient_clip_val=cfg.gradient_clip_val,\
        checkpoint_callback=checkpoint_callback,
        check_val_every_n_epoch=10,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        accumulate_grad_batches=cfg.batch_accumulate_size)  # 由于每个batch内只有一个样本，所以采用这样的处理方法

    model_ckpt = partial_state_dict(model, cfg.test_checkpoint)
    model.load_state_dict(model_ckpt)

    runner.model.model = model
    runner.model.trainer = trainer


    # start testing
    runner.test()


if __name__ == '__main__':
    main()
