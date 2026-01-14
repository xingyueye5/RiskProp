import argparse
import os
import os.path as osp

import torch
import yaml
from easydict import EasyDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from Driver.main_sac import set_deterministic, parse_configs
from Driver.src.enviroment import DashCamEnv
from mmaction.registry import RUNNERS

from RLlib.SAC.sac import SAC
from taa.metrics import AnticipationMetric

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

    # todo NEW SETTINGS
    # parser.add_argument('--drive_config', default="cfgs/sac_default.yml",
    #                     help='Configuration file for SAC algorithm.')
    parser.add_argument('--phase', default='train', choices=['train', 'test'],
                        help='Training or testing phase.')
    parser.add_argument('--gpu_id', type=int, default=0, metavar='N',
                        help='The ID number of GPU. Default: 0')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='The number of workers to load dataset. Default: 4')
    parser.add_argument('--baseline', default='none', choices=['random', 'all_pos', 'all_neg', 'none'],
                        help='setup baseline results for testing comparison')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--num_epoch', type=int, default=50, metavar='N',
                        help='number of epoches (default: 50)')
    parser.add_argument('--snapshot_interval', type=int, default=5, metavar='N',
                        help='The epoch interval of model snapshot (default: 5)')
    parser.add_argument('--test_epoch', type=int, default=-1,
                        help='The snapshot id of trained model for testing.')
    parser.add_argument('--output', default='./output/SAC',
                        help='Directory of the output. ')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not load pretrained weights for perception env model')
    parser.add_argument('--no_agent_ckpt', action='store_true',
                        help='Do not load pretrained agent checkpoints (SAC)')
    parser.add_argument('--fpr_max', type=float, default=0.1,
                        help='fpr_max used by AnticipationMetric')

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

    model = runner.model
    print("Model initialized, performing custom operations...")
    with open("Driver/cfgs/sac_ae_mlnet.yml", 'r') as f:
        driver_cfg = EasyDict(yaml.safe_load(f))
    driver_cfg.update(vars(args))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    driver_cfg.update(device=device)

    driver_cfg.SAC.image_shape = driver_cfg.ENV.image_shape
    driver_cfg.SAC.input_shape = driver_cfg.ENV.input_shape

    set_deterministic(driver_cfg.seed)

    # 模型初始化
    env = DashCamEnv(driver_cfg.ENV, device=driver_cfg.device)
    env.set_model(pretrained=not args.no_pretrained, weight_file=driver_cfg.ENV.env_model)   # allow training from scratch
    driver_cfg.ENV.output_shape = env.output_shape
    model.env = env
    model.cfg = driver_cfg

    # 不使用DRIVE的数据集

    # agent设置
    model.agent = SAC(driver_cfg.SAC, device=driver_cfg.device)
    ckpt_dir = os.path.join(driver_cfg.output, 'checkpoints')
    if not args.no_agent_ckpt:
        model.agent.load_models(ckpt_dir, driver_cfg)
    model.agent.set_status('eval')

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
