from __future__ import print_function, division
import argparse
import importlib
import logging
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import core.stereo_datasets as datasets

from torch.cuda.amp import GradScaler, autocast

MODEL_REGISTRY = {
    'base': 'core.pscvstereo_base',
    'tiny': 'core.pscvstereo_tiny',
}


def build_model(args):
    module = importlib.import_module(MODEL_REGISTRY[args.model])
    cls_name = 'PSCVStereoBase' if args.model == 'base' else 'PSCVStereoTiny'
    return getattr(module, cls_name)(max_disp=args.max_disp, iters=args.iters)


def parse_args(argv=None, default_model='base'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='pscvstereo_base',
                        help="experiment name; used for checkpoint filenames")
    parser.add_argument('--restore_ckpt',
                        help="path to .pth checkpoint to resume / fine-tune from")
    parser.add_argument('--mixed_precision', action='store_true',
                        help="enable automatic mixed-precision training")
    parser.add_argument('--model', default=default_model, choices=sorted(MODEL_REGISTRY.keys()),
                        help="PSCVStereo variant to train (base or tiny)")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=6,
                        help="training batch size")
    parser.add_argument('--train_datasets', nargs='+', default=[''],
                        help="dataset names to mix for training")
    parser.add_argument('--dataset_repeat', type=int, nargs='+', default=[1],
                        help="repeat count per dataset in --train_datasets (same order)")
    parser.add_argument('--data_root', default=None,
                        help="root directory containing dataset folders "
                             "(default: $STEREO_DATA_ROOT or <repo>/datasets)")
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="peak learning rate for OneCycleLR")
    parser.add_argument('--num_steps', type=int, default=100000,
                        help="number of optimization steps")
    parser.add_argument('--checkpoint_frequency', type=int, default=5000,
                        help="save a checkpoint every N optimization steps")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720],
                        help="random crop size as H W during training")
    parser.add_argument('--wdecay', type=float, default=.00001,
                        help="AdamW weight decay")
    parser.add_argument('--adam_beta2', type=float, default=0.99,
                        help="AdamW beta2 coefficient")
    parser.add_argument('--warmup_frac', type=float, default=0.05,
                        help="fraction of training steps used for linear LR warmup")
    parser.add_argument('--min_lr', type=float, default=0.000001,
                        help="minimum learning rate after cosine decay")
    parser.add_argument('--max_disp', type=int, default=None,
                        help="max disparity (default: 176 for tiny, 208 for base)")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1],
                        help="CUDA device ids for DataParallel")
    parser.add_argument('--rand_seed', type=int, default=1234,
                        help="random seed for torch / numpy")
    parser.add_argument('--iters', nargs='+', type=int, default=None,
                        help="update iters per pyramid level (default: 3 5 for tiny, 5 5 for base)")

    # Data augmentation
    parser.add_argument('--do_flip', default=False, choices=['h', 'v', 'hv'],
                        help="stereo flip mode: h, v, or hv; omitted means no flip")
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0.8, 1.2],
                        help="min max random resize scale before cropping")
    parser.add_argument('--max_train_disp', type=float, default=192.0,
                        help="training samples with larger max disparity are "
                             "downsampled until they fit; raise toward max_disp "
                             "to cover the top of the search range")
    parser.add_argument('--all_left_prob', type=float, default=0.05,
                        help="probability of replacing the right image with the left")
    parser.add_argument('--v_shift_prob', type=float, default=0.5,
                        help="probability of a small vertical shift between views")
    parser.add_argument('--eraser_aug_prob', type=float, default=0.5,
                        help="probability of random eraser occlusion augmentation")
    parser.add_argument('--ext_downsamp_prob', type=float, default=0.0,
                        help="probability of extra downsample-then-upsample augmentation")
    args = parser.parse_args(argv)
    if args.max_disp is None:
        args.max_disp = 176 if args.model == 'tiny' else 208
    if args.iters is None:
        args.iters = [3, 5] if args.model == 'tiny' else [5, 5]
    return args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sequence_loss(flow_preds, flow_gt, valid, check_finite=False):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0
    max_valid_each_pix_ = -torch.arange(
        valid.shape[-1], device=valid.device
    ).view(1, 1, 1, -1)
    valid = ((valid >= 0.5)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    if check_finite:
        assert torch.isfinite(flow_gt[valid.bool()]).all()

    occ_weight = torch.ones_like(flow_gt)
    occ_weight[flow_gt< max_valid_each_pix_] = 0.1
    occ_weight[:,:,0:4,:] = 0.1
    occ_weight[:,:,occ_weight.shape[-2]-4:occ_weight.shape[-2],:] = 0.1
    occ_weight[:,:,:,0:4] = 0.1
    occ_weight[:,:,:,occ_weight.shape[-1]-4:occ_weight.shape[-1]] = 0.1

    for i in range(n_predictions):
        if check_finite:
            assert torch.isfinite(flow_preds[i]).all()
        i_weight = 0.8**(n_predictions - i - 1)
        i_loss = ((flow_preds[i] - flow_gt)).abs() * occ_weight
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt.contiguous())**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.contiguous().view(-1)]

    metrics = {
        '4px': (epe < 4).float().mean().detach(),
        '2px': (epe < 2).float().mean().detach(),
        '1px': (epe < 1).float().mean().detach(),
        'epe': epe.mean().detach(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    if not 0.0 <= args.warmup_frac < 1.0:
        raise ValueError("--warmup_frac must be in [0, 1)")
    if not 0.0 < args.min_lr <= args.lr:
        raise ValueError("--min_lr must be positive and no greater than --lr")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, args.adam_beta2),
        weight_decay=args.wdecay,
        eps=1e-8,
    )

    total_steps = max(1, args.num_steps + 100)
    warmup_steps = int(args.num_steps * args.warmup_frac)
    min_lr_ratio = args.min_lr / args.lr

    def lr_factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir='runs')

    def _print_training_status(self):
        metrics = {k: self.running_loss[k] / Logger.SUM_FREQ
                   for k in sorted(self.running_loss.keys())}
        lr = self.scheduler.get_last_lr()[0]
        fields = [f"Step={self.total_steps}", f"lr={lr:.6g}"]
        fields += [f"{k}={v:.4f}" for k, v in metrics.items()]
        logging.info("\t".join(fields))

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k, v in metrics.items():
            self.writer.add_scalar(k, v, self.total_steps)
            self.running_loss[k] = 0.0
        self.writer.add_scalar(
            "learning_rate", self.scheduler.get_last_lr()[0], self.total_steps
        )

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]
        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    device = args.gpus[0]
    torch.backends.cudnn.benchmark = True  # auto-tune conv kernels for fixed input sizes
    logging.info(f"Building model {args.model} from {MODEL_REGISTRY[args.model]}")
    model = nn.DataParallel(build_model(args), device_ids=args.gpus)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt, map_location=torch.device('cpu'))

        # Diagnostic: print per-channel 1x1 weights (BN, depthwise convs, etc.)
        for key, value in checkpoint.items():
            if (key.split('.')[-1] == 'weight'
                    and len(value.shape) > 1
                    and value.shape[-1] == 1 and value.shape[-2] == 1):
                print(key, value.shape)

        model.load_state_dict(checkpoint, strict=False)
        logging.info("Done loading checkpoint")

    print("Parameter Count: %d" % count_parameters(model))

    optimizer, scheduler = fetch_optimizer(args, model)
    logger = Logger(model, scheduler)
    model.cuda(device)

    train_loader = datasets.fetch_dataloader(args)
    scaler = GradScaler(enabled=args.mixed_precision)

    model.train()

    should_keep_training = True
    total_steps = 0
    global_batch_num = 0
    pbar = tqdm(total=args.num_steps, desc="train", unit="step", dynamic_ncols=True)

    while should_keep_training:
        for i_batch, (_, *data_blob) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            image1, image2, flow, valid = [
                x.cuda(device, non_blocking=True) for x in data_blob
            ]

            assert model.training

            with autocast(enabled=args.mixed_precision):
                flow_predictions, _ = model(image1, image2)
                assert model.training
                loss, metrics = sequence_loss(
                    flow_predictions,
                    flow,
                    valid,
                    check_finite=global_batch_num % Logger.SUM_FREQ == 0,
                )

            metrics["live_loss"] = loss.detach()
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # GradScaler may skip optimizer.step() on inf/NaN grads; only
            # advance the LR schedule when the optimizer actually stepped.
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale:
                scheduler.step()

            logger.push(metrics)

            if total_steps > 0 and total_steps % args.checkpoint_frequency == 0:
                save_path = Path('checkpoints/%d_%s.pth' % (total_steps, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)
                model.train()

            total_steps += 1
            pbar.update(1)
            if total_steps > args.num_steps:
                should_keep_training = False
                break

    pbar.close()
    print("FINISHED TRAINING")
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


def main(argv=None, default_model='base'):
    args = parse_args(argv, default_model=default_model)

    torch.manual_seed(args.rand_seed)
    np.random.seed(args.rand_seed)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints").mkdir(exist_ok=True, parents=True)

    train(args)


if __name__ == '__main__':
    main()
