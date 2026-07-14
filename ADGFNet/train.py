import argparse
import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import json

from utils.utils import seed_pytorch, get_optimizer
from utils.datasets import NUDTSIRSTSetLoader
from utils.datasets import IRSTD1KSetLoader
from evaluation.TPFNFP import SegmentationMetricTPFNFP
from evaluation.mIoU import SamplewiseSigmoidMetric
from evaluation.mIoU import mIoU
from evaluation.pd_fa import PD_FA

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=["ADGFNet"], nargs="+", help="model_name list")
parser.add_argument("--dataset_names", default=["IRSTD-1K"], nargs="+", help="dataset_name list")
parser.add_argument("--img_norm_cfg", default=None, type=dict)
parser.add_argument("--img_norm_cfg_mean", default=None, type=float)
parser.add_argument("--img_norm_cfg_std", default=None, type=float)
parser.add_argument("--dataset_dir", default="./datasets", type=str)
parser.add_argument("--batchSize", type=int, default=8)
parser.add_argument("--save", default="./log", type=str)
parser.add_argument("--resume", default=None, nargs="+")
parser.add_argument("--pretrained", default=None, nargs="+")
parser.add_argument("--nEpochs", type=int, default=400)
parser.add_argument("--optimizer_name", default="Adam", type=str)
parser.add_argument("--optimizer_settings", default={"lr": 0.001}, type=dict)
parser.add_argument("--scheduler_name", default="MultiStepLR", type=str)
parser.add_argument("--scheduler_settings", default={"step": [100, 200, 300], "gamma": 0.5}, type=dict)
parser.add_argument("--loss_name", default="sr_structure", type=str)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--intervals", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--amp", action="store_true")

global opt
opt = parser.parse_args()

## Set img_norm_cfg
if opt.img_norm_cfg_mean is not None and opt.img_norm_cfg_std is not None:
    opt.img_norm_cfg = dict()
    opt.img_norm_cfg["mean"] = opt.img_norm_cfg_mean
    opt.img_norm_cfg["std"] = opt.img_norm_cfg_std

seed_pytorch(opt.seed)


def resolve_device(device_arg):
    if device_arg in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested via --device={device_arg}, but the current PyTorch build has no CUDA support."
        )
    return device


opt.device = resolve_device(opt.device)

if opt.amp and opt.device.type != "cuda":
    print("Warning: --amp requires CUDA, disabling AMP")
    opt.amp = False

# Models unstable under AMP FP16
_AMP_INCOMPATIBLE = {"RISTDnet", "DNANet"}
if opt.amp and opt.model_names[0] in _AMP_INCOMPATIBLE:
    print(f"Warning: {opt.model_names[0]} is unstable with AMP FP16, disabling AMP (using FP32)")
    opt.amp = False


def get_experiment_tag(model_name, loss_name):
    if loss_name:
        return f"{model_name}_{loss_name}"
    return model_name


def train(dataset_name, model_name):
    # Pass dataset and model directly instead of relying on mutated globals
    if dataset_name == "NUDT-SIRST":
        dataset_dir = r'./datasets/NUDT-SIRST/'
        train_set = NUDTSIRSTSetLoader(base_dir=dataset_dir, mode='trainval')
    elif dataset_name == "IRSTD-1K":
        dataset_dir = r'./datasets/IRSTD-1K/'
        train_set = IRSTD1KSetLoader(base_dir=dataset_dir, mode='trainval')
    else:
        raise NotImplementedError

    train_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
    )

    net = Net(model_name=model_name, mode="train", loss_name=opt.loss_name).to(opt.device)
    net.train()
    scaler = torch.cuda.amp.GradScaler() if opt.amp else None

    epoch_state = 0

    opt.best_miou = 0
    opt.best_miou_epoch = 0

    opt.best_niou = 0
    opt.best_niou_epoch = 0

    opt.best_fscore = 0
    opt.best_fscore_epoch = 0

    opt.best_pd = 0
    opt.best_pd_epoch = 0
    opt.best_fa = 1
    opt.best_fa_epoch = 0

    total_loss_list = []
    total_loss_epoch = []

    # Fix: Explicitly check for both current dataset and model in checkpoints
    if opt.resume:
        for resume_pth in opt.resume:
            if dataset_name in resume_pth and model_name in resume_pth:
                ckpt = torch.load(resume_pth, map_location=opt.device)
                net.load_state_dict(ckpt["state_dict"])
                epoch_state = ckpt["epoch"]
                total_loss_list = ckpt["total_loss"]
                for i in range(len(opt.scheduler_settings["step"])):
                    opt.scheduler_settings["step"][i] = (
                            opt.scheduler_settings["step"][i] - ckpt["epoch"]
                    )
                print(f"Resumed from {resume_pth}")

    if opt.pretrained:
        for pretrained_pth in opt.pretrained:
            if dataset_name in pretrained_pth and model_name in pretrained_pth:
                # Fix: Changed resume_pth -> pretrained_pth
                ckpt = torch.load(pretrained_pth, map_location=opt.device)
                net.load_state_dict(ckpt["state_dict"])
                print(f"Loaded pretrained weights from {pretrained_pth}")

    if opt.scheduler_name == "MultiStepLR":
        opt.scheduler_settings["epochs"] = opt.nEpochs
        if not all(k in opt.scheduler_settings for k in ("epochs", "step", "gamma")):
            opt.scheduler_settings = {"epochs": opt.nEpochs, "step": [200, 300], "gamma": 0.5}

    if opt.scheduler_name == "CosineAnnealingLR":
        opt.scheduler_settings["epochs"] = opt.nEpochs
        if not all(k in opt.scheduler_settings for k in ("epochs", "min_lr")):
            opt.scheduler_settings = {"epochs": opt.nEpochs, "min_lr": 1e-3}
    print(f"scheduler_name: {opt.scheduler_name}, scheduler_settings: {opt.scheduler_settings}")

    current_nEpochs = opt.scheduler_settings["epochs"]

    if opt.device.type == "cuda" and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    optimizer, scheduler = get_optimizer(
        net,
        opt.optimizer_name,
        opt.scheduler_name,
        opt.optimizer_settings,
        opt.scheduler_settings,
    )
    model_ref = net.module if isinstance(net, torch.nn.DataParallel) else net
    exp_tag = get_experiment_tag(model_name, opt.loss_name)

    for idx_epoch in range(epoch_state, current_nEpochs):
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            # Fix: Removed Variable
            img = img.to(opt.device)
            gt_mask = gt_mask.to(opt.device)
            if img.shape[0] == 1:
                continue

            optimizer.zero_grad()
            if opt.amp:
                with torch.cuda.amp.autocast():
                    # Fix: use net(img) instead of net.forward(img)
                    pred = net(img)
                    loss = model_ref.loss(pred, gt_mask)

                # Fix: use .item() to extract float and prevent memory leak
                total_loss_epoch.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = net(img)
                loss = model_ref.loss(pred, gt_mask)
                total_loss_epoch.append(loss.item())
                loss.backward()
                optimizer.step()

        scheduler.step()
        if (idx_epoch + 1) % 1 == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print('*********' + time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,*********'
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write('*********' + time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,,*********\n'
                        % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []

            if (idx_epoch + 1) % 10 == 0 or idx_epoch > 300:
                save_pth = opt.save + '/' + dataset_name + '/' + model_name + '/' + str(
                    idx_epoch + 1) + '.pth.tar'
                test_with_save(save_pth, idx_epoch, total_loss_list, net.state_dict())


def test_with_save(save_pth, idx_epoch, total_loss_list, net_state_dict):
    if dataset_name == "NUDT-SIRST":
        dataset_dir = r'./datasets/NUDT-SIRST/'
        test_set = NUDTSIRSTSetLoader(base_dir=dataset_dir, mode='test')
    elif dataset_name == "IRSTD-1K":
        dataset_dir = r'./datasets/IRSTD-1K/'
        test_set = IRSTD1KSetLoader(base_dir=dataset_dir, mode='test')
    else:
        raise NotImplementedError
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=model_name, mode='test').cuda()
    # ckpt = torch.load(save_pth)
    net.load_state_dict(net_state_dict)
    net.eval()

    eval_mIoU = mIoU()
    nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)

    eval_PD_FA = PD_FA()
    eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)

    for idx_iter, (img, gt_mask, size, _) in enumerate(test_loader):
        with torch.no_grad():
            img = Variable(img).cuda()
            # pred = net(img)
            pred = net.forward(img)
            # If the model output is unactivated logits, it must be converted to probability values via sigmoid first
            if getattr(net.model, "outputs_logits", False):
                pred = torch.sigmoid(pred)
            pred = pred[:, :, :size[0], :size[1]]

        gt_mask = gt_mask[:, :, :size[0], :size[1]]

        eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
        # nIou指标
        nIoU_metric.update(pred, gt_mask)  # 像素
        eval_PD_FA.update(pred[0, 0, :, :].cpu().detach().numpy(), gt_mask[0, 0, :, :].detach().numpy(), size)
        # eval_my_PD_FA.update(pred[0, 0, :, :].cpu().detach().numpy(), gt_mask[0, 0, :, :].detach().numpy())
        eval_mIoU_P_R_F.update(labels=gt_mask[0, 0, :, :].detach().numpy(),
                               preds=pred[0, 0, :, :].cpu().detach().numpy())

    Yin_pixAcc, Yin_mIoU = eval_mIoU.get()
    nIoU = nIoU_metric.get()

    pd, fa = eval_PD_FA.get()
    _, _, _, fscore = eval_mIoU_P_R_F.get()

    save_checkpoint({
        'epoch': idx_epoch + 1,
        'state_dict': net.state_dict(),
        'total_loss': total_loss_list,
    }, save_pth)

    if Yin_mIoU > opt.best_miou:
        opt.best_miou = Yin_mIoU
        opt.best_miou_epoch = idx_epoch + 1

    if nIoU > opt.best_niou:
        opt.best_niou = nIoU
        opt.best_niou_epoch = idx_epoch + 1

    if fscore > opt.best_fscore:
        opt.best_fscore = fscore
        opt.best_fscore_epoch = idx_epoch + 1

    if pd > opt.best_pd:
        opt.best_pd = pd
        opt.best_pd_epoch = idx_epoch + 1

    if fa < opt.best_fa:
        opt.best_fa = fa
        opt.best_fa_epoch = idx_epoch + 1

    print('pixAcc %.6f, mIoU: %.6f, nIoU: %.6f' % (Yin_pixAcc, Yin_mIoU, nIoU))
    opt.f.write('pixAcc %.6f, mIoU: %.6f, nIoU: %.6f' % (Yin_pixAcc, Yin_mIoU, nIoU) + '\n')
    print('Pd: %.6f, Fa: %.8f, fscore: %.6f' % (pd, fa, fscore))
    opt.f.write('Pd: %.6f, Fa: %.8f, fscore: %.6f' % (pd, fa, fscore) + '\n')

    print('Best mIoU: %.6f,when Epoch=%d, Best nIoU: %.6f,when Epoch=%d, Best fscore: %.6f,when Epoch=%d' % (
        opt.best_miou, opt.best_miou_epoch, opt.best_niou, opt.best_niou_epoch, opt.best_fscore, opt.best_fscore_epoch))
    opt.f.write('Best mIoU: %.6f,when Epoch=%d, Best nIoU: %.6f,when Epoch=%d, Best fscore: %.6f,when Epoch=%d' % (
        opt.best_miou, opt.best_miou_epoch, opt.best_niou, opt.best_niou_epoch, opt.best_fscore,
        opt.best_fscore_epoch) + '\n')

    print('Best Pd: %.6f,when Epoch=%d, Best Fa: %.8f,when Epoch=%d' % (
        opt.best_pd, opt.best_pd_epoch, opt.best_fa, opt.best_fa_epoch))
    opt.f.write('Best Pd: %.6f,when Epoch=%d, Best Fa: %.8f,when Epoch=%d' % (
        opt.best_pd, opt.best_pd_epoch, opt.best_fa, opt.best_fa_epoch) + '\n')


def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
    return save_path


def save_curves(metrics_history, save_dir, experiment_tag):
    curves_dir = os.path.join(save_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)

    with open(os.path.join(curves_dir, f"{experiment_tag}_metrics.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)

    epochs = metrics_history["epoch"]
    plots = [
        ("loss", "Loss"),
        ("mIoU", "mIoU"),
        ("pixAcc", "Pixel Accuracy"),
        ("PD", "Probability of Detection"),
        ("FA", "False Alarm"),
    ]

    fig, axes = plt.subplots(1, len(plots), figsize=(5 * len(plots), 4))
    for ax, (key, title) in zip(axes, plots):
        ax.plot(epochs, metrics_history[key], "o-", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(experiment_tag, fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(curves_dir, f"{experiment_tag}_curves.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Curves saved to {save_path}")


if __name__ == "__main__":
    os.makedirs(opt.save, exist_ok=True)

    for dataset_name in opt.dataset_names:
        for model_name in opt.model_names:
            exp_tag = get_experiment_tag(model_name, opt.loss_name)
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            opt.f = open(opt.save + '/' + dataset_name + '_' + model_name + '_' +
                         (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')

            print(f"Starting {dataset_name}\t{exp_tag}")
            train(dataset_name, model_name)
            print("\n")
            opt.f.close()
