import argparse
import time
import torch
from torch.utils.data import DataLoader
import os
from net import Net
from evaluation.mIoU import mIoU
from evaluation.pd_fa import PD_FA
from evaluation.TPFNFP import SegmentationMetricTPFNFP
from evaluation.mIoU import SamplewiseSigmoidMetric

from utils.datasets import (
    NUDTSIRSTSetLoader,
    IRSTD1KSetLoader,
)


parser = argparse.ArgumentParser(description="Test trained model on datasets")
parser.add_argument("--model_names", type=str, default=['ADGFNet', 'ADGFNetLite'],
                    help="Model name")
parser.add_argument("--dataset_names", nargs='+', default=['NUDT-SIRST', 'IRSTD-1K'],
                    choices=['NUDT-SIRST', 'IRSTD-1K'],
                    help="List of dataset names to test")
parser.add_argument("--patchSize", type=int, default=256, help="Testing patch size")
parser.add_argument("--dataset_dir", type=str, default='./datasets', help="Root directory of datasets")
parser.add_argument("--pth_dirs", type=str, default=['NUDT-SIRST/ADGFNet.pth.tar',
                                                     'NUDT-SIRST/ADGFNetLite.pth.tar',
                                                     'IRSTD-1K/ADGFNet.pth.tar',
                                                     'IRSTD-1K/ADGFNetLite.pth.tar'],
                    help="Path to trained model weights")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary prediction")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
parser.add_argument("--save_log", type=str, default='./checkpoints/', help="path of saved .pth")
global opt
opt = parser.parse_args()


def load_dataset(dataset_name, dataset_dir, mode='test'):
    if dataset_name == "NUDT-SIRST":
        return NUDTSIRSTSetLoader(base_dir=os.path.join(dataset_dir, 'NUDT-SIRST'), mode=mode)
    elif dataset_name == "IRSTD-1K":
        return IRSTD1KSetLoader(base_dir=os.path.join(dataset_dir, 'IRSTD-1K'), mode=mode)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def test_model():
    model = Net(model_name=opt.model_name, mode='test', patch_size=opt.patchSize).cuda()

    state_dict = torch.load(opt.pth_dir, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)

    model.eval()

    print(f"\nTesting model: {opt.model_name}")
    print(f"Testing on dataset: {opt.test_dataset_name}")
    test_set = load_dataset(opt.test_dataset_name, opt.dataset_dir, mode='test')
    test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    eval_mIoU = mIoU()
    nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)
    eval_PD_FA = PD_FA()
    eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)

    for img, gt_mask, size, _ in test_loader:
        with torch.no_grad():
            img = img.cuda()
            pred = model(img)
            if getattr(model.model, "outputs_logits", False):
                pred = torch.sigmoid(pred)
            pred = pred[:, :, :size[0], :size[1]]

        gt_mask = gt_mask[:, :, :size[0], :size[1]]
        # Update metrics
        eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
        nIoU_metric.update(pred, gt_mask)
        eval_PD_FA.update(pred[0, 0, :, :].cpu().numpy(), gt_mask[0, 0, :, :].numpy(), size)
        eval_mIoU_P_R_F.update(gt_mask[0, 0, :, :].numpy(), pred[0, 0, :, :].cpu().numpy())

    # Calculate final metrics
    pixAcc, mIoU_val = eval_mIoU.get()
    nIoU = nIoU_metric.get()
    pd, fa = eval_PD_FA.get()
    _, _, _, fscore = eval_mIoU_P_R_F.get()

    # Print results
    print(f"Results for {opt.test_dataset_name}:")
    print(f"Pixel Accuracy: {pixAcc:.4f}, mIoU: {mIoU_val:.4f}, nIoU: {nIoU:.4f}")
    print(f"F-score: {fscore:.4f}")
    print(f"Pd: {pd:.4f}, Fa: {fa * 1e6:.2f}E-6")
    opt.f.write(f"Results for {opt.test_dataset_name}:\n")
    opt.f.write(f"Pixel Accuracy: {pixAcc:.4f}, mIoU: {mIoU_val:.4f}, nIoU: {nIoU:.4f}\n")
    opt.f.write(f"F-score: {fscore:.4f}\n")
    opt.f.write(f"Pd: {pd:.4f}, Fa: {fa * 1e6:.2f}E-6\n")


if __name__ == "__main__":
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    for model_name in opt.model_names:
        for dataset_name in opt.dataset_names:
            for pth_dir in opt.pth_dirs:
                pth_dataset = pth_dir.split('/')[0]
                pth_model = os.path.splitext(os.path.splitext(os.path.basename(pth_dir))[0])[0]
                if dataset_name == pth_dataset and model_name == pth_model:
                    opt.test_dataset_name = dataset_name
                    opt.model_name = model_name
                    opt.train_dataset_name = pth_dir.split('/')[0]
                    print(pth_dir)
                    opt.f.write(pth_dir)
                    opt.f.write('\n')
                    print(opt.test_dataset_name)
                    opt.f.write(opt.test_dataset_name)
                    opt.f.write('\n')
                    opt.pth_dir = opt.save_log + pth_dir
                    test_model()
                    print('\n')
                    opt.f.write('\n')
    opt.f.close()
