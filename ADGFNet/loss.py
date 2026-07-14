import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = torch.sigmoid(preds[i]).float()
                gt = gt_masks.float()
                smooth = 1
                intersection = pred * gt
                loss = (intersection.sum() + smooth) / (
                    pred.sum() + gt.sum() - intersection.sum() + smooth
                )
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = torch.sigmoid(preds).float()
            gt = gt_masks.float()
            smooth = 1
            intersection = pred * gt
            loss = (intersection.sum() + smooth) / (
                pred.sum() + gt.sum() - intersection.sum() + smooth
            )
            loss = 1 - loss.mean()
            return loss


class BCESoftIoULoss(nn.Module):
    def __init__(self, bce_weight=1.0, iou_weight=1.0):
        super(BCESoftIoULoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.softiou = SoftIoULoss()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight

    def forward(self, preds, gt_masks):
        return self.bce_weight * self.bce(
            preds, gt_masks
        ) + self.iou_weight * self.softiou(preds, gt_masks)


class FixedGradientExtractor(nn.Module):
    def __init__(self):
        super(FixedGradientExtractor, self).__init__()
        kernel_v = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        kernel_h = torch.tensor(
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("weight_h", kernel_h)
        self.register_buffer("weight_v", kernel_v)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Expected BCHW tensor, got shape {}".format(tuple(x.shape)))
        grad_h = F.conv2d(x, self.weight_h, padding=1)
        grad_v = F.conv2d(x, self.weight_v, padding=1)
        return torch.sqrt(grad_h * grad_h + grad_v * grad_v + 1e-6)


class EdgeAlignmentLoss(nn.Module):
    def __init__(self, bce_weight=1.0, iou_weight=0.0):
        super(EdgeAlignmentLoss, self).__init__()
        self.grad = FixedGradientExtractor()
        self.softiou = SoftIoULoss()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight

    def forward(self, preds, gt_masks):
        preds_prob = torch.sigmoid(preds)
        pred_edge = self.grad(preds_prob)
        gt_edge = self.grad(gt_masks)
        pred_edge = pred_edge / (pred_edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        gt_edge = gt_edge / (gt_edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        loss = self.bce_weight * F.binary_cross_entropy_with_logits(pred_edge, gt_edge)
        if self.iou_weight > 0:
            loss = loss + self.iou_weight * self.softiou(pred_edge, gt_edge)
        return loss


class StructureConsistencyLoss(nn.Module):
    def __init__(self):
        super(StructureConsistencyLoss, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, preds, gt_masks):
        preds_prob = torch.sigmoid(preds)
        pred_local = self.pool(preds_prob)
        gt_local = self.pool(gt_masks)
        pred_residual = torch.abs(preds_prob - pred_local)
        gt_residual = torch.abs(gt_masks - gt_local)
        return F.l1_loss(pred_local, gt_local) + F.l1_loss(pred_residual, gt_residual)


class CompositeSegLoss(nn.Module):
    def __init__(
        self,
        bce_weight=1.0,
        iou_weight=1.0,
        structure_weight=0.0,
        edge_weight=0.0,
        edge_iou_weight=0.0,
    ):
        super(CompositeSegLoss, self).__init__()
        self.region_loss = BCESoftIoULoss(
            bce_weight=bce_weight, iou_weight=iou_weight
        )
        self.structure_loss = (
            StructureConsistencyLoss() if structure_weight > 0 else None
        )
        self.edge_loss = (
            EdgeAlignmentLoss(bce_weight=1.0, iou_weight=edge_iou_weight)
            if edge_weight > 0
            else None
        )
        self.structure_weight = structure_weight
        self.edge_weight = edge_weight

    def forward(self, preds, gt_masks):
        loss = self.region_loss(preds, gt_masks)
        if self.structure_loss is not None:
            loss = loss + self.structure_weight * self.structure_loss(preds, gt_masks)
        if self.edge_loss is not None:
            loss = loss + self.edge_weight * self.edge_loss(preds, gt_masks)
        return loss


def build_seg_loss(loss_name):
    name = (loss_name or "soft_iou").lower()
    if name in ("soft_iou", "softiou"):
        return SoftIoULoss()
    if name in ("bce_soft_iou", "bcesoftiou", "region"):
        return BCESoftIoULoss()
    if name in ("sr_base", "small_region_base"):
        return CompositeSegLoss(
            bce_weight=1.0,
            iou_weight=1.0,
            structure_weight=0.0,
            edge_weight=0.0,
        )
    if name in ("sr_structure", "small_region_structure"):
        return CompositeSegLoss(
            bce_weight=1.0,
            iou_weight=1.0,
            structure_weight=0.2,
            edge_weight=0.0,
        )
    if name in ("sr_edge", "small_region_edge"):
        return CompositeSegLoss(
            bce_weight=1.0,
            iou_weight=1.0,
            structure_weight=0.0,
            edge_weight=0.2,
            edge_iou_weight=0.0,
        )
    if name in ("sr_full", "small_region_full"):
        return CompositeSegLoss(
            bce_weight=1.0,
            iou_weight=1.0,
            structure_weight=0.2,
            edge_weight=0.2,
            edge_iou_weight=0.0,
        )
    raise ValueError("Unknown loss_name: {}".format(loss_name))

