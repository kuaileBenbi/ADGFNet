import threading
import numpy as np
import torch

__all__ = ['SegmentationMetricTPFNFP_nIoU']

def get_niou_prec_recall_fscore(total_tp, total_fp, total_fn):
    """
    Calculate nIoU, Precision, Recall and F-score.
    """
    niou = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp + total_fn)
    prec = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fp)
    recall = 1.0 * total_tp / (np.spacing(1) + total_tp + total_fn)
    fscore = 2.0 * prec * recall / (np.spacing(1) + prec + recall)

    return niou, prec, recall, fscore

class SegmentationMetricTPFNFP_nIoU(object):
    """Computes pixAcc and nIoU metric scores
    """

    def __init__(self, nclass):
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, labels, preds):
        """
        Update evaluation results.
        Calculate TP, FP and FN per sample, and store the results.
        """
        def evaluate_worker(label, pred):
            tp, fp, fn = batch_tp_fp_fn(pred, label, self.nclass)
            with self.lock:
                self.total_tp_list.append(tp)
                self.total_fp_list.append(fp)
                self.total_fn_list.append(fn)
            return

        if isinstance(preds, torch.Tensor):
            preds = (preds.detach().numpy() > 0).astype('int64')  # P
            labels = labels.numpy().astype('int64')  # T
            evaluate_worker(labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        elif isinstance(preds, np.ndarray):
            preds = ((preds / np.max(preds)) > 0.5).astype('int64')  # P
            labels = (labels / np.max(labels)).astype('int64')  # T
            evaluate_worker(labels, preds)
        else:
            raise NotImplemented

    def get_all(self):
        """
        Return TP, FP and FN of all samples.
        """
        return self.total_tp_list, self.total_fp_list, self.total_fn_list

    def get(self):
        """
        Calculate and return nIoU, Precision, Recall and F-score.
        """
        # Calculate IoU for each sample
        niou_list = []
        for tp, fp, fn in zip(self.total_tp_list, self.total_fp_list, self.total_fn_list):
            niou = 1.0 * tp / (np.spacing(1) + tp + fp + fn)
            niou_list.append(niou)

        # Average IoU over all samples
        niou = np.mean(niou_list)
        prec, recall, fscore = get_niou_prec_recall_fscore(
            np.sum(self.total_tp_list),
            np.sum(self.total_fp_list),
            np.sum(self.total_fn_list)
        )

        return niou, prec, recall, fscore

    def reset(self):
        """
        Reset internal states.
        """
        self.total_tp_list = []  # Store TP of each sample
        self.total_fp_list = []  # Store FP of each sample
        self.total_fn_list = []  # Store FN of each sample
        return

def batch_tp_fp_fn(predict, target, nclass):
    """
    Calculate TP, FP and FN for each sample.
    """
    mini = 1
    maxi = nclass
    nbins = nclass

    intersection = predict * (predict == target)  # TP

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))

    # areas of TN FP FN
    area_tp = area_inter[0]
    area_fp = area_pred[0] - area_inter[0]
    area_fn = area_lab[0] - area_inter[0]

    return area_tp, area_fp, area_fn