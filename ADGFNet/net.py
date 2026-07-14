import os
from loss import *
from model import *


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Net(nn.Module):
    def __init__(self, model_name, mode, loss_name=None, patch_size=None):
        super(Net, self).__init__()
        self.model_name = model_name
        self.loss_name = loss_name

        self.cal_loss = SoftIoULoss()
        if model_name == "ADGFNet":
            self.model = ADGFNet()
            self.cal_loss = build_seg_loss(loss_name or "sr_base")
        elif model_name == "ADGFNetLite":
            self.model = ADGFNetLite()
            self.cal_loss = build_seg_loss(loss_name or "sr_base")
        else:
            raise ValueError("Unknown model_name: {}".format(model_name))

        if loss_name is not None:
            self.cal_loss = build_seg_loss(loss_name)

    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
