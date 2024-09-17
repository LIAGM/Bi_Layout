"""
@date: 2021/7/19
@description:
"""

from torch.nn import L1Loss
# from torch.nn import BCELoss  # for opening
from torch.nn import BCEWithLogitsLoss  # for opening, and without using sigmoid
from loss.led_loss import LEDLoss
from loss.grad_loss import GradLoss
from loss.boundary_loss import BoundaryLoss
from loss.object_loss import ObjectLoss, HeatmapLoss
