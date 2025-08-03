import torch
from torchmetrics import Metric

from mld.utils.temos_utils import remove_padding
from .utils import calculate_mpjpe


class PosMetrics(Metric):

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "MPJPE"

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("mpjpe_sum", default=torch.tensor(0.), dist_reduce_fx="sum")

    def compute(self) -> dict:
        metric = dict(MPJPE=self.mpjpe_sum / self.count)
        return metric

    def update(self, joints_ref: torch.Tensor,
               joints_rst: torch.Tensor,
               lengths: list[int]) -> None:
        self.count += sum(lengths)
        joints_rst = remove_padding(joints_rst, lengths)
        joints_ref = remove_padding(joints_ref, lengths)
        for j1, j2 in zip(joints_ref, joints_rst):
            mpjpe = torch.sum(calculate_mpjpe(j1, j2))
            self.mpjpe_sum += mpjpe


class KeyPosMetrics(Metric):

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "MPJPE"

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("mpjpe_sum", default=torch.tensor(0.), dist_reduce_fx="sum")

    def compute(self) -> dict:
        metric = dict(MPJPE=self.mpjpe_sum / self.count)
        return metric

    def update(self, joints_ref: torch.Tensor,
               joints_rst: torch.Tensor,
               joints_mask: torch.Tensor,
               lengths: list[int]) -> None:
        self.count += torch.sum(joints_mask == 1).item()

        joints_mask = joints_mask.unsqueeze(-1)
        
        joints_rst = joints_rst * joints_mask
        joints_ref = joints_ref * joints_mask

        joints_rst = remove_padding(joints_rst, lengths)
        joints_ref = remove_padding(joints_ref, lengths)

        for j1, j2 in zip(joints_ref, joints_rst):
            mpjpe = torch.sum(calculate_mpjpe(j1, j2))
            self.mpjpe_sum += mpjpe


class KeyPosMetrics2D(Metric):

    def __init__(self, dist_sync_on_step: bool = True) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = "MPJPE"

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("mpjpe_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("mpjpe_2d_sum", default=torch.tensor(0.), dist_reduce_fx="sum")

    def compute(self) -> dict:
        # metric = dict(MPJPE=self.mpjpe_sum / self.count)
        metrics = dict()

        metrics["mpjpe_sum"] = self.mpjpe_sum / self.count
        metrics["mpjpe_2d_sum"] = self.mpjpe_2d_sum / self.count

        return metrics

    def update(self, joints_ref: torch.Tensor,
               joints_rst: torch.Tensor,
               
               joints_ref_2d: torch.Tensor,
               joints_rst_2d: torch.Tensor,

               joints_mask: torch.Tensor,
               lengths: list[int]) -> None:
               
        self.count += torch.sum(joints_mask == 1).item()

        joints_mask = joints_mask.unsqueeze(-1)
        
        joints_rst = joints_rst * joints_mask
        joints_ref = joints_ref * joints_mask

        joints_rst = remove_padding(joints_rst, lengths)
        joints_ref = remove_padding(joints_ref, lengths)

        
        joints_rst_2d = joints_rst_2d[...,:2] * joints_mask[...,:2]
        joints_ref_2d = joints_ref_2d[...,:2] * joints_mask[...,:2]
        

        joints_rst_2d = remove_padding(joints_rst_2d, lengths)
        joints_ref_2d = remove_padding(joints_ref_2d, lengths)

        
        for j1, j2 in zip(joints_ref, joints_rst):
            mpjpe = torch.sum(calculate_mpjpe(j1, j2))
            self.mpjpe_sum += mpjpe

        
        for j1, j2 in zip(joints_ref_2d, joints_rst_2d):
            mpjpe_2d = torch.sum(calculate_mpjpe(j1, j2))
            self.mpjpe_2d_sum += mpjpe_2d