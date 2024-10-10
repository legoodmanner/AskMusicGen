import torch
from torchmetrics import Metric
from models.layers import dbnProcessor
from mir_eval.beat import f_measure as beat_f1

class BeatF1MedianScore(Metric):
    def __init__(self, fps, **kwargs):
        super().__init__(**kwargs)
        self.add_state("f1_scores", default=[], dist_reduce_fx="cat")
        self.fps = fps
    
    def update(self, preds: torch.Tensor, tgt_second_seq: list[list[float]]) -> None:
        """
        preds: [N, seq, 2] or [N, 2, seq]
        tgt_second_seq: list[list[float] * N]
        """
        if preds.shape[1] == 2: 
            preds = preds.transpose(-2,-1)
        preds = preds[...,1]
        ests = dbnProcessor(preds, fps=self.fps)  # list of preds 
        for idx, e in enumerate(ests):
            f_score = beat_f1(reference_beats=tgt_second_seq[idx], estimated_beats=e)
            self.f1_scores.append(torch.tensor(f_score))

    def compute(self):
        if not self.f1_scores:
            return torch.tensor(0.0)
        return torch.mean(torch.stack(self.f1_scores))