import torch
import numpy as np
from torchmetrics import Metric
from models.layers import dbnProcessor
from mir_eval.beat import f_measure as beat_f1

def get_metric_from_task(config):
    task2metric = {
        "GTZAN_rhythm": BeatFMeasure,
        "GS_key": KeyAccRefined,
        "GTZAN_genre": torchmetrics.Accuracy,
        "MTG_genre": torchmetrics.AUROC,
        'GS_tempo': TempoAcc,
    }
    metric_config = config.model.peft.metric
    task = config.experiment.task.replace('_feature', '')
    metric = task2metric[task](**(metric_config if metric_config is not None else {}))
    return metric
   


class BeatF1MedianScore(Metric):
    def __init__(self, fps, **kwargs):
        super().__init__(**kwargs)
        self.add_state("f1_scores", default=[], dist_reduce_fx=None)
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
        print(ests)
        for idx, e in enumerate(ests):
            f_score = beat_f1(reference_beats=tgt_second_seq[idx], estimated_beats=e)
            self.f1_scores.append(torch.tensor(f_score))
            

    def compute(self):
        if not self.f1_scores:
            return torch.tensor(0.0)
        return torch.tensor(self.f1_scores).median()
    


import numpy as np
import torch
import torchmetrics
import mir_eval
from mir_eval.beat import validate
from madmom.features.beats import DBNBeatTrackingProcessor


class BCEBeatFMeasure(torchmetrics.Metric):
    def __init__(self, label_freq=75, downbeat=False, metric_type="both"):
        super().__init__()
        self.label_freq = label_freq
        self.downbeat = downbeat
        self.type = metric_type
        self.add_state("f_measure", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("matching", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("estimate", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("reference", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("f_measure_db", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("matching_db", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("estimate_db", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("reference_db", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds and target are assumed to be sequences of beat times
        def get_idx(x):
            indices = np.nonzero(x == 1)  #shape (2251,)
            indices = indices[0].flatten()
            indices = indices.astype(np.float32) / self.label_freq
            return indices
        
        proc = DBNBeatTrackingProcessor(fps=self.label_freq)
        estimated_beats = proc(preds[0].cpu().numpy())
        
        reference_beats = get_idx(target[:,0].cpu().numpy())
        f_measure_threshold=0.07
        mir_eval.beat.validate(reference_beats, estimated_beats)
        # # When estimated beats are empty, no beats are correct; metric is 0
        # if estimated_beats.size == 0 or reference_beats.size == 0:
        #     return 0.
        # Compute the best-case matching between reference and estimated locations
        matching = mir_eval.util.match_events(reference_beats,
                                            estimated_beats,
                                            f_measure_threshold)
        self.matching += len(matching)
        self.estimate += len(estimated_beats)
        self.reference += len(reference_beats)
        
        # proc = DBNDownBeatTrackingProcessor()
        proc = DBNBeatTrackingProcessor(min_bpm=18, max_bpm=72, fps=self.label_freq)
        estimated_beats = proc(preds[1].cpu().numpy())
        reference_beats = get_idx(target[:,1].cpu().numpy())
        mir_eval.beat.validate(reference_beats, estimated_beats)
        matching = mir_eval.util.match_events(reference_beats,
                                            estimated_beats,
                                            f_measure_threshold)
        self.matching_db += len(matching)
        self.estimate_db += len(estimated_beats)
        self.reference_db += len(reference_beats)
        
    def compute(self):
        def calculate(matching, estimate, reference):
            if estimate == 0 or reference == 0:
                return torch.tensor(0.0)
            precision = float(matching)/estimate
            recall = float(matching)/reference
            if precision == 0 and recall == 0:
                f_measure = 0.0
            else:
                f_measure = mir_eval.util.f_measure(precision, recall)
            return torch.tensor(f_measure)
        if self.type == "beat":
            return calculate(self.matching, self.estimate, self.reference)
        elif self.type == "downbeat":
            return calculate(self.matching_db, self.estimate_db, self.reference_db)
        else:
            return (calculate(self.matching, self.estimate, self.reference) + calculate(self.matching_db, self.estimate_db, self.reference_db)) / 2
        

class BeatFMeasure(BCEBeatFMeasure):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds and target are assumed to be sequences of beat times
        def get_idx(x):
            x = x.squeeze(0)
            if self.downbeat:
                indices = torch.nonzero((x == 2), as_tuple=False)
            else:
                indices = torch.nonzero((x == 1) | (x == 2), as_tuple=False)
            indices = torch.flatten(indices)
            indices = indices.float() / self.label_freq
            return indices.cpu().numpy()
        
        batch_size = preds.shape[0]
        for batch in range(batch_size):
            proc = DBNBeatTrackingProcessor(fps=self.label_freq)
            estimated_beats = proc(preds[batch][1].cpu().numpy())
            
            reference_beats = get_idx(target[batch])
            f_measure_threshold=0.07
            mir_eval.beat.validate(reference_beats, estimated_beats)
            matching = mir_eval.util.match_events(reference_beats,
                                                estimated_beats,
                                                f_measure_threshold)
            self.matching += len(matching)
            self.estimate += len(estimated_beats)
            self.reference += len(reference_beats)
        
    def compute(self):
        if self.estimate == 0 or self.reference == 0:
            return torch.tensor(0.0)
        precision = float(self.matching)/self.estimate
        recall = float(self.matching)/self.reference
        if precision == 0 and recall == 0:
            f_measure = 0.0
        else:
            f_measure = mir_eval.util.f_measure(precision, recall)
        
        return torch.tensor(f_measure)

class KeyAccRefined(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("scores", default=[], dist_reduce_fx=None)
        self.classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}
        self.id2class = {v: k for k, v in self.class2id.items()}
        
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        if preds.dim() == 2:
            preds = preds.argmax(dim=-1)
            preds = preds.long()
        correct = preds == labels
        # transfer preds, labels to long
        scores = [
            mir_eval.key.weighted_score(
                self.id2class[ref_key.item()], self.id2class[est_key.item()]
            )
            for ref_key, est_key in zip(labels, preds)
        ]
        self.correct += torch.sum(correct)
        self.total += len(labels)
        self.scores += scores

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0)
        accuracy = self.correct.float() / self.total
        scores = torch.tensor(self.scores).mean()
        return scores

from madmom.evaluation.tempo import tempo_evaluation


class TempoAcc(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("pscore", default=[], dist_reduce_fx=None)
        self.add_state("acc1", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        # preds shape: [bs, 300]
        # labels shape: [bs]

        detections = torch.max(preds,dim=-1).indices
        detections =  np.array([[detection.cpu().numpy()] for detection in detections])
        annotations = np.array([[label.cpu().numpy()] for label in labels])
        for detection, annotation in zip(detections, annotations):
            acc1 = tempo_evaluation(detections=detection, annotations=annotation, tolerance=0.04)[1]
            self.acc1 += int(acc1)
            self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0)
        res = self.acc1 / self.total
        return res
    

if __name__ == '__main__':
    # testing
    preds = torch.zeros((2, 300))
    preds[0, 140] = 0.5
    preds[0, 70] = 0.5
    preds[1, 60] = 0.5
    preds[1, 120] = 0.5
    labels = torch.tensor([140, 60])
    acc = TempoAcc()
    acc.update(preds, labels)
    res = acc.compute()
    print(res)