import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.use_focal_loss = use_focal_loss

    def __call__(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        
        predicts = predicts.permute(1, 0, 2)  # [T, B, C]
        B, T, C = predicts.shape[1], predicts.shape[0], predicts.shape[2]
        
        # Get labels and lengths
        labels = batch[1].astype("int32")
        label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
        
        # Convert labels to tensor
        labels_concat = []
        for label in labels:
            labels_concat.extend(label)
        labels_concat = torch.tensor(labels_concat, dtype=torch.long)
        
        # Prediction lengths (all same for batch)
        predict_lengths = torch.full((B,), T, dtype=torch.long)
        
        # Compute CTC loss
        loss = self.loss_func(predicts, labels_concat, predict_lengths, label_lengths)
        
        return {'loss': loss}


class SARLoss(nn.Module):
    def __init__(self, **kwargs):
        super(SARLoss, self).__init__()
        ignore_index = kwargs.get('ignore_index', 92)
        self.loss_func = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

    def __call__(self, predicts, batch):
        predict = predicts[:, :-1, :]  # ignore last index of outputs to be in same seq_len with targets
        label = batch[1].astype("int64")[:, 1:]  # ignore first index of target in loss calculation
        B, L = label.shape
        predict = torch.reshape(predict, [-1, predict.shape[-1]])
        label = torch.reshape(label, [-1])
        loss = self.loss_func(predict, label)
        return {'loss': loss}


class DistanceLoss(nn.Module):
    """
    Distillation loss
    """
    def __init__(self, mode="l2", **kwargs):
        super().__init__()
        assert mode in ["l1", "l2", "smooth_l1"]
        if mode == "l1":
            self.loss_func = nn.L1Loss(reduction='mean')
        elif mode == "l2":
            self.loss_func = nn.MSELoss(reduction='mean')
        elif mode == "smooth_l1":
            self.loss_func = nn.SmoothL1Loss(reduction='mean')

    def __call__(self, predicts, batch):
        fea_s = predicts["student"]
        fea_t = predicts["teacher"]
        fea_t.detach()
        loss = self.loss_func(fea_s, fea_t)
        return {"distill_loss": loss}


class LossFromOutput(nn.Module):
    def __init__(self, reduction='none', **kwargs):
        super().__init__()
        self.reduction = reduction

    def __call__(self, predicts, batch):
        loss = predicts
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return {'loss': loss}


class KLJSLoss(object):
    def __init__(self, mode='kl'):
        assert mode in ['kl', 'js', 'KL', 'JS']
        self.mode = mode.lower()

    def __call__(self, p1, p2, reduction="mean", eps=1e-5):
        if self.mode == 'kl':
            loss = self.kl_loss(p1, p2, reduction, eps)
        elif self.mode == 'js':
            loss = self.js_loss(p1, p2, reduction, eps)
        return loss

    def kl_loss(self, p1, p2, reduction="mean", eps=1e-5):
        p1 = F.softmax(p1, dim=-1)
        p2 = F.softmax(p2, dim=-1)
        loss = p2 * (torch.log(p2 + eps) - torch.log(p1 + eps))
        
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            pass
        return loss

    def js_loss(self, p1, p2, reduction="mean", eps=1e-5):
        p1 = F.softmax(p1, dim=-1)
        p2 = F.softmax(p2, dim=-1)
        m = (p1 + p2) / 2
        js = 0.5 * self.kl_loss(p1, m, reduction="none", eps=eps) + \
             0.5 * self.kl_loss(p2, m, reduction="none", eps=eps)
        
        if reduction == "mean":
            js = torch.mean(js)
        elif reduction == "sum":
            js = torch.sum(js)
        elif reduction == "none":
            pass
        return js


class DMLLoss(nn.Module):
    """
    Mutual learning loss
    """
    def __init__(self, act=None, use_log=False, **kwargs):
        super().__init__()
        if act is not None:
            self.act = getattr(torch, act)
        else:
            self.act = None
        self.use_log = use_log
        self.jskl_loss = KLJSLoss(mode="js")

    def _act_log(self, x):
        if self.act is not None:
            x = self.act(x)
        if self.use_log is True:
            x = torch.log_softmax(x, dim=-1)
        return x

    def __call__(self, predicts, batch):
        if not isinstance(predicts, dict):
            predicts = {"pred1": predicts[0], "pred2": predicts[1]}
        else:
            assert "pred1" in predicts and "pred2" in predicts
        pred1 = predicts['pred1']
        pred2 = predicts['pred2']
        if isinstance(pred1, list):
            pred1 = pred1[-1]
        if isinstance(pred2, list):
            pred2 = pred2[-1]
        
        pred1 = self._act_log(pred1)
        pred2 = self._act_log(pred2)
        
        loss = self.jskl_loss(pred1, pred2)
        return {"loss": loss}


class DistillationCTCLoss(nn.Module):
    def __init__(self, 
                 teacher_loss_weight=1.0,
                 student_loss_weight=1.0, 
                 distill_loss_weight=2.5,
                 **kwargs):
        super().__init__()
        self.teacher_loss_weight = teacher_loss_weight
        self.student_loss_weight = student_loss_weight 
        self.distill_loss_weight = distill_loss_weight
        
        self.ctc_loss = CTCLoss(**kwargs)
        self.distill_loss = KLJSLoss(mode='js')

    def __call__(self, predicts, batch):
        teacher_pred = predicts["teacher"]
        student_pred = predicts["student"]
        
        # Teacher loss
        teacher_loss = self.ctc_loss(teacher_pred, batch)["loss"]
        
        # Student loss 
        student_loss = self.ctc_loss(student_pred, batch)["loss"]
        
        # Distillation loss
        if isinstance(teacher_pred, (list, tuple)):
            teacher_pred = teacher_pred[-1]
        if isinstance(student_pred, (list, tuple)):
            student_pred = student_pred[-1]
            
        distill_loss = self.distill_loss(student_pred, teacher_pred)
        
        # Total loss
        total_loss = (self.teacher_loss_weight * teacher_loss + 
                     self.student_loss_weight * student_loss +
                     self.distill_loss_weight * distill_loss)
        
        return {
            "loss": total_loss,
            "teacher_loss": teacher_loss,
            "student_loss": student_loss, 
            "distill_loss": distill_loss
        }