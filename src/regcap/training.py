"""Lightning training, validation, evaluation, and prediction logging."""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import MeanMetric, Accuracy, Precision, Recall, F1Score
from typing import Optional


from .model import VulDetectionModel
from .rdp_schema import RDP_DIM

class VulDetectionSystem(pl.LightningModule):
    """Optimize classification and weak region targets, then log test predictions."""

    def __init__(
            self,
            model_args,
            lr: float = 1e-3,
            weight_decay: float = 0.0,

            attr_dim: int = RDP_DIM,
            disabled_views=(),
            use_rdp: bool = True,
            pooling: str = "weighted",

            lambda_reg: float = 1.0,
            lambda_cls: float = 1.0,
            pos_weight: Optional[float] = None,

            use_dynamic_threshold: bool = True,
            decision_threshold_init: float = 0.5,
            threshold_mode: str = "recall_at_precision",

            threshold_beta: float = 2.0,
            threshold_min_precision: float = 0.60,
            threshold_target_recall: Optional[float] = None,
            threshold_search_steps: int = 201,
            threshold_ema: float = 0.0,


            pred_dump_dir: str = "./pred_dumps",
            pred_tag: Optional[str] = None,
            dump_region_scores: bool = True,
            dump_logits: bool = False,
    ):
        super().__init__()
        if int(attr_dim) != RDP_DIM:
            raise ValueError("REGCAP requires the canonical 31-feature RDP")
        self.save_hyperparameters()

        self.model = VulDetectionModel(
            hidden_dim=model_args.hidden_dim,
            input_dim=model_args.input_dim,
            heads=getattr(model_args, "gat_heads", 4),
            ggnn_steps=getattr(model_args, "ggnn_steps", 6),
            attr_dim=int(attr_dim),
            disabled_views=disabled_views,
            use_rdp=use_rdp,
            pooling=pooling,
        )

        self.lr = lr
        self.weight_decay = weight_decay

        self.criterion_reg = self._masked_regression_loss
        if pos_weight is not None:
            self.criterion_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.criterion_cls = nn.BCEWithLogitsLoss()

        self.train_mse = MeanMetric(); self.train_mae = MeanMetric()
        self.val_mse   = MeanMetric(); self.val_mae   = MeanMetric()
        self.test_mse  = MeanMetric(); self.test_mae  = MeanMetric()

        self.train_cls_metrics = torch.nn.ModuleDict({
            "acc": Accuracy(task="binary"),
            "pre": Precision(task="binary"),
            "rec": Recall(task="binary"),
            "f1": F1Score(task="binary"),
        })

        self.register_buffer(
            "decision_threshold",
            torch.tensor(float(decision_threshold_init), dtype=torch.float32)
        )
        self._val_cls_probs = []
        self._val_cls_targets = []
        self._test_cls_probs = []
        self._test_cls_targets = []



        self._checked_attr_dim = False

        self._test_records = []

    @staticmethod
    def _binary_metrics_from_threshold(
            probs: torch.Tensor,
            target: torch.Tensor,
            threshold: float,
            beta: float = 2.0,
    ):
        probs = probs.float().reshape(-1)
        target = target.int().reshape(-1)
        pred = (probs >= threshold).int()

        tp = ((pred == 1) & (target == 1)).sum().float()
        tn = ((pred == 0) & (target == 0)).sum().float()
        fp = ((pred == 1) & (target == 0)).sum().float()
        fn = ((pred == 0) & (target == 1)).sum().float()

        precision = tp / (tp + fp).clamp(min=1.0)
        recall = tp / (tp + fn).clamp(min=1.0)
        acc = (tp + tn) / (tp + tn + fp + fn).clamp(min=1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)

        beta2 = beta * beta
        fbeta = (1.0 + beta2) * precision * recall / (beta2 * precision + recall).clamp(min=1e-12)

        return {
            "acc": acc,
            "pre": precision,
            "rec": recall,
            "f1": f1,
            "fbeta": fbeta,
        }

    @staticmethod
    def _search_best_threshold(
            probs: torch.Tensor,
            target: torch.Tensor,
            mode: str = "fbeta",
            beta: float = 2.0,
            min_precision: float = 0.60,
            target_recall: Optional[float] = None,
            search_steps: int = 201,
    ):
        probs = probs.float().reshape(-1)
        target = target.int().reshape(-1).bool()

        thresholds = torch.linspace(0.01, 0.99, steps=search_steps, device=probs.device)

        pred = probs.unsqueeze(0) >= thresholds.unsqueeze(1)
        y = target.unsqueeze(0)

        tp = (pred & y).sum(dim=1).float()
        tn = ((~pred) & (~y)).sum(dim=1).float()
        fp = (pred & (~y)).sum(dim=1).float()
        fn = ((~pred) & y).sum(dim=1).float()

        precision = tp / (tp + fp).clamp(min=1.0)
        recall = tp / (tp + fn).clamp(min=1.0)
        acc = (tp + tn) / (tp + tn + fp + fn).clamp(min=1.0)
        f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)

        beta2 = beta * beta
        fbeta = (1.0 + beta2) * precision * recall / (beta2 * precision + recall).clamp(min=1e-12)

        if mode == "recall_at_precision":
            valid = precision >= min_precision
            if valid.any():
                score = recall.masked_fill(~valid, -1.0)
            else:

                score = fbeta - (min_precision - precision).clamp(min=0.0) * 0.25

        elif mode == "precision_at_recall" and target_recall is not None:
            valid = recall >= target_recall
            if valid.any():
                score = precision.masked_fill(~valid, -1.0)
            else:
                score = fbeta - (target_recall - recall).clamp(min=0.0) * 0.25

        elif mode == "f1":
            score = f1

        else:
            score = fbeta

        best_idx = int(torch.argmax(score).item())
        best_thr = float(thresholds[best_idx].item())

        best_metrics = {
            "acc": float(acc[best_idx].item()),
            "pre": float(precision[best_idx].item()),
            "rec": float(recall[best_idx].item()),
            "f1": float(f1[best_idx].item()),
            "fbeta": float(fbeta[best_idx].item()),
        }
        return best_thr, best_metrics

    @staticmethod
    def _masked_regression_loss(pred: torch.Tensor,
                                target: torch.Tensor,
                                mask: torch.Tensor):

        w = mask.float()
        denom = w.sum().clamp(min=1.0)
        mse = ((pred - target) ** 2) * w
        mae = (pred - target).abs() * w
        mse_mean = mse.sum() / denom
        mae_mean = mae.sum() / denom
        return mse_mean, mae_mean

    def _step_impl(self, batch, stage: str):

        if (not self._checked_attr_dim) and hasattr(batch, "region_attr"):
            F_seen = int(batch.region_attr.size(-1)) if batch.region_attr.ndim == 3 else 0
            F_cfg = int(self.hparams.attr_dim)
            if F_seen != F_cfg:
                raise RuntimeError(
                    f"region_attr has {F_seen} columns; the model expects {F_cfg}."
                )
            self._checked_attr_dim = True

        out = self.model(batch)

        logits_r = out["region_logits"]
        mask_r   = out["region_mask"].bool()
        target_r = batch.region_score.float().clamp(0.0, 1.0)
        pred_r   = torch.sigmoid(logits_r)

        cls_logits = out["cls_logits"]
        y          = batch.y.float()

        mse_mean, mae_mean = self.criterion_reg(pred_r, target_r, mask_r)
        cls_bce = self.criterion_cls(cls_logits, y)

        loss = self.hparams.lambda_reg * mse_mean + self.hparams.lambda_cls * cls_bce

        probs = torch.sigmoid(cls_logits).detach()
        y_int = y.int()

        if stage == "train":

            self.train_mse.update(mse_mean.detach())
            self.train_mae.update(mae_mean.detach())

            for m in self.train_cls_metrics.values():
                m.update(probs, y_int)

            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=y.size(0))
            self.log("train_cls_bce", cls_bce.detach(), prog_bar=False, on_step=True, on_epoch=False, batch_size=y.size(0))
            self.log("train_region_mse_step", mse_mean.detach(), prog_bar=False, on_step=True, on_epoch=False, batch_size=y.size(0))

        elif stage == "val":

            self.val_mse.update(mse_mean.detach())
            self.val_mae.update(mae_mean.detach())

            self._val_cls_probs.append(probs.detach().cpu())
            self._val_cls_targets.append(y_int.detach().cpu())


            self.log("val_cls_bce_step", cls_bce.detach(), prog_bar=False, on_step=True, on_epoch=False, batch_size=y.size(0))
            self.log("val_region_mse_step", mse_mean.detach(), prog_bar=False, on_step=True, on_epoch=False, batch_size=y.size(0))

        else:

            self.test_mse.update(mse_mean.detach()); self.test_mae.update(mae_mean.detach())

            self._test_cls_probs.append(probs.detach().cpu())
            self._test_cls_targets.append(y_int.detach().cpu())

            thr = float(self.decision_threshold.item())
            preds = (probs >= thr).to(torch.int).cpu().tolist()

            file_names = getattr(batch, "file_name", None)
            if file_names is None:

                file_names = [f"sample_{i}" for i in range(len(y))]
            B = len(file_names)
            y_cpu = y_int.cpu().tolist()
            prob_cpu = probs.cpu().tolist()
            logit_cpu = cls_logits.detach().cpu().tolist() if self.hparams.dump_logits else [None] * B

            if self.hparams.dump_region_scores:
                pr = pred_r.detach().cpu()
                mk = mask_r.detach().cpu()
                region_scores = [pr[i][mk[i]].tolist() for i in range(B)]
            else:
                region_scores = [None] * B

            for i in range(B):
                rec = {
                    "file": file_names[i],
                    "y_true": int(y_cpu[i]),
                    "prob": float(prob_cpu[i]),
                    "pred": int(preds[i]),
                    "threshold": float(self.decision_threshold.item()),
                }
                if self.hparams.dump_logits:
                    rec["logit"] = float(logit_cpu[i])
                if self.hparams.dump_region_scores:
                    rec["region_scores"] = region_scores[i]
                self._test_records.append(rec)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step_impl(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step_impl(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step_impl(batch, "test")

    def on_train_epoch_end(self):

        self.log("train_region_mse", self.train_mse.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("train_region_mae", self.train_mae.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.train_mse.reset(); self.train_mae.reset()

        vals = {f"train_{k}": m.compute() for k, m in self.train_cls_metrics.items()}
        for k, v in vals.items():
            self.log(k, v, prog_bar=True, on_epoch=True, sync_dist=True)
        for m in self.train_cls_metrics.values():
            m.reset()

    def on_validation_epoch_end(self):

        self.log("val_region_mse", self.val_mse.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_region_mae", self.val_mae.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.val_mse.reset(); self.val_mae.reset()

        if len(self._val_cls_probs) > 0:
            val_probs = torch.cat(self._val_cls_probs, dim=0)
            val_targets = torch.cat(self._val_cls_targets, dim=0)
            self._val_cls_probs.clear()
            self._val_cls_targets.clear()

            if self.hparams.use_dynamic_threshold:
                raw_thr, _ = self._search_best_threshold(
                    probs=val_probs,
                    target=val_targets,
                    mode=self.hparams.threshold_mode,
                    beta=float(self.hparams.threshold_beta),
                    min_precision=float(self.hparams.threshold_min_precision),
                    target_recall=self.hparams.threshold_target_recall,
                    search_steps=int(self.hparams.threshold_search_steps),
                )

                ema = float(self.hparams.threshold_ema)
                old_thr = float(self.decision_threshold.item())
                new_thr = ema * old_thr + (1.0 - ema) * raw_thr
                new_thr = float(max(0.01, min(0.99, new_thr)))
                self.decision_threshold.fill_(new_thr)

            val_cls = self._binary_metrics_from_threshold(
                probs=val_probs,
                target=val_targets,
                threshold=float(self.decision_threshold.item()),
                beta=float(self.hparams.threshold_beta),
            )

            self.log("val_acc", val_cls["acc"], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_pre", val_cls["pre"], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_rec", val_cls["rec"], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_f1", val_cls["f1"], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_fbeta", val_cls["fbeta"], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("val_threshold", self.decision_threshold.detach(),
                     prog_bar=True, on_epoch=True, sync_dist=True)


    def on_test_epoch_end(self):

        self.log("test_region_mse", self.test_mse.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_region_mae", self.test_mae.compute(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.test_mse.reset(); self.test_mae.reset()

        if len(self._test_cls_probs) > 0:
            test_probs = torch.cat(self._test_cls_probs, dim=0)
            test_targets = torch.cat(self._test_cls_targets, dim=0)
            self._test_cls_probs.clear()
            self._test_cls_targets.clear()

            test_cls = self._binary_metrics_from_threshold(
                probs=test_probs,
                target=test_targets,
                threshold=float(self.decision_threshold.item()),
                beta=float(self.hparams.threshold_beta),
            )

            self.log("test_acc", test_cls["acc"], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("test_pre", test_cls["pre"], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("test_rec", test_cls["rec"], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("test_f1", test_cls["f1"], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("test_fbeta", test_cls["fbeta"], prog_bar=True, on_epoch=True, sync_dist=True)
            self.log("test_threshold", self.decision_threshold.detach(),
                     prog_bar=True, on_epoch=True, sync_dist=True)
        else:
            test_cls = {
                "acc": torch.tensor(0.0),
                "pre": torch.tensor(0.0),
                "rec": torch.tensor(0.0),
                "f1": torch.tensor(0.0),
                "fbeta": torch.tensor(0.0),
            }

        import json, csv, os
        tag = self.hparams.pred_tag or "test"
        dump_dir = self.hparams.pred_dump_dir
        os.makedirs(dump_dir, exist_ok=True)

        pred_jsonl = os.path.join(dump_dir, f"pred_{tag}.jsonl")
        with open(pred_jsonl, "w", encoding="utf-8") as f:
            for rec in self._test_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary = {
            "tag": tag,
            "num_samples": len(self._test_records),
            "metrics": {
                "acc": float(test_cls["acc"].item()),
                "pre": float(test_cls["pre"].item()),
                "rec": float(test_cls["rec"].item()),
                "f1": float(test_cls["f1"].item()),
                "fbeta": float(test_cls["fbeta"].item()),
                "threshold": float(self.decision_threshold),
                "region_mse": float(
                    self.trainer.callback_metrics.get("test_region_mse", torch.tensor(0.)).item()) if hasattr(self,
                                                                                                              "trainer") else None,
                "region_mae": float(
                    self.trainer.callback_metrics.get("test_region_mae", torch.tensor(0.)).item()) if hasattr(self,
                                                                                                              "trainer") else None,
            }
        }
        with open(os.path.join(dump_dir, f"summary_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        pred_csv = os.path.join(dump_dir, f"pred_{tag}.csv")
        with open(pred_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "y_true", "prob", "pred"])
            for r in self._test_records:
                w.writerow([r["file"], r["y_true"], r["prob"], r["pred"]])

        self._test_records.clear()

        print(f"[TEST DUMP] jsonl={pred_jsonl}")
        print(f"[TEST DUMP] csv  ={pred_csv}")
        print(f"[TEST DUMP] summary={os.path.join(dump_dir, f'summary_{tag}.json')}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
