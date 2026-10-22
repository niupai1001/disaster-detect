"""最小但完整的训练器：前向 → 损失 → 反向 → 保存 → 重载 → 推理。

刻意不引入 Lightning：第一版只需要一条可复现、可调试的训练链路。
骨干冻结部分用 ``no_grad``，避免 BatchNorm 统计漂移。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from ..data.pipeline import ModelBatch
from .losses import CombinedLoss
from .metrics import confusion, summarize


@dataclass
class StepLog:
    step: int
    epoch: int
    loss: dict
    latency_s: float
    lr: float


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    history: list[StepLog] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model,
        loss: CombinedLoss,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str | torch.device = "cpu",
        out_dir: str | Path = "outputs",
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.loss = loss
        self.device = torch.device(device)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip = grad_clip

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("没有任何可训练参数，检查冻结设置")
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.state = TrainState()

    # ------------------------------------------------------------------
    def _frozen_to_eval(self) -> None:
        """把冻结骨干固定为 eval。

        ``model.train()`` 会把所有子模块切到训练态，DropPath/随机深度、Dropout
        以及 BatchNorm 统计都会随之改变；冻结骨干若在 train 态运行，同一张图
        每次前向都会得到不同特征，训练目标变成"追逐噪声"。
        """
        for name in ("terramind", "anysat"):
            m = getattr(self.model, name, None)
            if m is not None:
                m.eval()
        text = getattr(self.model, "semantic", None)
        if text is not None and hasattr(text, "text"):
            text.text.eval()

    def train_step(self, batch: ModelBatch) -> dict:
        self.model.train()
        self._frozen_to_eval()
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        out = self.model(
            batch.x12,
            batch.x10,
            batch.s2_dates,
            region_mask=batch.mask.unsqueeze(1).float(),
            valid=batch.valid,
        )
        dt = time.perf_counter() - t0

        target = batch.mask.float().unsqueeze(1)
        valid = batch.valid.float().unsqueeze(1)

        losses = self.loss(
            fine_logits=out.fine_logits,
            coarse_logits=out.coarse_logits,
            presence_logits=out.presence_logits,
            target=target,
            valid=valid,
            region_embedding=out.region_embedding,
            class_prototype=self.model.semantic.class_prototype,
            region_label=torch.tensor([1.0, 0.0], device=self.device),  # 前景=1, 背景=0
            region_exists=out.region_exists,
        )
        losses.total.backward()
        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.grad_clip
            )
        self.optimizer.step()

        self.state.global_step += 1
        log = {
            "step": self.state.global_step,
            "epoch": self.state.epoch,
            "loss": losses.as_dict(),
            "latency_s": dt,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        self.state.history.append(StepLog(**log))
        return log

    @torch.no_grad()
    def evaluate(
        self,
        batches: Iterable[ModelBatch],
        threshold: float = 0.5,
        use_presence_gate: bool = False,
    ) -> dict:
        """评估只统计 valid 像素；无效/未知像素完全剔除（不改写成负样本）。"""
        self.model.eval()
        counters = []
        presence_stats = {"total": 0, "rejected": 0, "rejected_and_empty_gt": 0,
                          "empty_gt": 0, "false_alarm_on_empty": 0}
        for batch in batches:
            batch = batch.to(self.device)
            out = self.model(batch.x12, batch.x10, batch.s2_dates, return_features=False)
            prob_t = torch.sigmoid(out.fine_logits)
            prob_t = prob_t * batch.valid.unsqueeze(1).float()
            prob = prob_t.squeeze(1).cpu().numpy()
            pres = torch.sigmoid(out.presence_logits).squeeze(1).cpu().numpy()
            gt = batch.mask.cpu().numpy()
            v = batch.valid.cpu().numpy()
            for i in range(prob.shape[0]):
                pred = prob[i] > threshold
                # presence 门控只在显式开启时生效（无负样本训练时它是随机分数）
                rejected = bool(use_presence_gate and pres[i] <= threshold)
                if rejected:
                    pred = np.zeros_like(pred, dtype=bool)
                counters.append(confusion(pred, gt[i], v[i]))
                presence_stats["total"] += 1
                empty_gt = int((gt[i][v[i]] > 0).sum()) == 0
                if empty_gt:
                    presence_stats["empty_gt"] += 1
                    if pred[v[i]].sum() > 0:
                        presence_stats["false_alarm_on_empty"] += 1
                if rejected:
                    presence_stats["rejected"] += 1
                    presence_stats["rejected_and_empty_gt"] += int(empty_gt)
        metrics = summarize(counters)
        metrics["presence"] = presence_stats
        metrics["empty_tile_fpr"] = (
            presence_stats["false_alarm_on_empty"] / presence_stats["empty_gt"]
            if presence_stats["empty_gt"] else float("nan")
        )
        return metrics

    # ------------------------------------------------------------------
    def save(
        self,
        path: str | Path,
        extra: dict | None = None,
        exclude_prefixes: tuple[str, ...] = ("terramind.", "anysat.", "semantic.text."),
    ) -> Path:
        """保存 checkpoint。

        默认**不保存冻结骨干**（约 371M 参数 / 1.5GB）：它们在加载时会由官方预训练
        权重重新构建，若一并保存会让 checkpoint 膨胀数倍且容易与官方权重版本混淆。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sd = {
            k: v for k, v in self.model.state_dict().items()
            if not k.startswith(exclude_prefixes)
        }
        payload = {
            "model": sd,
            "excluded_prefixes": list(exclude_prefixes),
            "optimizer": self.optimizer.state_dict(),
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "history": [s.__dict__ for s in self.state.history],
            },
            "extra": extra or {},
        }
        torch.save(payload, path)
        return path

    def load(self, path: str | Path) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(payload["model"], strict=False)
        excluded = tuple(payload.get("excluded_prefixes", ()))
        # 只允许"被排除的冻结骨干"缺失，其余缺失都是真问题
        bad = [k for k in missing if not k.startswith(excluded)] if excluded else list(missing)
        if bad:
            raise RuntimeError(f"加载 checkpoint 后有非骨干参数缺失: {bad[:8]}")
        if unexpected:
            raise RuntimeError(f"checkpoint 含未知参数: {unexpected[:8]}")
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        s = payload.get("state", {})
        self.state.epoch = s.get("epoch", 0)
        self.state.global_step = s.get("global_step", 0)
        return payload.get("extra", {})

    def dump_history(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([s.__dict__ for s in self.state.history], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
