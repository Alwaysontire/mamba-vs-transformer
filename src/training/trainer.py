"""HF Trainer"""

from __future__ import annotations

import os

import torch
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils import WEIGHTS_NAME

from src.config.schemas import LossConfig
from src.training.losses import compute_loss


class LMTrainer(Trainer):
    """``Trainer`` для моделей ``forward(input_ids) -> logits``: loss считается снаружи по ``loss_cfg``"""

    def __init__(self, *args, loss_cfg: LossConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_cfg = loss_cfg

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits = model(inputs["input_ids"])
        loss = compute_loss(logits, inputs["labels"], self.loss_cfg)
        return (loss, logits) if return_outputs else loss

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        """Не-``PreTrainedModel`` HF пишет через ``safetensors.save_file``, а он падает на tied weights"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
