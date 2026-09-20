"""
This is a script for supervised image classification using the BigEarthNet v2.0 dataset.
Reference: Clasen et al., "reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis", 2024.
Repository: https://git.tu-berlin.de/rsim/reben-training-scripts
"""

from typing import List, Optional, Any, Sequence, Dict, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from configilm import ConfigILM
from configilm.ConfigILM import ILMConfiguration, ILMType
from configilm.extra.CustomTorchClasses import LinearWarmupCosineAnnealingLR
from huggingface_hub import PyTorchModelHubMixin

# Fallback for BENv2_utils.NEW_LABELS (absent from public configilm 0.4.10)
try:
    from configilm.extra.BENv2_utils import NEW_LABELS
except ImportError:
    NEW_LABELS = [
        "Continuous urban fabric",
        "Discontinuous urban fabric",
        "Industrial or commercial units",
        "Arable land",
        "Permanent crops",
        "Pastures",
        "Complex cultivation patterns",
        "Land principally occupied by agriculture",
        "Broad-leaved forest",
        "Coniferous forest",
        "Mixed forest",
        "Natural grasslands and sclerophyllous vegetation",
        "Transitional woodland-shrub",
        "Beaches, dunes, sands",
        "Inland wetlands",
        "Coastal wetlands",
        "Inland waters",
        "Marine waters",
        "Bare rock and sparsely vegetated areas"
    ]

# Lightweight inference-safe fallback for get_classification_metric_collection (absent from public configilm 0.4.10)
try:
    from configilm.metrics import get_classification_metric_collection
except ImportError:
    def get_classification_metric_collection(task="multilabel", average="macro", num_labels=19, prefix=""):
        return None

__author__ = "Leonard Hackel - BIFOLD/RSiM TU Berlin"


class BigEarthNetv2_0_ImageClassifier(pl.LightningModule, PyTorchModelHubMixin):
    """
    Wrapper around a pytorch module, allowing this module to be used in automatic
    training with pytorch lightning.
    Among other things, the wrapper allows us to do automatic training and removes the
    need to manage data on different devices (e.g. GPU and CPU).
    Also uses the PyTorchModelHubMixin to allow for easy saving and loading of the model to the Huggingface Hub.
    """

    def __init__(
            self,
            config: Optional[Union[ILMConfiguration, Dict[str, Any]]] = None,
            lr: float = 1e-3,
            warmup: Optional[int] = None,
            **kwargs: Any,
    ):
        super().__init__()
        self.lr = lr
        self.warmup = None if warmup is None or warmup < 0 else warmup

        # Construct ILMConfiguration from kwargs if config is not supplied or supplied as a dict
        if config is None or isinstance(config, dict):
            cfg_dict = dict(config) if isinstance(config, dict) else dict(kwargs)
            if "network_type" in cfg_dict and isinstance(cfg_dict["network_type"], int):
                cfg_dict["network_type"] = ILMType(cfg_dict["network_type"])
            valid_keys = {
                "timm_model_name", "hf_model_name", "image_size", "channels", "classes",
                "class_names", "network_type", "visual_features_out", "fusion_in",
                "fusion_out", "fusion_hidden", "v_dropout_rate", "t_dropout_rate",
                "fusion_dropout_rate", "fusion_method", "fusion_activation", "drop_rate",
                "use_pooler_output", "max_sequence_length", "load_pretrained_timm_if_available",
                "load_pretrained_hf_if_available"
            }
            filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
            config = ILMConfiguration(**filtered_cfg)

        self.config = config
        assert config.network_type == ILMType.IMAGE_CLASSIFICATION
        assert config.classes == 19
        self.model = ConfigILM.ConfigILM(config)
        self.val_output_list: List[dict] = []
        self.test_output_list: List[dict] = []
        self.loss = torch.nn.BCEWithLogitsLoss()

        # Training/validation metrics (safely skipped if metric collection is None)
        self.val_metrics_micro = get_classification_metric_collection(
            "multilabel", "micro", num_labels=config.classes, prefix="val/"
        )
        self.val_metrics_macro = get_classification_metric_collection(
            "multilabel", "macro", num_labels=config.classes, prefix="val/"
        )
        self.val_metrics_samples = get_classification_metric_collection(
            "multilabel", "sample", num_labels=config.classes, prefix="val/"
        )
        self.val_metrics_class = get_classification_metric_collection(
            "multilabel", None, num_labels=config.classes, prefix="val/"
        )
        self.test_metrics_micro = get_classification_metric_collection(
            "multilabel", "micro", num_labels=config.classes, prefix="test/"
        )
        self.test_metrics_macro = get_classification_metric_collection(
            "multilabel", "macro", num_labels=config.classes, prefix="test/"
        )
        self.test_metrics_samples = get_classification_metric_collection(
            "multilabel", "sample", num_labels=config.classes, prefix="test/"
        )
        self.test_metrics_class = get_classification_metric_collection(
            "multilabel", None, num_labels=config.classes, prefix="test/"
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat, y)
        self.log("train/loss", loss)
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            current_gpu_mem_mb = torch.cuda.memory_allocated(current_gpu) / 1024 ** 2
            self.log("train/GPU_memory_MB", current_gpu_mem_mb)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

        max_intervals = int(
            self.trainer.max_epochs * len(self.trainer.datamodule.train_ds) / self.trainer.datamodule.batch_size
        )
        if self.warmup is not None:
            print(f"Overwriting warmup with {self.warmup}")
            warmup = self.warmup
        else:
            warmup = 10000 if max_intervals > 10000 else 100 if max_intervals > 100 else 0

        print(f"Optimizing for up to {max_intervals} steps with warmup for {warmup} steps")

        lr_scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=warmup,
                max_epochs=max_intervals,
                warmup_start_lr=self.lr / 10,
                eta_min=self.lr / 10,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat, y)
        self.val_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_output_list = []

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.val_output_list]).mean()
        self.log("val/loss", avg_loss)

        preds = torch.cat([x["outputs"] for x in self.val_output_list])
        labels = torch.cat([x["labels"] for x in self.val_output_list]).long()

        if self.val_metrics_macro is not None:
            metrics_macro = self.val_metrics_macro(preds, labels)
            self.log_dict(metrics_macro)
            self.val_metrics_macro.reset()

        if self.val_metrics_micro is not None:
            metrics_micro = self.val_metrics_micro(preds, labels)
            self.log_dict(metrics_micro)
            self.val_metrics_micro.reset()

        if self.val_metrics_samples is not None:
            metrics_samples = self.val_metrics_samples(preds.unsqueeze(-1), labels.unsqueeze(-1))
            metrics_samples = {k: v.mean() for k, v in metrics_samples.items()}
            self.log_dict(metrics_samples)
            self.val_metrics_samples.reset()

        class_names = NEW_LABELS
        if self.val_metrics_class is not None:
            metrics_class = self.val_metrics_class(preds, labels)
            classwise_acc = {
                f"val/ClasswiseAccuracy/{class_names[i]}": metrics_class["val/MultilabelAccuracy_class"][i]
                for i in range(len(class_names))
            }
            self.log_dict(classwise_acc)
            self.val_metrics_class.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.model(x)
        loss = F.binary_cross_entropy_with_logits(x_hat, y)
        self.test_output_list += [{"loss": loss, "outputs": x_hat, "labels": y}]

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.test_output_list]).mean()
        self.log("test/loss", avg_loss)

        preds = torch.cat([x["outputs"] for x in self.test_output_list])
        labels = torch.cat([x["labels"] for x in self.test_output_list]).long()

        if self.test_metrics_macro is not None:
            metrics_macro = self.test_metrics_macro(preds, labels)
            self.log_dict(metrics_macro)
            self.test_metrics_macro.reset()

        if self.test_metrics_micro is not None:
            metrics_micro = self.test_metrics_micro(preds, labels)
            self.log_dict(metrics_micro)
            self.test_metrics_micro.reset()

        if self.test_metrics_samples is not None:
            metrics_samples = self.test_metrics_samples(preds.unsqueeze(-1), labels.unsqueeze(-1))
            metrics_samples = {k: v.mean() for k, v in metrics_samples.items()}
            self.log_dict(metrics_samples)
            self.test_metrics_samples.reset()

        class_names = NEW_LABELS
        if self.test_metrics_class is not None:
            metrics_class = self.test_metrics_class(preds, labels)
            classwise_acc = {
                f"test/ClasswiseAccuracy/{class_names[i]}": metrics_class["test/MultilabelAccuracy_class"][i]
                for i in range(len(class_names))
            }
            self.log_dict(classwise_acc)
            self.test_metrics_class.reset()

    def forward(self, batch):
        # because we are a wrapper, we call the inner function manually
        return self.model(batch)
