from typing import Literal, Optional, Self

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.notebook import trange  # type: ignore

from .attention_mapping import AttentionMapping, T


class ConvAttentionMapping(AttentionMapping):
    model: Optional[nn.Module] = None
    num_draft_layers: Optional[int] = None
    num_draft_heads: Optional[int] = None
    num_full_layers: Optional[int] = None
    num_full_heads: Optional[int] = None

    def __init__(
        self,
        path: Optional[str] = None,
        num_hidden_layers: int = 4,
        num_hidden_channels: int = 512,
        kernel_size: int = 9,
        dilation: int = 1,
        standardize_inputs: bool = False,
        normalize_inputs: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_channels = num_hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.standardize_inputs = standardize_inputs
        self.normalize_inputs = normalize_inputs
        self.dtype = dtype
        self.device = device
        if path is not None:
            parameters = torch.load(path, weights_only=False, map_location=device)
            self.model = parameters["model"].to(device, dtype)
            self.num_draft_layers = parameters["num_draft_layers"]
            self.num_draft_heads = parameters["num_draft_heads"]
            self.num_full_layers = parameters["num_full_layers"]
            self.num_full_heads = parameters["num_full_heads"]
            if "standardize_inputs" in parameters:
                self.standardize_inputs = parameters["standardize_inputs"]
            else:
                self.standardize_inputs = False

            if "normalize_inputs" in parameters:
                self.normalize_inputs = parameters["normalize_inputs"]
            else:
                self.normalize_inputs = True

    def init_model(self) -> nn.Module:
        assert (
            self.num_draft_layers is not None
            and self.num_draft_heads is not None
            and self.num_full_layers is not None
            and self.num_full_heads is not None
        )

        num_draft_channels = self.num_draft_layers * self.num_draft_heads
        num_full_channels = self.num_full_layers * self.num_full_heads

        layers = [
            nn.Conv1d(
                in_channels=num_draft_channels,
                out_channels=self.num_hidden_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=1,
                padding="same",
                padding_mode="replicate",
                device=self.device,
                dtype=self.dtype,
            ),
            nn.GELU(),
        ]

        for _ in range(self.num_hidden_layers - 1):
            layers.append(
                nn.Conv1d(
                    in_channels=self.num_hidden_channels,
                    out_channels=self.num_hidden_channels,
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    stride=1,
                    padding="same",
                    padding_mode="replicate",
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            layers.append(nn.GELU())

        layers.append(
            nn.Conv1d(
                in_channels=self.num_hidden_channels,
                out_channels=num_full_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                stride=1,
                padding="same",
                padding_mode="replicate",
                device=self.device,
                dtype=self.dtype,
            )
        )
        return nn.Sequential(*layers)

    def loss_fn(
        self,
        output: Tensor,
        target: Tensor,
        objective: Literal["mse", "kl_div"],
    ) -> Tensor:
        if objective == "mse":
            return F.mse_loss(output, target)
        elif objective == "kl_div":
            if self.normalize_inputs:
                target = (target / target.sum(dim=-1)[..., None]).log()
            else:
                target = target.log_softmax(dim=-1)

            output = output.log_softmax(dim=-1)
            return F.kl_div(output, target, reduction="batchmean", log_target=True)
        else:
            raise NotImplementedError

    def fit(
        self,
        draft_reduced_attentions: list[Tensor],
        full_reduced_attentions: list[Tensor],
        test_draft_reduced_attentions: Optional[list[Tensor]] = None,
        test_full_reduced_attentions: Optional[list[Tensor]] = None,
        objective: Literal["mse", "kl_div"] = "kl_div",
        num_iters: int = 256,
        lr: float = 1e-4,
        lr_decay: float = 0.1,
        weight_decay: float = 0.01,
        resume: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> Self:
        self.num_draft_layers = draft_reduced_attentions[0].shape[0]
        self.num_draft_heads = draft_reduced_attentions[0].shape[2]
        self.num_full_layers = full_reduced_attentions[0].shape[0]
        self.num_full_heads = full_reduced_attentions[0].shape[2]

        if resume and checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path,
                weights_only=False,
                map_location=self.device,
            )
            self.model = checkpoint["model"]
            assert self.model is not None
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                num_iters,
                lr * lr_decay,
            )
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])

            for p in self.model.parameters():
                p.requires_grad = True
        else:
            self.model = self.init_model()
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                num_iters,
                lr * lr_decay,
            )

        if torch.cuda.is_available():
            self.model.compile()

        min_test_loss = float("inf")
        progress_bar = trange(num_iters, desc="[]")
        for _ in progress_bar:
            train_losses = []
            permutation: list[int] = np.random.permutation(
                len(draft_reduced_attentions)
            ).tolist()
            for i in permutation:
                x = draft_reduced_attentions[i].to(self.device, self.dtype)
                y = full_reduced_attentions[i].to(self.device, self.dtype)
                loss = self.loss_fn(self(x), y, objective)
                if torch.isfinite(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

            scheduler.step()
            train_loss = sum(train_losses) / len(train_losses)

            if (
                test_draft_reduced_attentions is not None
                and test_full_reduced_attentions is not None
            ):
                test_losses = []
                for i in range(len(test_draft_reduced_attentions)):
                    x = test_draft_reduced_attentions[i].to(self.device, self.dtype)
                    y = test_full_reduced_attentions[i].to(self.device, self.dtype)
                    with torch.no_grad():
                        loss = self.loss_fn(self(x), y, objective)

                    if torch.isfinite(loss):
                        test_losses.append(loss.item())

                test_loss = sum(test_losses) / len(test_losses)
                if test_loss < min_test_loss and checkpoint_path is not None:
                    torch.save(
                        {
                            "model": self.model,
                            "num_draft_layers": self.num_draft_layers,
                            "num_draft_heads": self.num_draft_heads,
                            "num_full_layers": self.num_full_layers,
                            "num_full_heads": self.num_full_heads,
                            "standardize_inputs": self.standardize_inputs,
                            "normalize_inputs": self.normalize_inputs,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        },
                        checkpoint_path,
                    )

                min_test_loss = min(min_test_loss, test_loss)
                progress_bar.set_description(
                    f"[train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, min test loss: {min_test_loss:.4f}]"
                )
            else:
                progress_bar.set_description(f"[train loss: {train_loss:.4f}]")

        for p in self.model.parameters():
            p.requires_grad = False

        return self

    def stack(self, a: Tensor) -> Tensor:
        num_layers, batch_size, num_heads, seq_len = a.shape
        return a.transpose(0, 1).reshape(batch_size, num_layers * num_heads, seq_len)

    def unstack(self, a: Tensor) -> Tensor:
        assert self.num_full_layers is not None and self.num_full_heads is not None
        batch_size, _, seq_len = a.shape
        return a.view(
            batch_size,
            self.num_full_layers,
            self.num_full_heads,
            seq_len,
        ).transpose(0, 1)

    def map_single(self, a: Tensor) -> Tensor:
        assert self.model is not None

        if a.dtype != self.dtype:
            a = a.to(dtype=self.dtype)

        if a.device != self.device:
            a = a.to(device=self.device)

        if self.standardize_inputs:
            a -= a.mean(dim=-1)[..., None]
            a /= a.std(dim=-1)[..., None]
        elif self.normalize_inputs:
            a /= a.sum(dim=-1)[..., None]

        return self.unstack(self.model(self.stack(a)))

    def __call__(self, draft_reduced_attentions: T) -> T:
        if isinstance(draft_reduced_attentions, list):
            return [self.map_single(a) for a in draft_reduced_attentions]
        else:
            return self.map_single(draft_reduced_attentions)

    def save(self, path: str) -> None:
        assert (
            self.model is not None
            and self.num_draft_layers is not None
            and self.num_draft_heads is not None
            and self.num_full_layers is not None
            and self.num_full_heads is not None
        )
        torch.save(
            {
                "model": self.model,
                "num_draft_layers": self.num_draft_layers,
                "num_draft_heads": self.num_draft_heads,
                "num_full_layers": self.num_full_layers,
                "num_full_heads": self.num_full_heads,
                "standardize_inputs": self.standardize_inputs,
                "normalize_inputs": self.normalize_inputs,
            },
            path,
        )
