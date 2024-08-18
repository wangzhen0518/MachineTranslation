from dataclasses import dataclass
from typing import List, Optional, Union

import datasets
import torch
from datasets import Dataset
from torch import Tensor, nn, optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from model import get_attn_subsequent_mask


def clean_data(s: str) -> str:
    return s.replace(" ", "").strip()


def transfer_mask(mask: Tensor) -> Tensor:
    """将 bool mask 转换为 float mask"""
    match mask.dtype:
        case torch.bool:
            mask = torch.zeros_like(mask, dtype=torch.float, device=mask.device).masked_fill_(mask, -1e8)
        case torch.float:
            pass
        case _:
            raise TypeError(f"mask.dtype must be float or bool, but now is {mask.dtype}")
    return mask


@dataclass
class Batch:
    src: Tensor
    tgt: Tensor
    tgt_y: Tensor
    src_key_padding_mask: Tensor
    tgt_key_padding_mask: Tensor
    tgt_mask: Tensor
    src_text: str
    tgt_text: str
    device: Optional[torch.device]
    ntoken: int

    def __init__(
        self,
        src: Tensor,
        tgt: Tensor,
        src_key_padding_mask: Tensor,
        tgt_key_padding_mask: Tensor,
        src_text: str,
        tgt_text: str,
        device: Optional[torch.device],
    ):
        src, tgt = src.to(device), tgt.to(device)
        self.src = src
        self.tgt = tgt[:, :-1]
        self.tgt_y = tgt[:, 1:]

        src_key_padding_mask, tgt_key_padding_mask = src_key_padding_mask.to(device), tgt_key_padding_mask.to(device)
        self.src_key_padding_mask = transfer_mask((1 - src_key_padding_mask).type(torch.bool))
        self.tgt_key_padding_mask = transfer_mask((1 - tgt_key_padding_mask[:, :-1]).type(torch.bool))
        self.tgt_mask = transfer_mask(get_attn_subsequent_mask(self.tgt.size(1), device=device))

        self.src_text = src_text
        self.tgt_text = tgt_text

        self.device = device
        self.ntoken = tgt_key_padding_mask.sum().item()


class DataCollator:
    def __init__(self, tokenizer: AutoTokenizer, device: Optional[torch.device] = None) -> None:
        self.tokenizer = tokenizer
        self.device = device

    def collate(self, batch):
        src_text, tgt_text = zip(
            *(
                (
                    self.tokenizer.bos_token + clean_data(item["0"]) + self.tokenizer.eos_token,
                    self.tokenizer.bos_token + item["1"] + self.tokenizer.eos_token,
                )
                for item in batch
            )
        )
        src = self.tokenizer(src_text, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt")
        tgt = self.tokenizer(tgt_text, padding=True, truncation=True, add_special_tokens=False, return_tensors="pt")

        src, src_key_padding_mask = src["input_ids"], src["attention_mask"]
        tgt, tgt_key_padding_mask = tgt["input_ids"], tgt["attention_mask"]

        return Batch(src, tgt, src_key_padding_mask, tgt_key_padding_mask, src_text, tgt_text, self.device)


def save_state(epoch: int, model: nn.Module, optimizer: optim.Optimizer, loss: float, ckpt_pth: str, writer: SummaryWriter):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        ckpt_pth,
    )
    writer.add_text("checkpoint", f"Saved checkpoint at {ckpt_pth}", epoch)


def load_state(ckpt_pth: str, model: nn.Module, optimizer: optim.Optimizer, writer: SummaryWriter):
    ckpt = torch.load(ckpt_pth, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt["epoch"]
    loss = ckpt["loss"]
    writer.add_text("checkpoint", f"Load checkpoint at {ckpt_pth} | Epoch {epoch} | Loss {loss:.6f}", epoch)
    return epoch, loss


def load_model(ckpt_pth: str, model: nn.Module):
    ckpt = torch.load(ckpt_pth, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt["epoch"]
    loss = ckpt["loss"]
    return epoch, loss


def split_dataset(dataset: Dataset, train_radio: float, test_radio: float):
    train_test_split = dataset.train_test_split(test_size=test_radio, shuffle=True)
    train_radio = train_radio / (1 - test_radio)
    train_valid_split = train_test_split["train"].train_test_split(train_size=train_radio, shuffle=True)
    train_dataset = train_valid_split["train"]
    valid_dataset = train_valid_split["test"]
    test_dataset = train_test_split["test"]
    return train_dataset, valid_dataset, test_dataset


def load_dataset(dataset_file: Union[str, List[str]], train_radio: float, test_radio: float):
    dataset = datasets.load_dataset("csv", data_files=dataset_file, split="train")
    return split_dataset(dataset, train_radio, test_radio)
