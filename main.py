import argparse
import math
import os
import shutil
import time
from datetime import datetime
from typing import List, Optional, Union

import pandas as pd
import torch
from datasets import load_dataset
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from data import Batch, DataCollator, clean_data, load_model, load_state, save_state
from model import TransformerModel, get_attn_subsequent_mask

DATASET_DICT = {
    "small": [
        "./data/WMT-Chinese-to-English-Machine-Translation-newstest/damo_mt_testsets_zh2en_news_wmt18.csv",
        "./data/WMT-Chinese-to-English-Machine-Translation-newstest/damo_mt_testsets_zh2en_news_wmt19.csv",
        "./data/WMT-Chinese-to-English-Machine-Translation-newstest/damo_mt_testsets_zh2en_news_wmt20.csv",
    ],
    "large": "./data/WMT-Chinese-to-English-Machine-Translation-Training-Corpus/wmt_zh_en_training_corpus.csv",
}


def get_number_length(n: int) -> int:
    length = int(math.log10(n)) + 1
    return length


# def draw_attn(attn: Tensor,nhead:int):
#     attn=attn[-1].squeeze(0)[0]
#     attn=attn.squeeze(0).cpu().numpy()
#     fig=plt.figure(figsize=(nhead,nhead))
#     ax=fig.add_subplot(1,1,1)
#     ax.matshow(attn,cmap='viridis')
#     ax.set_xticklabels([''])


def get_tokenizer(name: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def train_one_epoch(
    model: TransformerModel,
    optimizer: optim.Optimizer,
    dataloader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    writer: SummaryWriter,
    epoch: int = 0,
    num_epochs: int = 0,
    total_ntokens: int = 0,
    device: Optional[torch.device] = None,
):
    epoch_loss, epoch_ntoken = 0.0, 0
    nbatch = len(dataloader)
    start = time.time()
    for i, batch in enumerate(dataloader):
        batch: Batch
        output = model.forward(
            batch.src.to(device),
            batch.tgt.to(device),
            batch.tgt_mask.to(device),
            batch.src_key_padding_mask.to(device),
            batch.tgt_key_padding_mask.to(device),
            batch.src_key_padding_mask.to(device),
        )

        output = output.reshape(-1, output.size(-1))
        tgt_y = batch.tgt_y.reshape(-1).to(device)
        loss: Tensor = loss_fn(output, tgt_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        epoch_loss += loss
        epoch_ntoken += batch.ntoken

        if i % 30 == 0:
            end = time.time()
            print(f"Epoch [{epoch:4d}/{num_epochs-1:4d}] | Batch [{i:4d}/{nbatch-1:4d}] | Loss {loss:.6f} | NToken {epoch_ntoken+total_ntokens} | Time {end-start:.2f}s")
            writer.add_scalar("training loss", loss, epoch_ntoken + total_ntokens)
            start = time.time()

    return epoch_loss / nbatch, epoch_ntoken


def train(
    model: TransformerModel,
    num_epochs: int,
    tokenizer: AutoTokenizer,
    dataset_file: str,
    batch_size: int,
    lr: float,
    log_dir: str,
    final_model_name: str,
    device: Optional[torch.device] = None,
):
    dataset = load_dataset("csv", data_files=dataset_file, split="train")
    data_collator = DataCollator(tokenizer, device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator.collate,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    timestamp = datetime.now()
    writer_dir = os.path.join(log_dir, timestamp.strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(writer_dir)

    ckpt_final = os.path.join(log_dir, final_model_name)
    # if os.path.exists(ckpt_final):
    #     load_state(ckpt_final, model, optimizer, writer)

    total_ntoken = 0
    loss_list: List[float] = []
    ntoken_list: List[int] = []
    for epoch in range(num_epochs):
        start = time.time()
        loss, ntoken = train_one_epoch(
            model,
            optimizer,
            dataloader,
            loss_fn,
            writer,
            epoch,
            num_epochs,
            total_ntoken,
            device,
        )

        total_ntoken += ntoken
        loss_list.append(loss)
        ntoken_list.append(total_ntoken)

        end = time.time()
        print(f"\nEpoch [{epoch:4d}/{num_epochs-1:4d}] | Loss {loss:.6f} | NToken {total_ntoken} | Time {end-start:.2f}s\n\n")

        ckpt_pth = os.path.join(writer_dir, f"model_epoch_{epoch}.pth")
        save_state(epoch, model, optimizer, loss, ckpt_pth, writer)

    shutil.copyfile(ckpt_pth, ckpt_final)
    return loss_list, ntoken_list


@torch.no_grad()
def greedy_decode(model: TransformerModel, tokenizer: AutoTokenizer, src: Tensor, max_len: int = 50, device: Optional[torch.device] = None):
    model.eval()
    memory = model.encode(src, None)

    tgt = torch.tensor([[tokenizer.bos_token_id]], device=device)
    new_token_id = tokenizer.bos_token_id

    cnt = 0
    while new_token_id != tokenizer.eos_token_id and cnt < max_len:
        tgt_mask = get_attn_subsequent_mask(tgt.size(1), device)
        output = model.decode(tgt, memory, tgt_mask, None)
        prob = model.generate(output[:, -1, :])
        new_token_id = torch.argmax(prob, dim=-1, keepdim=True)
        tgt = torch.cat((tgt, new_token_id), dim=1)
        cnt += 1
    return tgt[0].cpu()


def show_translation(src_text: str, tgt_text: str):
    print(f"Original: {src_text}\nTranslation: {tgt_text}\n\n")


def translate(
    model: TransformerModel,
    tokenizer: AutoTokenizer,
    src_text: str,
    max_len: int = 50,
    device: Optional[torch.device] = None,
) -> str:
    """Translate src and return the result"""
    src_text = tokenizer.bos_token + src_text + tokenizer.eos_token
    src = tokenizer(src_text, truncation=True, add_special_tokens=False, return_tensors="pt")
    src = src["input_ids"].to(device)

    result = greedy_decode(model, tokenizer, src, max_len, device)
    result = tokenizer.decode(result, skip_special_tokens=True)
    return result


def translate_one_dataset(
    model: TransformerModel,
    tokenizer: AutoTokenizer,
    src_file: str,
    target_pth: str,
    max_len: int = 50,
    device: Optional[torch.device] = None,
):
    """将所有数据进行翻译，翻译结果写入到新文件中"""
    print("Datatset:", src_file)
    dataset = pd.read_csv(src_file)
    n = len(dataset)

    target_file = src_file.replace(".csv", "_translated.csv")
    target_file = os.path.join(target_pth, target_file)

    index = ["original", "target", "translated"]
    df = pd.DataFrame(columns=index)
    for idx, src_text, tgt_text in dataset.itertuples():
        src_text = clean_data(src_text)
        result = translate(model, tokenizer, src_text, max_len, device)
        new_row = pd.DataFrame([[src_text, tgt_text, result]], columns=index)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(target_file, index=False)
        print(f"Count [{idx:4d}/{n-1:4d}]")
        show_translation(src_text, result)


def translate_dataset(
    model: TransformerModel,
    tokenizer: AutoTokenizer,
    src_file: Union[str, List[str]],
    target_pth: str,
    max_len: int = 50,
    device: Optional[torch.device] = None,
):
    if isinstance(src_file, str):
        src_file = [src_file]

    if not isinstance(src_file, List):
        raise NotImplementedError(f"src_file need to be a list, not {src_file}")

    for file in src_file:
        translate_one_dataset(model, tokenizer, file, target_pth, max_len, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--sentence", type=str)
    parser.add_argument("-d", "--dataset", type=str, choices=["large", "small"], default="small")
    parser.add_argument("--log", type=str, default="logs")
    parser.add_argument("--result", type=str, default="res", help="directory to save dataset translation results")
    parser.add_argument("--final_model", type=str, help="final name of model", default="model_final.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    has_task = args.train | args.translate | args.demo
    if not has_task:
        exit(0)

    device = torch.device(args.device)

    embed_dim = 128
    ff_dim = 512
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.1
    dataset_file = DATASET_DICT[args.dataset]

    tokenizer = get_tokenizer()

    model = TransformerModel(
        len(tokenizer.vocab),
        embed_dim,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        ff_dim,
        dropout,
    ).to(device)

    if args.train:
        lr = 3e-4
        batch_size = 10
        num_epochs = 1000

        loss_list, ntoken_list = train(
            model,
            num_epochs,
            tokenizer,
            dataset_file,
            batch_size,
            lr,
            args.log,
            args.final_model,
            device,
        )

    ckpt_final = os.path.join(args.log, args.final_model)
    if os.path.exists(ckpt_final):
        load_model(ckpt_final, model)
    else:
        raise FileNotFoundError(f"We can not find model in {ckpt_final}")

    if args.translate:
        translate_dataset(model, tokenizer, dataset_file, args.result, 50, device)

    if args.demo:
        sentence = args.sentence
        translated = translate(model, tokenizer, sentence, device=device)
        show_translation(sentence, translated)


if __name__ == "__main__":
    main()
