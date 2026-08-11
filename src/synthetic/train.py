from __future__ import annotations

from tokenisation import CharacterTokeniser, DiscreteTokeniser
from synthetic_datasets import *
from synthetic_models import TinyCausalTransformer

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from pathlib import Path
import itertools

from dataclasses import dataclass, field
from typing import Literal

DatasetKind = Literal["forward", "forward+reverse", "modular", "markov"]

@dataclass(frozen=True)
class TrainingConfig:  
    dataset_kind: DatasetKind
    n: int = 6
    pairs: list[tuple[int,int]] = field(default_factory=lambda: list(itertools.combinations_with_replacement(range(1,10),2)))
    mod: int = 37
    mc_chain_states: list[str] = field(default_factory=lambda: ["a","b","c","d"])
    mc_chain_dataset_length: int = 1000
    mc_chain_length: int = 8
    train_fraction: float = 0.9
    batch_size: int = 32
    epochs: int = 8000
    learning_rate: float = 3e-3
    weight_decay: float = 0.15
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    d_hidden: int = 512
    dropout: float = 0.0
    max_seq_len: int = 128
    checkpoint_name: str = "lit_end_fib_fwd"

def build_tokeniser(cfg: TrainingConfig):
    if cfg.dataset_kind == "forward":
        return CharacterTokeniser.build()
    elif cfg.dataset_kind == "forward+reverse":
        return CharacterTokeniser.build()
    elif cfg.dataset_kind == "modular":
        return DiscreteTokeniser.build_tokeniser_for_modular_sequences(mod = cfg.mod)
    elif cfg.dataset_kind == "markov":
        return DiscreteTokeniser.build_tokeniser_for_markov_chains(chain_states=cfg.mc_chain_states)
    else:
        raise ValueError(f"Unknown dataset kind: {cfg.dataset_kind}")

def build_dataset(cfg: TrainingConfig):
    if cfg.dataset_kind == "forward":
        return LittleEndianFibDataset(pairs=cfg.pairs, n=cfg.n, tokeniser=build_tokeniser(cfg)) #type: ignore
    elif cfg.dataset_kind == "forward+reverse":
        return LittleEndianFibDatasetForwardReverse(pairs=cfg.pairs, n=cfg.n, tokeniser=build_tokeniser(cfg)) #type: ignore
    elif cfg.dataset_kind == "modular":
        return ModularFibDataset(pairs=cfg.pairs, n=cfg.n, mod=cfg.mod)
    elif cfg.dataset_kind == "markov":
        return MarkovChainDataset(transition_matrix={'a':{'a':0.25,'b':0.25,'c':0.25,'d':0.25},'b':{'a':0.5,'b':0.2,'c':0.3},'c':{'b':1},'d':{'a':0.5,'b':0.5}}, initial_state='a', chain_length=cfg.mc_chain_length, dataset_length=cfg.mc_chain_dataset_length)
    else:
        raise ValueError(f"Unknown dataset kind: {cfg.dataset_kind}")

presets = {
    "forward": TrainingConfig(dataset_kind="forward"),
    "forward+reverse": TrainingConfig(dataset_kind = "forward+reverse", checkpoint_name="lit_end_fib_fwd+rev"),
    "modular": TrainingConfig(dataset_kind="modular", checkpoint_name="mod_fib", max_seq_len=32),
    "markov": TrainingConfig(dataset_kind="markov", checkpoint_name="markov", learning_rate=3e-4, weight_decay=0.01, dropout=0.1, max_seq_len=32, epochs = 500),
}

def main(cfg: TrainingConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokeniser = build_tokeniser(cfg)
    dataset = build_dataset(cfg)

    train_size = int(cfg.train_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_dataset_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_dataset_batch,
    )

    model = TinyCausalTransformer(
        vocab_size=len(tokeniser.vocab),
        max_seq_len=cfg.max_seq_len,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_hidden=cfg.d_hidden,
        dropout=cfg.dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimiser, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                checkpoint_dir / f"{cfg.checkpoint_name}.pt",
            )

        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    loss_fn: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)

        optimiser.zero_grad()
        logits = model(input_ids, attention_mask)

        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

        total_loss += loss.item()

    return total_loss / len(loader)

if __name__ == "__main__":
    main(presets["markov"])    

