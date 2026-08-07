from __future__ import annotations

from tokenisation import CharacterTokeniser
from dataset import LittleEndianFibDataset, collate_little_endian_batch
from synthetic_models import TinyCausalTransformer

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from pathlib import Path
import itertools

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokeniser = CharacterTokeniser.build()
    pairs = list(itertools.combinations_with_replacement(range(1,3),2))

    dataset = LittleEndianFibDataset(
        pairs=pairs,
        n=5,
        tokeniser=tokeniser
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_little_endian_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_little_endian_batch,
    )

    model = TinyCausalTransformer(
        vocab_size=len(tokeniser.vocab),
        max_seq_len=128,
        d_model=512,
        n_layers=2,
        n_heads=4,
        d_hidden=512,
        dropout=0.0,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.15)

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(12000):
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
                checkpoint_dir / "best.pt",
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
    main()    

