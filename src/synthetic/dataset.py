from tokenisation import CharacterTokeniser
from generators import generate_little_endian_fib_datapoint
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Creating a custom Dataset object (inherits class properties and methods of 'Dataset' class)
class LittleEndianFibDataset(Dataset[dict[str,torch.Tensor]]):
    pairs: list[tuple[int,int]]
    # Custom initialisation method, populates the class with objects and values necessary in __len__ and __getitem__
    def __init__(self, pairs: list[tuple[int, int]], n: int, tokeniser: CharacterTokeniser) -> None:
        self.pairs = pairs
        self.n = n
        self.tokeniser = tokeniser

    def __len__(self) -> int: 
        return len(self.pairs)

    # Generates, tokenises and forms input target pair out of synthetic data
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, b = self.pairs[idx]
        text, mode = generate_little_endian_fib_datapoint(a, b, self.n)
        token_ids = self.tokeniser.encode(text=text, mode=mode)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }

# Collate function for use in DataLoader padding to uniform length within a batch and initialising an attention tensor for masking the padding tokens
def collate_little_endian_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]: 
    input_ids = pad_sequence(
        [sample["input_ids"] for sample in samples],
        batch_first=True,
        padding_value=0,
    )
    target_ids = pad_sequence(
        [sample["target_ids"] for sample in samples],
        batch_first=True,
        padding_value=-100,
    )
    # A boolean tensor indicating the padding ids positions
    attention_mask = (input_ids != 0).long()

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
    }
