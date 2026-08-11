from tokenisation import CharacterTokeniser, DiscreteTokeniser
from generators import generate_little_endian_fib_datapoint, generate_markov_chain, generate_modular_fib_datapoint, generate_reversed_fib_datapoint
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

#-------Little Endian Forward Fibonacci Dataset and Collate Function-------
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
def collate_dataset_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]: 
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

#-------Forward+Reverse Little Endian Fibonacci Dataset and Collate Function-------
class LittleEndianFibDatasetForwardReverse(Dataset[dict[str,torch.Tensor]]):
    pairs: list[tuple[int,int]]
    def __init__(self, pairs: list[tuple[int, int]], n: int, tokeniser: CharacterTokeniser) -> None:
        self.pairs = pairs
        self.n = n
        self.tokeniser = tokeniser

    def __len__(self) -> int: 
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, b = self.pairs[idx]
        # Randomly choose to generate a forward or reverse sequence
        if torch.rand(1).item() < 0.5:
            text, mode = generate_little_endian_fib_datapoint(a, b, self.n)
        else:
            text, mode = generate_reversed_fib_datapoint(a, b, self.n)
        token_ids = self.tokeniser.encode(text=text, mode=mode)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }

#-------Modular Fibonacci Dataset and Collate Function-------
class ModularFibDataset(Dataset[dict[str,torch.Tensor]]):
    pairs: list[tuple[int,int]]
    def __init__(self, pairs: list[tuple[int, int]], n: int, mod: int) -> None:
        self.pairs = pairs
        self.n = n
        self.mod = mod
        self.tokeniser = DiscreteTokeniser.build_tokeniser_for_modular_sequences(mod = mod)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, b = self.pairs[idx]

        text = generate_modular_fib_datapoint(a, b, self.n, self.mod)
        token_ids = self.tokeniser.encode(sequence=list(text))
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }

#-------Markov Chain Dataset and Collate Function-------
class MarkovChainDataset(Dataset[dict[str,torch.Tensor]]):
    def __init__(self, transition_matrix: dict[str, dict[str, float]], initial_state: str, chain_length: int, dataset_length: int) -> None:
        self.transition_matrix = transition_matrix
        self.initial_state = initial_state
        self.chain_length = chain_length
        self.dataset_length = dataset_length
        self.tokeniser = DiscreteTokeniser.build_tokeniser_for_markov_chains(chain_states=list(transition_matrix.keys()))

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sequence = generate_markov_chain(self.transition_matrix, self.initial_state, self.chain_length)
        token_ids = self.tokeniser.encode(sequence=sequence)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }

    

    