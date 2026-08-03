from tokenisation import CharacterTokeniser, DiscreteTokeniser
from generators import generate_little_endian_fib_datapoint, generate_modular_fib_datapoint, generate_reversed_fib_datapoint, generate_markov_chain
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

#---Character-Wise vocabulary training target pair generation functions---
def make_little_endian_fib_target_train_pair(a: int,b: int, n:int):
    """
    Creates a tokenised training target pair little endian Fibonacci sequence
    args:
        a: index 0 entry of Fibonacci sequence (big endian format integer)
        b: index 1 entry of Fibonacci sequence (big endian format integer)
        n: number of terms to produce in Fibonacci sequence
    """
    datapoint = generate_little_endian_fib_datapoint(a, b, n)[0]
    tokenised_datapoint = CharacterTokeniser.encode(self = CharacterTokeniser.build(), mode = "forward", text=datapoint)
    input = tokenised_datapoint[:-1]
    output = tokenised_datapoint[1:]
    target_train_pair = [input, output]
    return target_train_pair

def make_reversed_fib_training_target_pair(a: int, b: int, n: int):
    """
    Creates a tokenised training target pair character-reversed Fibonacci sequence (will result in little-endian numbers)
    args:
        a: index 0 entry of Fibonacci sequence (big endian format integer)
        b: index 1 entry of Fibonacci sequence (big endian format integer)
        n: number of terms to produce in Fibonacci sequence
    """
    datapoint = generate_reversed_fib_datapoint(a, b, n)[0]
    tokenised_datapoint = CharacterTokeniser.encode(self = CharacterTokeniser.build(), mode = "reverse", text=datapoint)
    input = tokenised_datapoint[:-1]
    output = tokenised_datapoint[1:]
    training_target_pair = [input, output]
    return training_target_pair

#---Discrete state vocabulary training target pair generation functions
def make_modular_fib_training_target_pair(a: int, b: int, n: int, mod: int):
    """
    Creates a tokenised training target pair modular Fibonacci sequence
    args:
        a: index 0 entry of Fibonacci sequence (given as non-modular integer)
        b: index 1 entry of Fibonacci sequence (given as non-modular integer)
        n: number of terms to produce in Fibonacci sequence
        mod: the modulo arithmetic value
    """
    datapoint = generate_modular_fib_datapoint(a, b, n, mod)
    tokenised_datapoint = DiscreteTokeniser.encode(self = DiscreteTokeniser.build_tokeniser_for_modular_sequences(mod), symbols = datapoint)
    input = tokenised_datapoint[:-1]
    output = tokenised_datapoint[1:]
    training_target_pair = [input, output]
    return training_target_pair

def make_markov_chain_training_target_pair(transition_matrix: dict[str, dict[str,float]], initial_state: str, length: int):
    """
    Creates a tokenised training target pair markov chain, using the transition matrix 
    args:
        transition_matrix: a matrix represented via nested dictionaries that defines the Markov chain to be used
        initial_state: a string representing the initial state of the Markov chain
        length: integer representing the length of the Markov chain to be produced
    """
    datapoint = generate_markov_chain(transition_matrix, initial_state, length)
    chain_states = [key for key in transition_matrix]
    tokenised_datapoint = DiscreteTokeniser.encode(self = DiscreteTokeniser.build_tokeniser_for_markov_chains(chain_states), symbols = datapoint)
    input = tokenised_datapoint[:-1]
    output = tokenised_datapoint[1:]
    training_target_pair = [input, output]
    return training_target_pair

class LittleEndianFibDataset(Dataset[dict[str,torch.Tensor]]):
    pairs: list[tuple[int,int]]
    def __init__(self, pairs: list[tuple[int, int]], n: int, tokeniser: CharacterTokeniser) -> None:
        self.pairs = pairs
        self.n = n
        self.tokeniser = tokeniser

    def __len__(self) -> int: 
        return len(self.pairs)

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

    attention_mask = (input_ids != 0).long()

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
    }
