from tokenisation import *
from dataset import *
from synthetic_models import *
from torch.utils.data import DataLoader
import itertools

tokeniser = CharacterTokeniser.build()

pairs = list(itertools.combinations_with_replacement(range(1,100),2))

dataset = LittleEndianFibDataset(
    pairs=pairs,
    n=15,
    tokeniser=tokeniser
)

loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_little_endian_batch,
)