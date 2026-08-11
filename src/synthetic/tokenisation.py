from dataclasses import dataclass
from typing import Literal

Mode = Literal["forward", "reverse"]

# Dataclass defining tokenisation methods for forwards and reverse little-endian Fibonacci sequences
@dataclass(frozen=True)
class CharacterTokeniser:
    vocab: dict[str, int]
    inverse_vocab: dict[int, str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    reverse_token: str = "<rev>"
    forward_token: str = "<fwd>"
    width_token: str = "$"

    @classmethod
    #instantiates an instance of this class, setting a vocab and inverse vocab suitable for little-endian Fibonacci sequences.
    def build(cls) -> "CharacterTokeniser":
        tokens = ["<pad>", "<bos>", "<eos>", "$", "<rev>", "<fwd>", ",", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        vocab = {token: index for index, token in enumerate(tokens)}
        inverse_vocab = {index: token for token, index in vocab.items()}
        return cls(vocab=vocab, inverse_vocab=inverse_vocab)

    # The encoding method that makes this class a tokeniser, converting a string of characters into a list of token ids
    def encode(self, text: str, mode: Mode, add_special_tokens: bool = True) -> list[int]:
        if add_special_tokens:    
            token_ids = [self.vocab[self.bos_token]]
        else: 
            token_ids = []

        token_ids.append(self.vocab[self.forward_token] if mode == "forward" else self.vocab[self.reverse_token])

        for character in text:
            if character not in self.vocab:
                raise ValueError(f"Unknown character: {character}")
            token_ids.append(self.vocab[character])

        if add_special_tokens:
            token_ids.append(self.vocab[self.eos_token])

        return token_ids

    # The inverse of the encoding method
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        special_tokens = {self.pad_token, self.bos_token, self.eos_token, self.reverse_token, self.forward_token, self.width_token}
        characters = []
        for token_id in token_ids:
            character = self.inverse_vocab[token_id]
            if skip_special_tokens and character in special_tokens:
                continue
            characters.append(character) #type: ignore
        return "".join(characters) #type: ignore

# Dataclass defining tokenisation methods for the 'discrete' sequences (markov chains and modular Fibonacci sequences), similar to above
@dataclass(frozen=True)
class DiscreteTokeniser:
    vocab: dict[str, int]
    inverse_vocab: dict[int, str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    # Two build methods depending on the tokens that will be encountered.
    @classmethod
    def build_tokeniser_for_markov_chains(cls, chain_states: list[str]) -> "DiscreteTokeniser":
        tokens = [cls.pad_token, cls.bos_token, cls.eos_token, *chain_states]
        vocab = {token: index for index, token in enumerate(tokens)}
        inverse_vocab = {index: token for token, index in vocab.items()}
        return cls(vocab=vocab, inverse_vocab=inverse_vocab)

    @classmethod
    def build_tokeniser_for_modular_sequences(cls, mod: int) -> "DiscreteTokeniser":
        symbols : list[str] = list(map(str, range(mod)))
        tokens = [cls.pad_token, cls.bos_token, cls.eos_token, *symbols]
        vocab = {token: index for index, token in enumerate(tokens)}
        inverse_vocab = {index: token for token, index in vocab.items()}
        return cls(vocab=vocab, inverse_vocab=inverse_vocab)

    def encode(self, sequence: list[str], add_special_tokens: bool = True) -> list[int]:
        token_ids: list[int] = []
        if add_special_tokens:
            token_ids.append(self.vocab[self.bos_token])
        
        for symbol in sequence:
            if symbol not in self.vocab:
                raise ValueError(f"Unknown symbol: {symbol}")
            token_ids.append(self.vocab[symbol])
        
        if add_special_tokens:
            token_ids.append(self.vocab[self.eos_token])
        
        return token_ids
    
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> list[str]:
        special_tokens = {self.pad_token, self.bos_token, self.eos_token}
        symbols: list[str] = []
        for token_id in token_ids:
            symbol = self.inverse_vocab[token_id]
            if skip_special_tokens and symbol in special_tokens:
                continue
            symbols.append(symbol)
        return symbols
