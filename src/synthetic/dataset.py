from tokenisation import CharacterTokeniser, DiscreteTokeniser
from generators import generate_little_endian_fib_datapoint, generate_modular_fib_datapoint, generate_reversed_fib_datapoint

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
    tokenised_datapoint = CharacterTokeniser.encode(self = CharacterTokeniser.build(), mode = "forward", text=datapoint)
    input = tokenised_datapoint[:-1]
    output = tokenised_datapoint[1:]
    target_train_pair = [input, output]
    return target_train_pair

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
    target_train_pair = [input, output]
    return target_train_pair

