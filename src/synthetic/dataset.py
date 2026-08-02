from tokenisation import CharacterTokeniser, DiscreteTokeniser
from generators import generate_little_endian_fib_datapoint, generate_modular_fib_datapoint, generate_reversed_fib_datapoint, generate_markov_chain

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
