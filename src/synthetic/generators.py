# Synthetic Data Generation for General Fibonacci sequences
def fib_next(a: int, b: int) -> int:
    return a + b

def generate_fib_sequence(a: int, b: int, n: int) -> list[int]:
    sequence = [a, b]
    for i in range(2, n):
        next_value = fib_next(sequence[i - 2], sequence[i - 1])
        sequence.append(next_value)
    return sequence

def reverse_sequence(sequence: list[int]) -> str:
    concatenated_sequence = ','.join(str(num) for num in sequence)
    reversed_sequence = concatenated_sequence[::-1]
    return reversed_sequence

def generate_fib_datapoint(a: int, b: int, n: int) -> tuple[str, bool]: #type: ignore
    fib_sequence = generate_fib_sequence(a, b, n)
    return (','.join(str(num) for num in fib_sequence), False)

def generate_reversed_fib_datapoint(a: int, b: int, n: int) -> tuple[str, bool]: 
    fib_sequence = generate_fib_sequence(a, b, n)
    reversed_fib_sequence = reverse_sequence(fib_sequence)
    return (reversed_fib_sequence, True)

def generate_modular_fib_sequence(a: int, b: int, n: int, mod: int) -> list[int]:
    sequence = [a % mod, b % mod]
    for i in range(2, n):
        next_value = fib_next(sequence[i - 2], sequence[i - 1]) % mod
        sequence.append(next_value)
    return sequence

# Synthetic Data Generation for Markov Chains
import random

def generate_markov_chain(transition_matrix: dict[str, dict[str, float]], initial_state: str, length: int) -> list[str]:
    current_state = initial_state
    chain = [current_state]
    for _ in range(length - 1):
        next_states = list(transition_matrix[current_state].keys())
        probabilities = list(transition_matrix[current_state].values())
        current_state = random.choices(next_states, weights=probabilities)[0]
        chain.append(current_state)
    return chain
