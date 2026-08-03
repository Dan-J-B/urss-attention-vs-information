from typing import Literal
#---Helper functions---
def fib_next(a: int, b: int) -> int:
    return a + b

def generate_fib_sequence(a: int, b: int, n: int) -> list[int]:
    sequence = [a, b]
    for i in range(2, n):
        next_value = fib_next(sequence[i - 2], sequence[i - 1])
        sequence.append(next_value)
    return sequence

# Takes a list of integers and returns a string representation of the sequence reversed character-wise, meaning all big-endian numbers become little-endian and the sequence order is also reversed. Outputs a string with numeric values separated by commas
def reverse_sequence(sequence: list[int]) -> str:
    concatenated_sequence = ','.join(str(num) for num in sequence)
    reversed_sequence = concatenated_sequence[::-1]
    return reversed_sequence

# Takes a list of integers and returns a string with numeric values in little-endian format, separated by commas
def big_to_little_endian(num: int) -> str:
    characters = str(num)
    little_endian_characters = characters[::-1]
    return little_endian_characters

# Creates a modular Fibonacci sequence, returns a list of integers representing the Fibonacci sequence modulo the given mod value.m
def generate_modular_fib_sequence(a: int, b: int, n: int, mod: int) -> list[int]:
    sequence = [a % mod, b % mod]
    for i in range(2, n):
        next_value = fib_next(sequence[i - 2], sequence[i - 1]) % mod
        sequence.append(next_value)
    return sequence

#---Synthetic Datapoint generating functions---

# Creates a little-endian Fibonacci sequence data point ready for tokenisation, returns a tuple with the sequence string at index 0 and the direction 'forward' at index 1
def generate_little_endian_fib_datapoint(a: int, b: int, n: int) -> tuple[str, Literal["forward"]]:
    fib_sequence = generate_fib_sequence(a, b, n)
    little_endian_sequence = [big_to_little_endian(num) for num in fib_sequence]
    return (','.join(str(num) for num in little_endian_sequence), "forward")

# Creates a reversed Fibonacci sequence data point ready for tokenisation, returns a tuple with the sequence string at index 0 and the direction 'reverse' at index 1. Note again that this sequence is in little-endian format, as it was reversed character-wise. 
def generate_reversed_fib_datapoint(a: int, b: int, n: int) -> tuple[str, Literal["reverse"]]: 
    fib_sequence = generate_fib_sequence(a, b, n)
    reversed_fib_sequence = reverse_sequence(fib_sequence)
    return (reversed_fib_sequence, "reverse")

# Creates a Fibonacci sequence under a specified modular arithmetic. Outputs a string.
def generate_modular_fib_datapoint(a: int, b: int, n: int, mod: int) -> list[str]:
    modular_fib_sequence = generate_modular_fib_sequence(a, b, n, mod)
    modular_fib_datapoint = [str(num) for num in modular_fib_sequence]
    return modular_fib_datapoint

# Synthetic Data Generation for Markov Chains
import random

# 
def generate_markov_chain(transition_matrix: dict[str, dict[str, float]], initial_state: str, length: int) -> list[str]:
    current_state = initial_state
    chain = [current_state]
    for _ in range(length - 1):
        next_states = list(transition_matrix[current_state].keys())
        probabilities = list(transition_matrix[current_state].values())
        current_state = random.choices(next_states, weights=probabilities)[0]
        chain.append(current_state)
    return chain

# some tests
print(generate_reversed_fib_datapoint(1,1,15))
print(generate_little_endian_fib_datapoint(1,1,15))
print(generate_modular_fib_datapoint(1,1,15,10))
print(generate_markov_chain({'a':{'b':0.1,'c':0.9},'b':{'a':0.5,'b':0.2,'c':0.3},'c':{'a':0,'b':1}},'a',100))