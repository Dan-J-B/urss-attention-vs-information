import sys

print("Python executable:", sys.executable)

try:
    import torch
except ImportError as e:
    print("PyTorch import failed:", e)
    raise SystemExit(1)

print("PyTorch version:", torch.__version__)
print("Torch file:", torch.__file__)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))

# Basic tensor test
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
c = a + b

print("Tensor a:", a)
print("Tensor b:", b)
print("a + b:", c)

# Small matrix multiply test
x = torch.randn(2, 3)
y = torch.randn(3, 4)
z = x @ y

print("Matrix multiply output shape:", z.shape)
print("All checks passed.")