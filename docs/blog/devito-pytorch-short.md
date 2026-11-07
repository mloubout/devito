# Put a Devito physics layer inside your PyTorch network

*A symbolic PDE, optimized native loops, and gradients that reach the layers
before the solver.*

A neural network can learn how to prepare a physical state, evolve it with a
numerical solver, then learn how to interpret the result. The interesting part
is making that solver a working layer in the training graph.

Here is the network in this example:

```python
import torch
from torch import nn
from examples.pytorch.devito_layer import DevitoDiffusion

network = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1),
    nn.Tanh(),
    DevitoDiffusion(steps=2, coefficient=0.1),
    nn.Conv2d(4, 1, kernel_size=1),
)

inputs = torch.randn(8, 1, 16, 16)
target = nn.functional.avg_pool2d(inputs, 3, stride=1, padding=1)
optimizer = torch.optim.Adam(network.parameters(), lr=0.03)

optimizer.zero_grad(set_to_none=True)
loss = nn.functional.mse_loss(network(inputs), target)
loss.backward()
optimizer.step()
```

Both convolutions learn. The first receives its gradient through the Devito
layer. Forward and backward execute generated C++ kernels; the layer supplies
an explicit discrete adjoint for its fixed diffusion stencil.

## PyTorch owns the tensors. Devito builds the loops.

The backend uses the ordinary Devito API. An allocator chooses tensor storage;
a language chooses CPU code generation:

```python
from devito import Eq, Grid, Operator, TimeFunction, TorchAllocator

grid = Grid(shape=(64, 64), extent=(63., 63.))
u = TimeFunction(name='u', grid=grid, space_order=2,
                 allocator=TorchAllocator())
u.data[0].fill_(1)
op = Operator(Eq(u.forward, u + 0.1*u.laplace), language='torch',
              opt=('advanced', {'openmp': True}))
op.apply(time_M=9)
assert isinstance(u.data, torch.Tensor)
```

Devito still expands the derivatives and optimizes its loop representation,
including blocking, SIMD and OpenMP. The generated C++ obtains typed pointers
from tensor storage and runs those loops. Existing tensors can also be bound
through `TorchAllocator(tensor=values)`.
Tensor views also retain Devito's buffered indexing and MPI decomposition rules.

## A complete training example

The accompanying demo trains the opening network for 40 optimizer updates on a
small image-smoothing task. Its loss curve is an integration result, not a
speed comparison or a scientific-model benchmark.

![Training loss for a network with learned convolutions before and after a
Devito diffusion layer.](assets/torch-layer-training.svg)

Input gradients are checked against a pure-PyTorch reference and finite
differences. The example supports CPU float32/float64 tensors, batching and
channels. It copies into padded tensor workspaces and returns independent
outputs. Its diffusion coefficient is fixed; the layer supplies first-order
input gradients for this specific stencil.

## The CUDA path uses existing OpenACC support

At the backend level, `TorchAllocator(device='cuda:0')` can supply tensor
pointers directly to `language='openacc', compiler='nvc'`. Devito emits
`deviceptr` clauses for those allocations. CUDA interoperability has runtime
coverage in the NVIDIA CI job. This is separate from the CPU-only example layer.

## Try it

Use the [feature branch](https://github.com/devitocodes/devito/tree/feature/torch-backend),
install a compatible PyTorch build and C++20 compiler, then run:

```sh
python -m pip install -e '.[torch,tests]'
OMP_NUM_THREADS=2 python -m examples.pytorch.devito_layer
```

The backend is under development. It does not automatically differentiate
arbitrary Devito programs. The useful building block is concrete: tensor-owned
storage, optimized symbolic kernels, and an explicit gradient rule that makes
a solver part of a trainable network.

[Read the full walkthrough: discretization, adjoint, generated code and CUDA
interoperability →](devito-pytorch.md)
