# A Devito physics layer inside a PyTorch network

*Learned features in, a generated diffusion kernel in the middle, gradients
flowing all the way back to the encoder.*

A physical model becomes much more useful to a learning system when it can sit
inside the network itself. A learned encoder can prepare the state, a numerical
solver can evolve it, and a learned decoder can turn the result into a
prediction. Training then needs to cross the solver in both directions.

That is the example we will build here:

```python
from torch import nn
from examples.pytorch.devito_layer import DevitoDiffusion

network = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1),
    nn.Tanh(),
    DevitoDiffusion(steps=2, coefficient=0.1),
    nn.Conv2d(4, 1, kernel_size=1),
)
```

The middle module runs Devito-generated C++ in its forward pass. Its explicit
backward rule also runs Devito-generated C++, allowing a loss to update the
convolution **before** the physics layer as well as the one after it.

The physics module is fixed; the surrounding convolutions are learned. For this
particular diffusion discretization, the backward operator has a simple, exact
form. We will derive that rule, check it numerically, and train the resulting
network.

Underneath the layer, the integration still uses ordinary Devito Functions,
`TorchAllocator`, and `language='torch'`. Numerical buffers stay in PyTorch-owned
storage, while Devito supplies its symbolic discretization and optimized loops.

This is a development-branch example, not a feature claim for an unmodified
upstream Devito installation. Run the examples from this checkout. The layer is
CPU-only and provides first-order input gradients for the specific linear
stencil described below; arbitrary Devito Operators do not acquire automatic
differentiation by selecting the torch language.

[Want the quick version? Read the short overview →](devito-pytorch-short.md)

## Start with a field, finish with a tensor

Consider the diffusion equation

\[
\frac{\partial u}{\partial t} = \alpha\nabla^2u.
\]

We will use a forward Euler time step and centered spatial differences. PyTorch
will build the initial Gaussian; Devito will expand the derivatives and generate
the loops that advance it.

From this checkout, install a compatible PyTorch CPU or CUDA build using
PyTorch's installation instructions, followed by:

```sh
python -m pip install -e '.[torch,tests]'
```

This branch follows the Python, NumPy and SymPy requirements of Devito main.
CPU code generation needs a C++20 compiler. The examples use Devito's configured
compiler, including GCC on Linux and Clang on Apple Silicon.

Here is a complete CPU example:

```python
import torch
from devito import Eq, Grid, Operator, TimeFunction, TorchAllocator

# Unit spacing makes the discretization and time-step choice easy to inspect.
grid = Grid(shape=(64, 64), extent=(63.0, 63.0))
u = TimeFunction(name='u', grid=grid, time_order=1, space_order=2,
                 allocator=TorchAllocator())

axis = torch.linspace(-1.0, 1.0, 64)
xx, yy = torch.meshgrid(axis, axis, indexing='ij')
initial = torch.exp(-20.0 * (xx.square() + yy.square()))
u.data[0].copy_(initial)

alpha, dt = 0.5, 0.1
op = Operator(
    Eq(u.forward, u + dt * alpha * u.laplace),
    language='torch',
    opt=('advanced', {'openmp': True}),
)

allocation = u._data_allocated.data_ptr()
steps = 50
op.apply(time_M=steps - 1)

# Buffered TimeFunction indexing wraps through the two stored time levels.
final_field = u.data[steps]
assert u._data_allocated.data_ptr() == allocation
assert torch.isfinite(final_field).all()
assert final_field.max() < initial.max()

print('Peak:', final_field.max().item())
print('Squared norm:', final_field.square().sum().item())
```

![Initial and evolved Gaussian fields, with a center-line profile showing the
lower peak and wider distribution after diffusion.](assets/torch-diffusion.svg)

*Results from the code above: 50 steps, `dt=0.1`, and `alpha=0.5`.
The final peak is approximately 0.826. Both heatmaps use the same color scale.*

The time step satisfies the usual stability bound for this particular explicit
2D diffusion scheme with unit spacing. The outer halo starts at zero and is not
updated by the domain equation, providing zero ghost-cell values. Different
PDEs, discretizations, or boundary conditions need their own numerical choices.

There is no conversion of the evolving field to a NumPy array. Initialization,
inspection, and reductions use tensor operations. The generated kernel writes
the same allocation throughout the time loop.

## Turn the solver into an actual layer

The standalone field shows the storage and compiler path. A network needs an
additional contract: a forward map on `[batch, channels, height, width]` tensors
and a backward map that returns the gradient with respect to that input.

`DevitoDiffusion` treats each batch/channel plane independently and applies the
same fixed diffusion step several times. Its complete implementation is in
[examples/pytorch/devito_layer.py](https://github.com/devitocodes/devito/blob/feature/torch-backend/examples/pytorch/devito_layer.py).
It uses the standard allocator and Operator API internally.

The implementation flattens batch and channel into an independent `plane`
Dimension. A two-buffer `TimeFunction` carries those planes plus the spatial
dimensions. The generated equation is still simply:

```python
Eq(state.forward, state + coefficient * state.laplace)
```

Here `coefficient` is `alpha * dt` for unit grid spacing. The layer accepts CPU
float32 and float64 tensors and requires the coefficient to lie in `[0, 0.25]`,
the stability range for this 2D explicit scheme. The number of steps and the
coefficient are fixed configuration, not trainable tensors.

A compiled plan is cached for each shape, dtype, and physics configuration.
Every call gets its own padded workspace and output tensor. That keeps one
forward call from overwriting another call's activations before backward runs.
It also means this example layer copies data into and out of its workspaces;
it is not an end-to-end zero-copy layer. The numerical buffers remain tensors.

The module wiring is conventional PyTorch. With input validation omitted from
this excerpt, its essential structure is:

```python
class DevitoDiffusion(nn.Module):
    def __init__(self, steps=3, coefficient=0.1):
        super().__init__()
        self.steps = steps
        self.coefficient = coefficient

    def forward(self, value):
        batch, channels, height, width = value.shape
        plan = _plan(batch * channels, height, width, value.dtype,
                     self.steps, self.coefficient)
        return _DiffusionFunction.apply(value, plan)
```

`_plan` supplies the cached Devito Operator and workspace construction. The
autograd function below supplies the forward/backward contract. Both are
included alongside the full module in the example source.

## The backward pass is a discrete adjoint

For a single spatial plane, write one diffusion step as `A = I + c L`, where `L`
is the five-point discrete Laplacian. After `m` steps, the forward map is
`F(x) = A^m x`.

For an upstream gradient `g`, the required input gradient is `(A^m)^T g`.
With constant `c`, a uniform unit-spaced grid, and the same zero-ghost boundary
treatment used in forward, this particular matrix `A` is symmetric. Therefore
`(A^m)^T g = A^m g`: we can run the same compiled stencil on the upstream gradient.

The autograd bridge is small. This is the bridge used by the example module:

```python
class _DiffusionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, plan):
        ctx.plan = plan
        return plan.run(value)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return ctx.plan.run(grad_output), None
```

`plan.run` creates a fresh workspace, copies in detached tensor values, executes
the cached Devito Operator, and returns an independent output. The bridge tells
PyTorch how to connect that native computation to the surrounding graph. The
complete source imports `once_differentiable` from
`torch.autograd.function` and constructs the plan inside `DevitoDiffusion.forward`.

No forward trajectory is needed for this linear, constant-coefficient input
gradient: the backward map does not depend on the input values. Forward and
backward each use their own two-buffer time stepping.

This symmetry argument belongs to the **discretized operator**, including its
boundary treatment. Spatially varying coefficients, other boundary conditions,
nonlinear equations, or gradients with respect to physical parameters need a
separately derived backward rule. The example deliberately supports first-order
input gradients only; it does not implement higher-order differentiation.

## Train across the Devito layer

Now the opening network can be trained with an ordinary PyTorch optimizer.
The toy task below maps random input images to a local average. It makes the
interface and gradient path easy to test without an external dataset.

```python
import torch
from torch import nn
from examples.pytorch.devito_layer import DevitoDiffusion

torch.manual_seed(7)
inputs = torch.randn(8, 1, 16, 16)
target = nn.functional.avg_pool2d(inputs, 3, stride=1, padding=1)

network = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3, padding=1),
    nn.Tanh(),
    DevitoDiffusion(steps=2, coefficient=0.1),
    nn.Conv2d(4, 1, kernel_size=1),
)
optimizer = torch.optim.Adam(network.parameters(), lr=0.03)
losses = []

for _ in range(40):
    optimizer.zero_grad(set_to_none=True)
    prediction = network(inputs)
    loss = nn.functional.mse_loss(prediction, target)
    loss.backward()

    # These weights are upstream of Devito: their gradient crosses the solver.
    assert torch.isfinite(network[0].weight.grad).all()
    assert network[0].weight.grad.abs().sum() > 0

    losses.append(loss.item())
    optimizer.step()

with torch.no_grad():
    losses.append(nn.functional.mse_loss(network(inputs), target).item())
```

![Training loss for the network containing a Devito diffusion layer, showing
both learned convolutions connected through the explicit adjoint.](assets/torch-layer-training.svg)

*One seeded CPU run: MSE fell from approximately 0.1276 to 0.0081 over 40 optimizer
updates. This is a small integration demonstration, not a performance benchmark
or a claim about a trained scientific model.*

The gradients update both learned convolutions. They cross the fixed diffusion
layer through its explicit adjoint; they do not update `coefficient` or `steps`.

You can run the complete demo directly:

```sh
DEVITO_LOGGING=ERROR OMP_NUM_THREADS=2 python -m examples.pytorch.devito_layer
```

The backward rule is checked against a pure-PyTorch grouped-convolution reference
and finite differences with `torch.autograd.gradcheck`. Tests also cover multiple
forward calls before backward and noncontiguous input tensors. A decreasing loss
alone would not establish that the input gradient is correct.

## Bring a tensor you already own

Often the tensor exists before Devito enters the program. `TorchAllocator` can
bind a Function directly to that allocation, following the existing
`ExternalAllocator` pattern:

```python
import torch
from devito import Eq, Function, Grid, Operator, TorchAllocator

values = torch.randn(64, 64)
expected = 2 * values + 1
pointer = values.data_ptr()

f = Function(
    name='f', grid=Grid(shape=tuple(values.shape)), space_order=0,
    allocator=TorchAllocator(tensor=values),
    initializer=lambda _: None,  # Preserve the tensor's existing contents.
)

transform = Operator(Eq(f, 2 * f + 1), language='torch')
transform.apply()

assert f.data.data_ptr() == pointer
assert values.data_ptr() == pointer
torch.testing.assert_close(values, expected)
```

`values` now contains the result. The no-op initializer matters: normal Devito
allocation initializes field data, and binding existing storage should not
silently erase the values we want to compute on.

This example uses `space_order=0`, so the allocation is just the domain. A field
with halo cells or padding needs a tensor matching its **full allocated shape**,
including those regions. Its domain view may be noncontiguous even though the
underlying allocation is contiguous.

The usual runtime override also works:

```python
replacement = torch.zeros_like(values)
transform.apply(f=replacement)
torch.testing.assert_close(replacement, torch.ones_like(replacement))
```

The storage contract is explicit: matching dtype and device, contiguous storage,
correct layout, and sufficient alignment. An incompatible tensor produces an
error instead of an implicit conversion or device copy.

## What actually gets compiled?

The CPU path retains Devito's loop representation and optimization machinery.
For a simple `Eq(u.forward, u + 1)` operator with `space_order=0`, the generated
C++ contains a tensor view and a typed pointer, followed by ordinary stencil
loops. Abbreviated to show the essential operations:

```cpp
float *u __attribute__((aligned(64))) = at::from_blob(
    u_vec->data,
    {static_cast<int64_t>(u_vec->size[0]),
     static_cast<int64_t>(u_vec->size[1]),
     static_cast<int64_t>(u_vec->size[2])},
    at::TensorOptions().dtype(at::kFloat).device(at::kCPU)).data_ptr<float>();

for (int time = time_m; time <= time_M; ++time) {
    const int current = time % 2;
    const int next = (time + 1) % 2;
    for (int x = x_m; x <= x_M; ++x)
        for (int y = y_m; y <= y_M; ++y)
            uL0(next, x, y) = uL0(current, x, y) + 1;
}
```

`from_blob` creates a non-owning view. PyTorch retains ownership of the original
allocation; creating the view does not copy its numerical contents. The `uL0`
macro represents Devito's flattened indexing, including allocation strides and,
where needed, halo offsets in the indexed expressions.

For more involved equations, the same pipeline expands finite differences,
performs symbolic simplification and common-subexpression elimination, and
applies its loop optimizations. The torch language uses flat indexing to avoid
C variable-length arrays in C++. SIMD and optional OpenMP parallelism remain
part of Devito's code generation.

You can inspect the complete source directly:

```python
print(op.ccode)
```

The kernel is compiled through Devito's native JIT. The compiler configuration
supplies the Torch headers, libraries, and C++ ABI setting. There is no separate
extension-building step for the user.

These are architectural capabilities, not benchmark results. Numerical parity
and functional tests do not establish a speedup over PyTorch operators, another
Devito target, or a hand-written kernel. Performance comparisons need measured
workloads, controlled thread counts, appropriate warm-up, and separate reporting
of compilation and execution time.

## CUDA tensors and OpenACC

Storage and code generation are separate choices. CPU code can use the torch
language; GPU tensor storage can use Devito's existing OpenACC language.

The following example binds an existing CUDA tensor and updates it directly:

```python
import torch
from devito import Eq, Function, Grid, Operator, TorchAllocator

values = torch.randn(256, 256, device='cuda:0')
expected = values * 0.5
pointer = values.data_ptr()

f = Function(
    name='f', grid=Grid(shape=tuple(values.shape)), space_order=0,
    allocator=TorchAllocator(tensor=values), initializer=lambda _: None,
)
scale = Operator(Eq(f, 0.5 * f), platform='nvidiaX',
                 language='openacc', compiler='nvc')
scale.apply()

assert values.data_ptr() == pointer
torch.testing.assert_close(values, expected)
```

The allocator identifies device-resident storage. OpenACC then emits
`deviceptr(f)` clauses and omits the usual host-to-device mapping and copy
operations for that Function. The kernel accesses the CUDA allocation owned by
PyTorch. Host-resident inputs in the same operator retain the existing OpenACC
data-management behavior.

PyTorch and OpenACC can use different CUDA streams. The current implementation
synchronizes before and after native execution and checks that the OpenACC
device matches the tensor's device. This provides a synchronous interoperability
path; it is not a promise of asynchronous overlap.

CUDA execution is covered in the existing NVIDIA OpenACC CI job, including
float32/float64 stencils, external tensors, runtime overrides, mixed host/device
inputs and nondefault producer streams. The job requires working CUDA PyTorch
and `nvc++`; a missing CUDA runtime fails its prerequisite check. The optional
second-device test requires two visible GPUs.

Choose a PyTorch wheel compatible with the NVIDIA driver. The CI image uses the
official CUDA 12.4 index for its current driver. The compiler's library loader
binds NVIDIA kernels to their linked OpenACC runtime, avoiding interposition
from GCC's `libgomp` loaded by Torch. Apple CPU compilation likewise reuses an
already-loaded LLVM OpenMP runtime, including when selecting synthetic CPU
targets. These are toolchain concerns handled beneath the ordinary Operator API.

The example `DevitoDiffusion` layer remains CPU-only. CUDA tensor interoperability
at the backend level does not supply a CUDA autograd layer or an adjoint for an
arbitrary PDE.

## Tensor storage with Devito's indexing and MPI behavior

The tensor implementation also participates in Devito's data abstraction:

```text
Data
├── NumpyData(Data, np.ndarray)
└── TorchData(Data, torch.Tensor)
```

The shared `Data` base owns modulo indexing, decomposition metadata, global/local
index translation, distributed assignment, and gather. Numerical storage and
local indexing operations are supplied by each backend.

This is why the buffered time access `u.data[steps]` in the diffusion example
works. The tensor view inherits Devito's modulo rules. Slicing a distributed
field preserves its decomposition metadata, and tensor arithmetic results keep
the relevant metadata too.

The same design supports MPI execution. With an MPI runtime and `.[mpi]`
installed, the diffusion example can be launched with multiple processes:

```sh
DEVITO_MPI=basic OMP_NUM_THREADS=2 mpiexec -n 2 python diffusion.py
```

Here `diffusion.py` is the standalone Gaussian diffusion example above, saved
as a script. This MPI path is separate from the autograd layer; the example
`DevitoDiffusion` module does not support distributed execution.
Each process builds a local field for its domain partition. For distributed
initialization, initialize the domain through Devito's shared assignment rules:
replace `u.data[0].copy_(initial)` with `u.data[0] = initial`. That assignment
selects each rank's portion of the replicated initial tensor. Local field views
and reductions remain local; a global result can be assembled explicitly with
`u.data_gather(rank=0)`.

CPU tensor communication uses host views that share its storage. CUDA MPI
communication currently uses explicit host staging and synchronization; it does
not assume CUDA-aware MPI. That transport choice is distinct from the OpenACC
kernel path, which consumes CUDA tensor pointers directly.

The inheritance makes the communication logic reusable. It does not erase
backend differences: PyTorch cannot represent negative-stride views, so reverse
reads materialize copies; direct reverse-slice assignment writes back to the
original storage. Read-only accessors return independent tensor snapshots.

## What has been demonstrated

The example now goes beyond sharing tensor storage: a standard `nn.Module`
executes generated Devito kernels in forward and backward, and a network with
learned operations on both sides can train through it. Its input gradients match
a PyTorch reference and pass finite-difference checks in float64. The layer tests
also exercise float32, batching, channels, noncontiguous input, and independent
workspaces for repeated calls.

The underlying backend validation covers CPU C++ generation, blocked 3D stencils,
OpenMP numerical parity, existing tensor bindings, runtime overrides, and storage
validation. The shared-data tests cover multi-rank MPI halo exchange, gather, reverse
slicing, and generated MPI stencil execution.
Those backend MPI capabilities are distinct from the CPU-only layer example.

The scope remains explicit: this is a validated first-order adjoint for one fixed
linear operator, not automatic differentiation of arbitrary PDE programs. Trainable physical coefficients, higher-order derivatives, and performance
comparisons remain separate work. Within that scope, the
connection is useful and concrete: a symbolic discretization can become a
working computational layer inside a PyTorch network.


## Reproduce and inspect

The examples and backend in this article track the
[feature branch](https://github.com/devitocodes/devito/tree/feature/torch-backend).
The [layer source](https://github.com/devitocodes/devito/blob/feature/torch-backend/examples/pytorch/devito_layer.py)
and [backend guide](https://github.com/devitocodes/devito/blob/feature/torch-backend/docs/torch.md)
provide the implementation and API details.

```sh
OMP_NUM_THREADS=2 python -m pytest -q tests/test_torch_layer.py
OMP_NUM_THREADS=2 python -m pytest -q tests/test_torch_codegen.py tests/test_torch_allocators.py
OMP_NUM_THREADS=2 python -m pytest -q tests/test_data_backends.py
# Requires a compatible CUDA PyTorch installation, NVIDIA GPU and HPC SDK:
OMP_NUM_THREADS=2 python -m pytest -q tests/test_torch_openacc.py -k cuda
```

The current branch has green CPU, MPI, NVIDIA GPU, notebook, example and lint
workflows. That is functional validation for the tested configurations, not a
performance claim or a guarantee for every compiler and driver combination.
The source data beside the figures records the CPU training run shown here.

This revision was checked against
[backend f1d5df9ac](https://github.com/devitocodes/devito/commit/f1d5df9ac), including
the [NVIDIA runtime workflow](https://github.com/devitocodes/devito/actions/runs/42615875272).
