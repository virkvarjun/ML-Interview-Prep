# ML Interview Prep

Hands-on Jupyter notebooks for machine learning interview preparation, covering **NumPy**, **PyTorch**, and **JAX**.

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Register the kernel for Jupyter
python -m ipykernel install --user --name=ml-interview-prep --display-name="ML Interview Prep"

# Launch JupyterLab
jupyter lab
```

## Notebooks

### NumPy (`notebooks/numpy/`)
| Notebook | Topics |
|----------|--------|
| `01_array_basics.ipynb` | Creation, indexing, slicing, reshaping |
| `02_broadcasting_vectorization.ipynb` | Broadcasting rules, vectorized ops, performance |
| `03_linear_algebra.ipynb` | Dot products, norms, eigendecomposition, SVD |

### PyTorch (`notebooks/pytorch/`)
| Notebook | Topics |
|----------|--------|
| `01_tensors_autograd.ipynb` | Tensor ops, `requires_grad`, backward pass |
| `02_neural_networks.ipynb` | `nn.Module`, layers, activations, loss functions |
| `03_training_loop.ipynb` | DataLoader, optimizer, training/eval loop |

### JAX (`notebooks/jax/`)
| Notebook | Topics |
|----------|--------|
| `01_functional_arrays.ipynb` | Immutable arrays, random keys, functional style |
| `02_jit_vmap_grad.ipynb` | `jit`, `vmap`, `grad`, `value_and_grad` |
| `03_neural_networks_jax.ipynb` | Flax-style pure functions, training with Optax |

## Interview Tips

- **NumPy**: Know broadcasting rules cold. Be able to implement matrix multiply from scratch.
- **PyTorch**: Understand the computation graph, `detach()`, in-place ops, and GPU transfer.
- **JAX**: Emphasize immutability, pure functions, `jit`/`vmap`/`grad` transforms, and PRNG keys.
