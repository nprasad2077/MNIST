# MNIST Handwritten Digit Classification

A learning project for training a neural network on the MNIST dataset using PyTorch and DuckDB, with Jupyter notebooks for step-by-step visualization.

## Quick Start

```powershell
# 1. Run the setup script (creates venv, installs deps, extracts data)
.\setup.ps1

# 2. Open VSCode in this folder
code .

# 3. Open notebooks/01_data_loading.ipynb
#    Select kernel: "MNIST (Python)"
#    Run all cells
```

## Manual Setup

If you prefer to set up manually:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel for VSCode
python -m ipykernel install --user --name mnist --display-name "MNIST (Python)"

# Extract data (if not already done)
Expand-Archive -Path data\MNIST_ORG.zip -DestinationPath data -Force
```

## VSCode Notebook Setup

1. Install the **Jupyter** extension in VSCode (`ms-toolsai.jupyter`)
2. Open any `.ipynb` file
3. Click "Select Kernel" → "Jupyter Kernel" → **MNIST (Python)** (view kernels installed: `jupyter kernelspec list`. May need to refresh list.)
4. Run cells with `Shift+Enter`

## Project Structure

```
MNIST/
├── data/
│   ├── MNIST_ORG.zip              # Original compressed data
│   ├── train-images.idx3-ubyte    # 60,000 training images (28x28)
│   ├── train-labels.idx1-ubyte    # Training labels (0-9)
│   ├── t10k-images.idx3-ubyte     # 10,000 test images
│   ├── t10k-labels.idx1-ubyte     # Test labels
│   └── mnist.duckdb               # Generated after running notebook 01
├── notebooks/
│   ├── 01_data_loading.ipynb      # Parse IDX files → DuckDB, explore with SQL
│   └── 02_model_training.ipynb    # Train neural net, visualize results
├── requirements.txt
├── setup.ps1
└── README.md
```

## Notebooks

### 01 - Data Loading & DuckDB Storage
- Parses raw MNIST IDX binary format
- Stores all 70,000 images in DuckDB (queryable with SQL)
- Explores label distribution and pixel intensity
- Visualizes sample digits

### 02 - Model Training & Visualization
- Loads data from DuckDB into PyTorch
- Trains a fully-connected neural network (784 → 256 → 128 → 10)
- Tracks loss and accuracy per epoch
- Visualizes training curves, confusion matrix, and misclassified examples
- Saves model weights and training history

## Key Hyperparameters to Experiment With

| Parameter | Default | Effect |
|-----------|---------|--------|
| Learning Rate | 0.001 | Step size for optimization |
| Hidden Layers | 256, 128 | Model capacity |
| Dropout | 0.2 | Regularization strength |
| Batch Size | 128 | Gradient noise vs speed |
| Epochs | 15 | Training duration |

## Tech Stack

- **DuckDB** - In-process SQL database for data storage and exploration
- **PyTorch** - Neural network training framework
- **Matplotlib/Seaborn** - Visualization
- **Jupyter** - Interactive notebook environment in VSCode

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [DuckDB Documentation](https://duckdb.org/docs/)
