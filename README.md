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

# Export  Notebook
python -m jupyter nbconvert --to markdown "notebook.ipynb"
```

## VSCode Notebook Setup

1. Install the **Jupyter** extension in VSCode (`ms-toolsai.jupyter`)
2. Open any `.ipynb` file
3. Click "Select Kernel" → "Jupyter Kernel" → **MNIST (Python)** (view kernels installed: `jupyter kernelspec list`. May need to refresh list.)
4. Run cells with `Shift+Enter`

## Project Structure

```powershell
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

| Parameter     | Default  | Effect                     |
| ------------- | -------- | -------------------------- |
| Learning Rate | 0.001    | Step size for optimization |
| Hidden Layers | 256, 128 | Model capacity             |
| Dropout       | 0.2      | Regularization strength    |
| Batch Size    | 128      | Gradient noise vs speed    |
| Epochs        | 15       | Training duration          |

## Tech Stack

- **DuckDB** - In-process SQL database for data storage and exploration
- **PyTorch** - Neural network training framework
- **Matplotlib/Seaborn** - Visualization
- **Jupyter** - Interactive notebook environment in VSCode

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [DuckDB Documentation](https://duckdb.org/docs/)

## Notebook 03 - CNN (Convolutional Neural Network)

The fully-connected network in notebook 02 treats each pixel independently, ignoring spatial relationships. The CNN in notebook 03 exploits the 2D structure of images to achieve significantly higher accuracy (~99.3% vs ~98.3%).

### Architecture

```markdown
Input (784 flat) → Reshape to (1, 28, 28)
→ Conv2d(1→32, 3×3) + BatchNorm + ReLU + MaxPool(2×2)   → (32, 14, 14)
→ Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool(2×2)  → (64, 7, 7)
→ Flatten → Linear(3136→128) + ReLU + Dropout(0.3)
→ Linear(128→10)
```

### Why CNNs Work Better for Images

| Feature                | Fully-Connected                            | CNN                                                |
| ---------------------- | ------------------------------------------ | -------------------------------------------------- |
| Spatial awareness      | None — each pixel is independent           | Conv filters detect local patterns (edges, curves) |
| Translation invariance | None — digit must be in exact position     | Pooling layers provide position tolerance          |
| Parameter efficiency   | 784×256 = 200k params in first layer alone | 3×3×32 = 288 params in first conv layer            |
| Typical MNIST accuracy | ~98.3%                                     | ~99.3%                                             |

### Key Enhancements Over Notebook 02

- **Convolutional layers** detect local spatial patterns regardless of position
- **MaxPooling** reduces spatial dimensions and provides translation invariance
- **BatchNorm2d** stabilizes training for conv layers (same idea as BatchNorm1d in notebook 02)
- **ReduceLROnPlateau scheduler** automatically halves learning rate when test loss plateaus

## Notebook 04 - Advanced CNN with Data Augmentation

Building on notebook 03, this pushes accuracy toward 99.5%+ with two key techniques:

### Data Augmentation

Each training image receives random transforms every epoch, effectively creating new training examples:

- **±10° rotation** — digits can be slightly tilted
- **±10% translation** — digits can shift within the frame
- **90-110% scaling** — digits can vary in size

This dramatically reduces overfitting since the model never sees the exact same image twice.

### Deeper Architecture (3 Conv Layers)

```markdown
Input (1, 28, 28)
→ Conv2d(1→32, 3×3) + BatchNorm + ReLU
→ Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
→ Conv2d(64→128, 3×3) + BatchNorm + ReLU + MaxPool(2×2)
→ Flatten → Linear(4608→256) + ReLU + Dropout(0.4)
→ Linear(256→10)
```

The third conv layer extracts higher-level features (digit parts, stroke junctions) that a 2-layer CNN cannot represent.

### Results Progression

| Notebook | Architecture                            | Test Accuracy |
| -------- | --------------------------------------- | ------------- |
| 02       | FC (784→256→128→10) + BatchNorm         | ~98.5%        |
| 03       | CNN (2 conv layers)                     | ~99.3%        |
| 04       | Deep CNN (3 conv layers) + Augmentation | ~99.5%+       |
