# W-HGAD

Official implementation of W-HGAD: a Wasserstein-based heterogeneous graph neural network for uncertainty-aware anomaly detection on graphs.

## About

This repository contains the implementation of the W-HGAD model as described in our paper. W-HGAD is designed for uncertainty-aware anomaly detection on heterogeneous graphs, with a focus on the PolitiFact dataset.

## Contents

- `data/`: Directory containing the PolitiFact dataset
- `W-HGAD_PolitiFact.py`: Python script implementing W-HGAD for the PolitiFact dataset

## Key Dependencies

- PyTorch
- torch_geometric
- numpy
- scikit-learn

## Usage

To run the W-HGAD model on the PolitiFact dataset:

```
python W-HGAD_PolitiFact.py
```

Ensure all dependencies are installed before running the script.


## Contact

For questions or issues, please open an issue in this repository.

Thank you for your interest in W-HGAD!
