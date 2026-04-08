# W-HGAD

**Official implementation of W-HGAD**, as presented in our ICASSP 2025 paper:

> **Wasserstein Heterogeneous Graph Neural Networks for Uncertainty-Aware Anomaly Detection**  
> Chen Chen, Yunchun Li, Boxuan Jiao, Guorui Zhao, Wei Li  
> *ICASSP 2025 – IEEE International Conference on Acoustics, Speech and Signal Processing*

---

## Overview

W-HGAD is a Wasserstein-based heterogeneous graph neural network for uncertainty-aware anomaly detection on graphs. It leverages optimal transport theory to model distributional uncertainty across heterogeneous graph structures, with evaluation on the PolitiFact dataset.

## Repository Structure

```
W-HGAD/
├── data/                      # PolitiFact dataset
└── W-HGAD_PolitiFact.py       # Main implementation
```

## Requirements

- Python ≥ 3.8
- PyTorch
- torch_geometric
- numpy
- scikit-learn

Install dependencies via:

```bash
pip install torch torch_geometric numpy scikit-learn
```

## Usage

To run W-HGAD on the PolitiFact dataset:

```bash
python W-HGAD_PolitiFact.py
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{chen2025wasserstein,
  title={Wasserstein Heterogeneous Graph Neural Networks for Uncertainty-Aware Anomaly Detection},
  author={Chen, Chen and Li, Yunchun and Jiao, Boxuan and Zhao, Guorui and Li, Wei},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## Contact

For questions or issues, please open a GitHub issue in this repository.
