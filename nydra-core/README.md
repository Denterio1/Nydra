# Nydra Core (Rust)

High-performance statistical compute engine for the Nydra Autonomous Data Agent.

## Features
- Parallelized descriptive statistics (Mean, Std Dev, Skewness, Kurtosis)
- Robust outlier detection (IQR, MAD)
- Information theory kernels (Shannon Entropy)
- Parallel correlation matrices (Pearson, Spearman)

## Build Requirements
- Rust 1.78+
- Python 3.11+
- Maturin 1.5+

## Development
```bash
cd nydra-core
maturin develop --release
```
