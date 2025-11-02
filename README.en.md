# Banduk

## Rule Explanation

[Link to Banduk Rules](https://hanbat1mj.github.io/about-banduk-en)

## AlphaZero

Source code was partially used from [Learning AI by Analyzing AlphaZero.](https://github.com/Jpub/AlphaZero)

> Improvements
>
> - Handles Self-play and evaluate processes in parallel.
> - During Self-play, it creates a checkpoint file after a certain number of games are completed to prevent data loss if the process terminates for any reason (e.g., Colab disconnection)
> - Uses Dirichlet noise (for Self-play only) to increase the diversity of the initial search, allowing for broader exploration.
> - Drastically improved training speed by reducing the unnecessarily large model size.  
>   -> Significant reduction in self-play and evaluation time.  
>   => This makes it possible to train the model even on a Macbook.
> - Generates 8 data augmentations per game when collecting data during Self-play (rotation at 90, 180, 270 degrees, and rotations after a horizontal flip, for a total of 8)

## Total AlphaZero Training Time

> Ran `train_cycle.py` for 5 days on an Apple Silicon M3 Pro.
>
> Achived a performance level equivalent to a **high-intermediate Banduk player!**

## Example Game: AlphaZero vs. Human

![AI-vs-Human](/assets/AI_vs_Human.png)

- Black: Human (@hanbat1mj)
- White: AI

> 18 points : 18.5 points. `The human developer lost...`
>
> - (The 0.5 points is the handicap `komi` given to White.)

### Development Environments

- HW: Apple Silicon M3 Pro
- OS: macOS Sonoma 14.8.1
- Python: 3.10.17 (Anaconda)

### Key Package Versions

- TensorFlow: 2.16.2 (Optimized for Apple Silicon)
  - tensorflow-macos: 2.16.2
  - tensorflow-metal: 1.2.0
- NumPy: 1.26.4
- Keras: 3.10.0
- Ray: 2.46.0 (for parallel processing)
- h5py: 3.13.0
- Matplotlib: 3.10.3
- more-itertools: 10.7.0 (Used to find the positions of stones with the same ID in a list)

### How to Install

1. Install Anaconda (if you don't have it)
2. Create and activate a Conda environment

```bash
conda create -n alpha_zero_env python=3.10
conda activate alpha_zero_env
```

3. Install necessary packages

```bash
pip install tensorflow tensorflow-macos tensorflow-metal
pip install numpy keras ray h5py matplotlib
```

### How to Run

1. AI Starts as Black

```bash
python ai_first_play.py
```

2. Human starts as Black

```bash
python human_first_play.py
```

### How to Train the Model

```bash
python train_cycle.py
```
