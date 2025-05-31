# Genetic Programming with Indexed Memory for Partially Observable CartPole

This repository contains an implementation of Genetic Programming (GP) augmented with indexed memory to solve a partially observable version of the CartPole environment from Gymnasium.

## Overview

The goal is to evolve policies using GP that leverage external memory to compensate for missing velocity observations in the CartPole environment, where only cart position and pole angle are observable. The system is implemented using the DEAP library and interfaces with Gymnasium.

## Key Features

- **Partially Observable Environment**: A custom `PartialObservationWrapper` modifies the CartPole-v1 environment to provide only `[cart_position, pole_angle]`, omitting velocities.
- **Indexed Memory**: A fixed-size memory vector (size=4) stores floating-point values, reset to zeros per episode, and persists across time steps within an episode.
- **GP Primitives**:
  - **Memory Operations**:
    - `mem_read(index)`: Reads from memory at `int(index) % memory_size`.
    - `mem_write(index, value)`: Writes value to memory at `int(index) % memory_size` and returns value.
  - **Arithmetic**: `add`, `sub`, `mul`, `protectedDiv`.
  - **Trigonometric**: `sin`, `cos`.
  - **Conditional**: `if_then_else(condition, out1, out2)` returns `out1` if `condition > 0`, else `out2`.
  - **Ephemeral Constants**: Random floats in `[-1, 1]`.
- **Action Selection**: GP tree outputs a float, thresholded at 0 to select discrete actions (0: left, 1: right).
- **Fitness**: Average reward over 20 episodes (max 500 steps per episode).

## Implementation Details

- **DEAP Setup**:
  - Population: 100 individuals.
  - Generations: 50.
  - Crossover Probability: 0.8.
  - Mutation Probability: 0.2.
  - Selection: Tournament (size=3).
  - Tree Constraints: Max height of 17.

- **Memory Handling**: 
  - Memory is a global list within `evalRL`, ensuring each multiprocessing worker has its own instance.
  - No autoregressive state is hardcoded, adhering to assignment constraints.

## Experiments

- **With Memory**: Full primitive set including `mem_read` and `mem_write`.
- **Without Memory**: Excludes memory primitives for baseline comparison.
- **Fully Observable**: Uses all four observation variables (baseline from `genetic_programming_1_400611523.ipynb`).

## Results

- **Training Curves**: Plotted in `gp_indexed_memory_400611523.ipynb`:
  - Indexed Memory: Converges to ~500 fitness by generation 20, matching the fully observable policy.
  - No Memory: Stagnates at low fitness (~50-100), unable to compensate for missing velocities.
  - Fully Observable: Reaches ~500 fitness, serving as an upper bound.

- **Best Tree**: Visualized using PyGraphviz, showing complex use of memory and conditionals (saved as `best_tree.png`).

## Dependencies

- Python 3.10
- deap==1.4.1
- gymnasium[classic-control]==1.0.0
- numpy==1.26.4
- matplotlib
- pygraphviz
- pygame==2.6.1

## Running the Code

1. Install dependencies: `pip install -r requirements.txt`.
2. Run the notebook: `jupyter notebook gp_indexed_memory_400611523.ipynb`.
   - Executes both experiments and generates plots.

## Approach

- **Environment**: Wrapped CartPole-v1 to limit observations.
- **GP System**: Extended DEAP’s primitive set with memory primitives inspired by Teller’s "Learning Mental Models".
- **Evaluation**: Parallelized using multiprocessing, with memory isolation per process.
