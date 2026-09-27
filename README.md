# Evolved Virtual Creatures: Mountain Climbing GA

A multi-threaded Genetic Algorithm (GA) implementation inspired by Karl Sims' seminal work on "Evolved Virtual Creatures." This project simulates the evolution of morphologically diverse creatures designed to navigate and climb complex, Gaussian-generated mountain terrains using the PyBullet physics engine.

## 🚀 Overview

The system evolves both the physical structure (morphology) and the control systems (neural/oscillator-based) of virtual creatures. Each creature starts at the base of a mountain and is tasked with climbing as high as possible. Over successive generations, the algorithm selects for traits that favor stability, mobility, and vertical ascent.

## 🛠️ Key Features

- **Dynamic Morphology**: Creatures are built from a hierarchical set of links (boxes, cylinders, spheres) defined by their genome.
- **Genetic Algorithm Core**:
    - **Population Management**: Handles populations of virtual agents with full generational cycles.
    - **Elitism**: Preserves the top performers (Top-2) to the next generation without mutation.
    - **Diversity Injection**: Automatically introduces random genomes every generation to prevent premature convergence.
    - **Adaptive Mutation**: Real-time adjustment of mutation rates (Point, Grow, Shrink) based on fitness stagnation.
- **Multi-Threaded Execution**: Leverages Python's `multiprocessing` to parallelize physics simulations across all available CPU cores, significantly speeding up the evolutionary process.
- **Sophisticated Fitness Function**: A multi-objective fitness evaluator that rewards:
    - Vertical height gain.
    - Proximity and contact with the mountain.
    - Upright orientation.
    - Directional alignment toward the peak.
    - Penalizes excessive mass, complexity, and "cheating" (height without contact).
- **Environment**: Procedural generation of mountain terrain using Gaussian distributions, creating varied challenges for the evolving creatures.

## 🧬 Genetic Operations

The project implements a comprehensive suite of genetic operators:

1.  **Crossover**: Single-point crossover that blends link structures between two parents.
2.  **Point Mutation**: Randomly tweaks physical and control parameters (joint friction, motor amplitude, etc.).
3.  **Grow Mutation**: Adds new links to the creature's body, increasing its complexity.
4.  **Shrink Mutation**: Removes links, simplifying the morphology.
5.  **Tournament Selection**: Selects parents through competitive bouts (k=5) to maintain high selection pressure.

## 📂 Project Structure

- `src/multi_ga.py`: The main entry point for the training process. Runs the multi-threaded GA.
- `src/genome.py`: Defines the genetic representation, URDF generation, and mutation logic.
- `src/creature.py`: Manages creature state, motor control, and the multi-objective fitness function.
- `src/simulation.py`: Interfaces with PyBullet to handle the physics environment and multi-threaded simulation pools.
- `src/environment.py`: Procedural terrain generation (Mountains, Arenas, Obstacles).
- `src/cw-envt.py`: A utility script to visualize and test evolved elite creatures in a GUI environment.

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- PyBullet
- NumPy

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pybullet numpy
   ```

## 🏃 Usage

### 1. Training (Evolution)
To start the evolutionary process, run the multi-threaded GA:
```bash
python src/multi_ga.py
```
This will run in `DIRECT` mode (no GUI) for maximum performance. Fitness logs and elite creature genomes will be saved as `.csv` files.

### 2. Visualization
To watch an evolved creature (e.g., from `elite_cr3.csv`) in the GUI simulation:
```bash
python src/cw-envt.py
```

## 📊 Fitness Function Details
The fitness function is the heart of the evolution. It is calculated as:
`Fitness = (Contact + Proximity + Exploration + Uprightness + Directional + Climbing) - Penalties`

- **Climbing Progress**: Up to 40 points for height gained while touching the mountain.
- **Anti-Cheat**: Heavy penalties for gaining height without being in contact with the mountain (preventing "floating" bugs).
- **Complexity Limit**: Discourages creatures with more than 11 links to maintain simulation stability.

## 🎓 Acknowledgments
- Inspired by **Karl Sims'** "Evolved Virtual Creatures" (1994).
- Built using the **PyBullet** physics engine.

---