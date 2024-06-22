README.txt

Title: Optimization Algorithms for Sparse Peaks, Modular Graph Coloring, and Neural Network Weight Optimization


---------------------------------------------------------------------
Overview
---------------------------------------------------------------------
This project evaluates the performance of Randomized Hill Climbing (RHC), Simulated Annealing (SA), and Genetic Algorithms (GA) on the Sparse Peaks and Modular Graph Coloring optimization problems. Additionally, these algorithms are applied to optimize weights for a neural network used in predicting medication adherence.

---------------------------------------------------------------------
Requirements
---------------------------------------------------------------------
Ensure that you have the following installed on your system:


For LaTeX:
- LaTeX distribution (e.g., TeX Live, MiKTeX)
- An editor to write and compile LaTeX (e.g., TeXShop, Overleaf, TeXworks, Visual Studio Code with LaTeX Workshop extension)


The following Python libraries are required to run the code:
- Python 3.6 or higher
- numpy
- pandas
- scikit-learn
- matplotlib
- torch (PyTorch)
- mlrose
- mlrose_hiive


---------------------------------------------------------------------
File Included 
---------------------------------------------------------------------
The project consists of the following files in GaTech_Box(https://gatech.box.com/s/0n7zssv7ic6mivzce7eeujsokccxg0mr):
1. algorithm.py - Contains the implementation of RHC, SA, and GA for Sparse Peaks and Modular Graph Coloring problems.
2. adhe.py - Contains the implementation of the neural network for medication adherence prediction and the optimization algorithms.
3. mlrose1.py - A script for running and analyzing the optimization algorithms.
4. README.txt - This file containing instructions for running the code.

---------------------------------------------------------------------
Instructions for Running the Code
---------------------------------------------------------------------

To run the Sparse Peaks and Modular Graph Coloring optimization: python algorithm.py

To run the neural network optimization for medication adherence prediction: python adhe.py

To run and analyze fucusing on optimization algorithms for Neural Network: python mlrose1.py


The `algorithm.py` script will:
1. Define the Sparse Peaks and Modular Graph Coloring problems.
2. Implement the RHC, SA, and GA optimization algorithms.
3. Configure and tune the hyperparameters for each algorithm.
4. Run the optimization algorithms on the problems.
5. Print the best solutions and their fitness scores to the terminal.
6. Generate plots to visualize the fitness over iterations and other relevant metrics using Matplotlib.

The `adhe.py` script will:
1. Load and preprocess the dataset.
2. Define the neural network architecture using PyTorch.
3. Apply RHC, SA, and GA to optimize the neural network weights.
4. Generate and display plots of the learning curves and fitness scores using Matplotlib.

The `mlrose1.py` script will:
1. Load and preprocess the dataset.
2. Define the neural network and fitness function.
3. Set up and run the optimization algorithms (RHC, SA, and GA) using the `mlrose` and `mlrose_hiive` libraries.
4. Record the best fitness scores, fitness curves, and computation times.
5. Plot fitness over iterations, best fitness comparison, and wall clock time using Matplotlib.

---------------------------------------------------------------------
Results
---------------------------------------------------------------------
The results of the optimization algorithms will be displayed in the terminal/command prompt and visualized using Matplotlib. The plots include:
- Fitness scores of RHC, SA, and GA over iterations.
- Best fitness scores achieved by each algorithm.
- Wall clock time taken by each algorithm for optimization.

---------------------------------------------------------------------
Contact
---------------------------------------------------------------------
For further assistance, please contact:
- Name: David Hur
- davidhur@gatech.edu








