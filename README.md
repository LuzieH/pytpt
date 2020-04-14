# transitions

Implementation of Transition Path Theory for:
- stationary Markov chains (transition_paths.py),
- for periodically varying Markov chains (transition_paths_periodic.py),
- for time-inhomogenous Markov chains over finite time intervals (transition_paths_finite.py).

based on: 
Helfmann, L., Ribera Borrell, E., Schütte, C., & Koltai, P. (2020). Extending Transition Path Theory: Periodically-Driven and Finite-Time Dynamics. arXiv preprint arXiv:2002.07474.

## Quick Start
1. Clone the project in a local repository:
```
git clone https://github.com/LuzieH/transitions.git
```
2. Create Virtual Environment (Optional):
```
python -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
```
3. Install requirements with the Package Installer for Python3:
```
pip install -r requirements
```
4. Create folders to store npy files and plots: 
```
mkdir examples/data
mkdir examples/charts
```
Rem: The above mentioned folders are ignored. 
