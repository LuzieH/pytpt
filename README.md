# pytpt

Implementation of Transition Path Theory for:
- stationary Markov chains (stationary.py),
- for periodically varying Markov chains (periodic.py),
- for time-inhomogenous Markov chains over finite time intervals (finite.py).

based on: 
Helfmann, L., Ribera Borrell, E., Schütte, C., & Koltai, P. (2020). Extending Transition Path Theory: Periodically-Driven and Finite-Time Dynamics. arXiv preprint arXiv:2002.07474.

## pytpt Package Installation
1. Clone the project in a local repository:
```
git clone https://github.com/LuzieH/pytpt.git
```
2. Add the package to your local python library:
```
pip install -e pytpt
```
## Quick Start (run examples)
1. Clone the project in a local repository:
```
git clone https://github.com/LuzieH/pytpt.git
```
2. Install pytpt package
```
pip install -e pytpt
```
3. Install project requirements:
```
pip install -r requirements
```
4. Run small network example
```
python examples/small_network_construction.py
python examples/small_network_example.py
python examples/small_network_plotting.py
``` 
5. Run triplewell example
```
python examples/triplewell_construction.py
python examples/triplewell_example.py
python examples/triplewell_plotting.py
``` 
