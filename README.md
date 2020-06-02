# PyTPT

Implementation of Transition Path Theory for:
- stationary Markov chains (pytpt/stationary.py),
- for periodically varying Markov chains (pytpt/periodic.py),
- for time-inhomogenous Markov chains over finite time intervals (pytpt/finite.py).

Based on: 
Helfmann, L., Ribera Borrell, E., Schütte, C., & Koltai, P. (2020). Extending Transition Path Theory: Periodically-Driven and Finite-Time Dynamics. [arXiv preprint arXiv:2002.07474.](https://arxiv.org/pdf/2002.07474.pdf)  

## PyTPT Package Installation
1. Clone the project in a local repository: 
`
git clone https://github.com/LuzieH/pytpt.git
`
2. Add the package to your local python library:
` 
pip install -e .
` 
 
## Quick Start (run examples)
1. Clone the project in a local repository
`
git clone https://github.com/LuzieH/pytpt.git
`
\
and install pytpt: 
`
pip install -e .
`
2. Install project requirements:
` 
pip install -r requirements
` 
3. Run small network example
```
python examples/small_network_construction.py
python examples/small_network_example.py
python examples/small_network_plotting.py
``` 
4. Run triplewell example
```
python examples/triplewell_construction.py
python examples/triplewell_example.py
python examples/triplewell_plotting.py
``` 
