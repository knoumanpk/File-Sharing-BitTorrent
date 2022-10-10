# Rarest-First with Probabilistic Mode-Suppression
Official implementation of [Rarest-First with Probabilistic Mode-Suppression]

## Prerequisites
- Python 3.7

A requirements.txt is provided for use with pip or Anaconda.

## File description
- `file_sharing.py`: Main file for running simulations.
- `Network.py`: Class implementations of Network, Peer, and Seed.
- `Logger.py`: Class implementation of Logger object that records simulation variables.
- `utils.py`: Functions necessary for modeling interactions of peers and the seed.
- `log_utils.py`: Functions for logging simulation results.

## Running an experiment
For generating results presented in [Rarest-First with Probabilistic Mode-Suppression], you could run:

`python file_sharing.py`

In `file_sharing.py`, there are functions: stability_check(), scalability_check(), sojourn_time_performance(), single_swarm_sojourn_times(), flash_crowd_response(), and one_club_escape(). Within each function, you may set up multiple configurations of the model and get their results in one run.

The results of each simulation run are stored in the directory `./../results/` with its own timestamp.
