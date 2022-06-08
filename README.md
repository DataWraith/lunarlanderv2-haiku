# Solving LunarLander-v2

[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

This repository contains my solution for the [LunarLander-v2](https://www.gymlibrary.ml/environments/box2d/lunar_lander/) OpenAI gym environment.
The code is written in Python using libraries from the 
[DeepMind JAX ecosystem](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research).

The code solves the LunarLander-v2 environment in approximately 40&thinsp;000 frames
(about 120 episodes).
I count the environment as solved once the average reward
(measured over the preceeding 100 episodes)
exceeds 200.

## References

* Chen, Xinyue, Che Wang, Zijian Zhou, and Keith Ross. 2021. "Randomized
Ensembled Double Q-Learning: Learning Fast Without a Model." arXiv.
<https://doi.org/10.48550/ARXIV.2101.05982>.

* Ferret, Johan, Olivier Pietquin, and Matthieu Geist. 2020.
"Self-Imitation Advantage Learning." arXiv.
<https://doi.org/10.48550/ARXIV.2012.11989>.

* Lee, Kimin, Michael Laskin, Aravind Srinivas, and Pieter Abbeel. 2020.
"SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep
Reinforcement Learning." arXiv.
<https://doi.org/10.48550/ARXIV.2007.04938>.

* Sinha, Samarth, Homanga Bharadhwaj, Aravind Srinivas, and Animesh Garg. 2020. 
"D2rl: Deep Dense Architectures in Reinforcement Learning." arXiv.
<https://doi.org/10.48550/ARXIV.2010.09163>.
