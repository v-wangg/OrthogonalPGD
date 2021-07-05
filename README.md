# Evading Adversarial Example Detection Defenseswith Orthogonal Projected Gradient Descent

This repository contains the code used in the [paper](https://arxiv.org/abs/2106.15023)

The main attack class can be found in `attack.py`. We then give a few examples of the case studies we talk about in the paper.

We include defense scripts for the trapdoor and DLA defenses because we re-implemented these in pytorch ourself, and so we need no author permission to distribute these files. We do not include pre-trained models because the original defense authors trained models for us---we will obtain their permission to release those pretrained models in a full code release. Similarly we will release other defense with author permission on code reuse.

The script "run_experiment.py" contains the necessary boilerplate to run an attack given pretrained model weights.
