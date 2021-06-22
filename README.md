# Evading Adversarial Example Detection Defenseswith Orthogonal Projected Gradient Descent

This repository contains the code used in the paper [include link]

## Structure
The main attack class can be found in `attack.py`. We then give a few examples of the case studies we talk about in the paper. 

### Trapdoor/Honeypots
run
```python
python3 trapdoor.py
```
to attack the classifier and detector on cifar10. The script runs an Orthogonal Projected Gradient Descent attack for 100 steps at epsilon 8/255 and prints the results (accuracy and ROC curve).

### Dense Layer Analysis
run
```python
python3 dla.py
```
to attack the classifier and detector on cifar10. The script runs an Orthogonal Projected Gradient Descent attack for 100 steps at epsilon 8/255 and prints the results. The model and detector can be found in `defense.py`.

