# Learning Morphology and Control for a Microrobot
This is the repository for the paper "Learning Morphology and Control for a 
Microrobot". More information can be found on our website 
[here](https://sites.google.com/view/learning-robot-morphology/).
Included are demos for running the experiments laid out in the paper.

## Installation
Most of the pip dependencies can be installed with `pip install -r
requirements.txt`. 
First activate the venv with `source venv/bin/activate`, then install GPy with
```
git clone https://github.com/SheffieldML/GPy.git
pip install -e GPy
```
Due to the complexity of managing different python versions and dependencies
between different modules, we recommend running this repository on the
associated [docker image](https://github.com/tholiao/learn_robot_docker/).

## Running Experiments 
Experiments can be run either locally or via Docker. 

### Running locally
First, start V-REP running locally. Then use `main.py` to call the relevant
experiment, for example:
```
python main.py hpcbbo --init_uc=5 --init_cn=5 --uc_runs_per_cn=5 \
                      --batch_size=5 --total=5 --obj_f=1 --contextual
```
Or
```
python main.py random --init_uc=5 --total=5 --obj_f=1
```
Call `python main.py -h` for additional help


### Running on Docker
Instructions for setting up Docker are included in the [related
repository](https://github.com/tholiao/learn_robot_docker).

## Simulator Setup
Before running any of the experiments, make sure V-REP is open (see the V-REP
documentation for troubleshooting issues with installation/booting). Scenes are
automatically loaded and can be found in `scenes/`. The default simulator
settings should work fine, but check that the following settings are correct:
* Physics engine: Bullet 2.78
* Time step: 50 ms

## Citation

Should you find this code useful, please support us by citing our paper:
```
T. Liao, G. Wang, B. Yang, R. Lee, S. Levine, K. Pister, R. Calandra. 
Data-efficient Learning of Morphology and Controller for a Microrobot.
In IEEE Int. Conf on Robotics and Automatation, ICRA '19, Montreal, Canada, May
2019. 
```
