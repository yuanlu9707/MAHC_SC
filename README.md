# MAHC_SC
MAHC_SC for "MAHC-SC: Micro-Macro Mobility Alignment and Dynamic Group-based Hypergraph Collaboration for Human-UAV Spatial Crowdsourcing"
## :page_facing_up: Description
MAHC_SC包含M3A和DGHC
## :wrench: Dependencies
- Python == 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch == 1.8.1](https://pytorch.org/)
- NVIDIA GPU (RTX 8000) + [CUDA 11.7](https://developer.nvidia.com/cuda-downloads)
### Installation
1. Clone repo
    ```bash
    git clone https://github.com/yuanlu9707/MAHC_SC.git
    cd MAHC_SC
    ```
2. Install dependent packages
    ```
    pip install -r requirements.txt
    ```
## :zap: Quick Inference

Get the usage information of the project
```bash
cd code
python main.py -h
```
Then the usage information will be shown as following
```
usage: main.py [-h] env_name method_name mode

positional arguments:
  env_name     the name of environment (KAIST or purdue)
  method_name  the name of method (MAPPO_COMM)
  mode         train or test

optional arguments:
  -h, --help   show this help message and exit
```
python main.py KAIST MAPPO test
python main.py purdue MAPPO test
```

## :computer: Training

We provide complete training codes for MAHC_SC.<br>
You could adapt it to your own needs.

1. You can modify the config files 
[human_drone_SC/code/environment/KAIST/conf.py](https://github.com/BIT-MCS/human_drone_SC/tree/main/code/environment/KAIST/conf.py) and
[human_drone_SC/code/environment/purdue/conf.py](https://github.com/BIT-MCS/human_drone_SC/tree/main/code/environment/NCSU/conf.py) for environments.<br>
For example, you can control the number of drones in the environment by modifying this line
	```
	[43]  'uav_num': 6,
	```
2. You can modify the config file 
[human_drone_SC/code/method/MAPPO/conf.py](https://github.com/BIT-MCS/human_drone_SC/tree/main/code/method/fd_mappo_cubicmap/conf.py) for method.<br>

3. Training
	```
	python main.py KAIST MAPPO train
	python main.py NCSU MAPPO train
	```
	The log files will be stored in [human_drone_SC/log](https://github.com/BIT-MCS/human_drone_SC/tree/main/log).

## :checkered_flag: Testing
	```
	python main.py KAIST MAPPO test
	python main.py NCSU MAPPO test
	```
## :scroll: Acknowledgement

This work is supported by Chinese national key research plan 2021YFF0901205. 
<br>
Corresponding author: Lei Yang.

## :e-mail: Contact

If you have any question, please email `yuanlu9707@163.com`.

## Paper
If you are interested in our work, please cite our paper as
```
MAHC-SC: Micro-Macro Mobility Alignment and Dynamic Group-based Hypergraph Collaboration for Human-UAV Spatial Crowdsourcing
```
