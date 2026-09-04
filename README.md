# MAHC-SC

Official implementation of **MAHC-SC: Micro-Macro Mobility Alignment and Dynamic Group-based Hypergraph Collaboration for Human-UAV Spatial Crowdsourcing**.

## Description

MAHC-SC contains two main components:

- **M3A (Micro-Macro Mobility Alignment):** models mobility at the individual and regional levels and aligns the two representations through bidirectional individual-to-region and region-to-individual objectives.
- **DGHC (Dynamic Group-based Hypergraph Collaboration):** periodically constructs UAV collaboration groups and performs group-level information exchange through hypergraph communication.

The repository provides environments for the **KAIST** and **Purdue** datasets and implements MAHC-SC on top of MAPPO.

## Requirements

- Python 3.8
- PyTorch 1.13.0
- NumPy 1.19.5
- Numba 0.53.1
- Matplotlib 3.3.4
- scipy==1.6.2
- scikit-learn==0.24.1
- An NVIDIA GPU and a CUDA installation compatible with the selected PyTorch build

Using [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is recommended.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yuanlu9707/MAHC_SC.git
   cd MAHC_SC
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```text
MAHC_SC/
|-- environment/
|   |-- KAIST/              # KAIST environment and configuration
|   `-- purdue/             # Purdue environment and configuration
|-- method/
|   `-- MAPPO/
|       |-- conf.py         # Method and MAHC-SC hyperparameters
|       |-- pred.py         # M3A mobility prediction and representation module
|       |-- R2I.py          # Individual-region alignment objectives
|       |-- model.py        # Policy, hypergraph, and DGHC modules
|       |-- train.py        # Training procedure
|       `-- test.py         # Evaluation procedure
|-- main.py                 # Entry point
`-- util.py
```

## Configuration

The current entry point reads the experiment settings directly from the `args` object in [`main.py`](https://github.com/yuanlu9707/MAHC_SC/blob/main/main.py):

```python
args = type('Args', (object,), {
    'env_name': 'KAIST',
    'method_name': 'MAPPO_COMM',
    'mode': 'train',
})()
```

Available settings are:

- `env_name`: `KAIST` or `purdue`
- `method_name`: `MAPPO_COMM` for the complete MAHC-SC model, or `MAPPO` for MAPPO with M3A but without DGHC communication
- `mode`: `train` or `test`

Environment parameters, including the number of UAVs and dataset paths, can be changed in:

- [`environment/KAIST/conf.py`](https://github.com/yuanlu9707/MAHC_SC/blob/main/environment/KAIST/conf.py)
- [`environment/purdue/conf.py`](https://github.com/yuanlu9707/MAHC_SC/blob/main/environment/purdue/conf.py)

Method, M3A, and DGHC hyperparameters can be changed in:

- [`method/MAPPO/conf.py`](https://github.com/yuanlu9707/MAHC_SC/blob/main/method/MAPPO/conf.py)

For example, the number of UAVs is controlled by:

```python
'uav_num': 6,
```

## Training

Set `mode` to `train` in `main.py`, select the environment, and use `MAPPO_COMM` to train the complete MAHC-SC model.

### KAIST

```python
'env_name': 'KAIST',
'method_name': 'MAPPO_COMM',
'mode': 'train',
```

### Purdue

```python
'env_name': 'purdue',
'method_name': 'MAPPO_COMM',
'mode': 'train',
```

Start training from the repository root:

```bash
python main.py
```

The output directory is controlled by `root_path` in `method/MAPPO/conf.py`.

## Testing

Set `mode` to `test` in `main.py` and keep `env_name` and `method_name` consistent with the trained checkpoint:

```python
args = type('Args', (object,), {
    'env_name': 'KAIST',
    'method_name': 'MAPPO_COMM',
    'mode': 'test',
})()
```

Then run:

```bash
python main.py
```

## Acknowledgement

This work is supported by the National Key Research and Development Program of China (2021YFF0901205).

Corresponding author: Lei Yang.

## Contact

For questions, please contact `yuanlu9707@163.com`.

## Citation

If you find this repository useful, please cite:

```text
MAHC-SC: Micro-Macro Mobility Alignment and Dynamic Group-based Hypergraph Collaboration for Human-UAV Spatial Crowdsourcing
```
