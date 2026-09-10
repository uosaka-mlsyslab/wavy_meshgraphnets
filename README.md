# Wave-Dynamics-Augmented MeshGraphNets

This repository contains the implementation of the Wave and Mix updates
presented in:

**Wave-Dynamics-Augmented Graph Neural Networks for Mesh-Based Physical Simulation**  
Masashi Mitani, Satoshi Noguchi, Takuya Konishi, and Yoshinobu Kawahara  
ICONIP 2026

The implementation is based on DeepMind's MeshGraphNets codebase.
It supports the CylinderFlow and FlagSimple datasets.

## Setup

Install Miniconda if `conda` is not available:

```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

If running as root in a container, use `source /root/.bashrc` instead.

Create the environment and install the required packages:

```bash
conda create --name tf1 -c conda-forge tensorflow-gpu=1.15 python=3.7
conda activate tf1
pip install -r requirements.txt
```

## Data

Datasets are not included in this repository.
Please download the CylinderFlow and FlagSimple datasets following the
instructions in the official MeshGraphNets repository:

https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets

Set `${DATA}` to the directory containing the downloaded datasets.

Each dataset directory should contain `meta.json`, `train.tfrecord`,
`valid.tfrecord`, and `test.tfrecord`.

## Train

For example, the Mix update can be trained on FlagSimple as follows:

```bash
python -m wavy_meshgraphnets.run_model \
  --mode=train \
  --model=cloth \
  --dataset_dir=${DATA}/flag_simple \
  --checkpoint_dir=${DATA}/chk_flag \
  --num_training_steps=100000 \
  --update_rule=mix \
  --seed=42
```

The command above uses 100,000 training steps as a lightweight example.
The experiments reported in the paper use 10,000,000 training steps.

Use `--model=cfd` with `${DATA}/cylinder_flow` for CylinderFlow.

The available update rules are:

- `--update_rule=mix`: Mix update in the paper
- `--update_rule=wave`: Wave update in the paper

## Evaluate

For example, a trained Mix model on FlagSimple can be evaluated as follows:

```bash
python -m wavy_meshgraphnets.run_model \
  --mode=eval \
  --model=cloth \
  --dataset_dir=${DATA}/flag_simple \
  --checkpoint_dir=${DATA}/chk_flag \
  --rollout_split=valid \
  --rollout_path=${DATA}/rollout_flag_valid.pkl \
  --num_rollouts=10 \
  --update_rule=mix \
  --seed=42
```

Evaluation saves rollout trajectories to `--rollout_path` and logs MSE values.

## Attribution

This implementation is based on MeshGraphNets:

```bibtex
@inproceedings{pfaff2021learning,
  title={Learning Mesh-Based Simulation with Graph Networks},
  author={Tobias Pfaff and Meire Fortunato and Alvaro Sanchez-Gonzalez and Peter W. Battaglia},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

## Citation

If you use this code, please cite:

**Wave-Dynamics-Augmented Graph Neural Networks for Mesh-Based Physical Simulation**  
Masashi Mitani, Satoshi Noguchi, Takuya Konishi, and Yoshinobu Kawahara  
33rd International Conference on Neural Information Processing (ICONIP 2026).
