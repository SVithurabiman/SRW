# [ICLR 2026] THE SELF-RE-WATERMARKING TRAP: FROM EXPLOIT TO RESILIENCE

Official implementation of [THE SELF-RE-WATERMARKING TRAP: FROM EXPLOIT TO RESILIENCE](https://openreview.net/pdf?id=st1hrLTP14).

---

## Table of Contents

- [Requirements](#requirements)
- [Setup & Run](#setup--run)
- [Usage](#usage)
- [Configuration](#configuration)
- [Citation](#citation)
<!-- - [Contributing](#contributing) -->
<!-- - [License](#license) -->

---

## Requirements

- Python 3.8+  
- Conda (recommended)  
- CUDA 11.8 if using GPU  

All dependencies are captured in `environment.yml` for reproducibility.

---

## Setup & Run

We provide a bash script that handles environment creation, checkpoint download, and running training/testing.

1. **Clone the repository:**

```bash
git clone --recurse-submodules https://github.com/SVithurabiman/SRW.git
```

2. **Run the setup script:**

```bash
bash run_all.sh [train|test]
```
This script will:

- Create a Conda environment from environment.yml

- Activate the environment

- Download pretrained checkpoints from HuggingFace into `results/`

- Run training if `train` is passed as an argument. 
  After running, outputs and logs will be saved in ```results/.```

- Run testing/evaluation if `test` is passed as an argument


## **Usage**

If you prefer to run manually:

**Training**
```
    
python -m SRW.train --dataset_config ./SRW/configs/dataset.yaml --exp_config ./SRW/configs/train.yaml

```
**Visualizing Training**

This repository uses TensorBoard to log training progress.

Launch TensorBoard:
```bash
tensorboard --logdir results --port 6006
```

Then open in your browser:

```
http://localhost:6006
```

**Evaluation / Testing**

```
python -m SRW.test --dataset_config ./SRW/configs/dataset.yaml --exp_config ./SRW/configs/test.yaml
```

## **Configuration**

All experiment settings are in the configs/ folder:

- configs/train.yaml – default training settings

- configs/test.yaml – default test settings

- configs/dataset.ymal - default Image and Message settings

Modify paths, hyperparameters, or model options in these YAML files as needed.

## **Citation**
If you find this work useuful, please cite: 

```bibtext
@inproceedings{senthuran2026selfrewatermarking,
  title={The Self‑Re‑Watermarking Trap: From Exploit to Resilience},
  author={Vithurabiman Senthuran and Yong Xiang and Iynkaran Natgunanathan and Uthayasanker       
          Thayasivam},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=st1hrLTP14}
}
```