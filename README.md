# EquiMod: An Equivariance Module to Improve Visual Instance Discrimination
⚠️ **Important: This branch does not contain any code. Please switch to either the SimCLR or BYOL branch for code access.** ⚠️

## Overview
This is the official repository of EquiMod, which introduces a generic equivariance module specifically designed to improve visual instance discrimination in self-supervised learning models. EquiMod strategically addresses the balance between invariance and sensitivity to image augmentations, a crucial aspect in representation learning. By predicting the displacement in embedding spaces caused by augmentations, EquiMod enriches the representations. This allows models like BYOL and SimCLR, when integrated with EquiMod, to not only retain essential augmentation-related information but also to enhance classification performances significantly on datasets such as CIFAR10 and ImageNet. For a deeper dive into our methodology and results, we invite you to consult our [published article](https://arxiv.org/abs/2211.01244).

## Branches
This repository is organized into two primary branches, each representing a distinct application of EquiMod to different self-supervised learning frameworks: SimCLR and BYOL.

Each branch provides a unique perspective on the adaptability and utility of EquiMod in improving self-supervised learning models.

## Results
We conducted linear evaluations on representations learned from ImageNet and CIFAR10. These evaluations demonstrate EquiMod's effectiveness as EquiMod consistently improved performance across all baselines on both datasets, with the exception of BYOL trained over 1000 epochs. However, it enhanced BYOL's performance in shorter training cycles (100 and 300 epochs, see the article for more details).

Below is a table showing the results of our linear evaluation:

| Method                    | ImageNet Top-1 | ImageNet Top-5 | CIFAR10 Top-1 | CIFAR10 Top-5 |
|---------------------------|----------------|----------------|---------------|---------------|
| SimCLR*                   | 71.57          | 90.48          | 90.96         | 99.73         |
| SimCLR* + EquiMod         | 72.30          | 90.84          | 92.79         | 99.78         |
| BYOL*                     | 74.03          | 91.51          | 90.44         | 99.62         |
| BYOL* + EquiMod           | 73.22          | 91.26          | 91.57         | 99.71         |

Table 1: Linear Evaluation; top-1 and top-5 accuracies (in %) under linear evaluation on ImageNet and CIFAR10 (* denote our re-implementations)

## Usage and Installation
This implementation of EquiMod has been specifically developed for use on the Jean Zay supercomputer, leveraging its advanced computational resources. To effectively deploy and utilize EquiMod:

1. **System Requirements**
   - Designed primarily for execution on the Jean Zay supercomputer.
   - Requires CUDA 11.2 and PyTorch 1.10.0.

2. **Setting Up on Jean Zay**
   - Utilize the SLURM job scheduler for managing and executing tasks.
   - Load all necessary modules and dependencies as per Jean Zay's configuration.
   - Configure the provided `.slurm` script according to your project's specific needs, including resource allocation and job parameters.

3. **Adapting for Other Systems**
   - While the code is tailored for Jean Zay, it may require adaptations for use in different computational environments.
   - Users looking to deploy EquiMod on other systems should modify the setup and execution environment to align with their system's specifications.
   - The `.slurm` files and scripts included in the repository serve as a guide for adapting the setup for other high-performance computing systems.

## Citation
If you use EquiMod in your research, please cite our work:
```bibtex
@article{devillers2022equimod,
  title={Equimod: An equivariance module to improve self-supervised learning},
  author={Devillers, Alexandre and Lefort, Mathieu},
  journal={arXiv preprint arXiv:2211.01244},
  year={2022}
}
```

