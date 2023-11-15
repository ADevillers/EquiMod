#!/bin/bash
#SBATCH -A xwh@v100             # Change to match the right project
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-32g
#SBATCH --job-name=expe
#SBATCH --time=20:00:00
#SBATCH --nodes=1               # TODO: Define the number of nodes
#SBATCH --ntasks-per-node=2     # = --gres=gpu:
#SBATCH --gres=gpu:2            # = --ntasks-per-node
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --output=./logs/expe_%j.out
#SBATCH --error=./logs/expe_%j.err

# Move to the path where the script is submitted
cd ${SLURM_SUBMIT_DIR}

# Clean modules
module purge
module unload cuda

# Load used modules
module load cuda/11.2
module load pytorch-gpu/py3/1.10.0

# Echo commands
set -x

# Execute python code
srun python src/main.py \
    --computer=jeanzay \
    --hardware=multi-gpu \
    --precision=mixed \
    --nb_workers=10 \
    --expe_name=equimod_simclr_cifar10 \
    --dataset_name=cifar10 \
    --resnet_type=resnet18 \
    --lmbd=1.0 \
    --nb_epochs=800 \
    --nb_epochs_warmup=10 \
    --batch_size=512 \
    --lr_init=4.0 \
    --momentum=0.9 \
    --weight_decay=1e-6 \
    --eta=1e-3 \
    --z_dim=128 \
    --y_dim=128 \
    --temperature_z=0.5 \
    --temperature_y=0.2 \
    --clsf_every=100 \
    --save_every=100 \
    --nb_epochs_clsf=90 \
    --batch_size_clsf=256 \
    --lr_init_clsf=0.2 \
    --momentum_clsf=0.9 \
    --weight_decay_clsf=0.0 \
    --proj_head_eq_layers="2048-2048-" \
    --proj_head_t_layers="128" \
    --predictor_eq_layers="one" \
#    --checkpoint=./checkpoints/expe_729631_even.pt \
