# Euler Cluster Guide — ACT Training via Isaac Lab Docker Interface

This guide explains how to run ACT training jobs on the **ETH Zurich Euler cluster** using the Isaac Lab Docker/Singularity pipeline. The `act_copy/` folder added to the Isaac Lab repo is used as the training entry point, submitted as a SLURM job through the standard `cluster_interface.sh` script.

---

## Prerequisites

- Access to the Euler cluster (`username@euler.ethz.ch`)
- Docker installed locally with the Isaac Lab image already built
- Apptainer installed locally (for Singularity image export)
- SSH key set up for passwordless login to Euler (recommended)

For building the Docker image locally, follow the official Isaac Lab guide:
[https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html](https://isaac-sim.github.io/IsaacLab/main/source/deployment/index.html)

---

## Step 1 — First-time cluster setup

SSH into Euler and create the required persistent directories:

```bash
ssh username@euler.ethz.ch
mkdir -p ~/isaaclab/data_storage
```

The `data_storage/` directory is mounted into the container at `/workspace/isaaclab/data_storage` and persists between jobs — checkpoints and datasets should be written here.

---

## Step 2 — Configure `.env.cluster`

Edit `docker/cluster/.env.cluster` in your local Isaac Lab repo. The config used for this project:

```bash
# Job scheduler
CLUSTER_JOB_SCHEDULER=SLURM

# Isaac Sim cache directory on the cluster (must end in docker-isaac-sim)
CLUSTER_ISAAC_SIM_CACHE_DIR=/cluster/project/meboldt/student_Lucas_Michael/docker-isaac-sim

# Isaac Lab directory on the cluster (must end in isaaclab)
CLUSTER_ISAACLAB_DIR=/cluster/project/.../isaaclab

# Cluster login
CLUSTER_LOGIN=username@euler.ethz.ch

# Directory on the cluster where the .sif Singularity image is stored
CLUSTER_SIF_PATH=/cluster/project/.../scratch

# Whether to delete the temporary code copy after the job finishes
REMOVE_CODE_COPY_AFTER_JOB=false

# Python script that is executed inside the container for each submitted job
CLUSTER_PYTHON_EXECUTABLE=act_copy/imitate_episodes.py
```

> `CLUSTER_PYTHON_EXECUTABLE` points to `act_copy/imitate_episodes.py` for ACT training.
> Other available entry points used in this project:
> ```
> scripts/imitation_learning/robomimic/play_cvae.py          # robomimic policy rollout
> scripts/imitation_learning/isaaclab_mimic/generate_dataset.py  # dataset generation
> ```

---

## Step 3 — Add `data_storage` bind mount to `run_singularity.sh`

The cluster launch script `docker/cluster/run_singularity.sh` must bind-mount `data_storage/` from the persistent cluster directory into the container. Add the two highlighted lines to the `singularity exec` call:

```bash
singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/workspace/isaaclab/logs:rw \
    -B $CLUSTER_ISAACLAB_DIR/data_storage:/workspace/isaaclab/data_storage:rw \   # <-- add this
    --nv --writable --containall $TMPDIR/$2.sif \
```

The `logs/` bind ensures training logs are written back to the persistent cluster directory.
The `data_storage/` bind ensures checkpoints and datasets survive after the job ends.

---

## Step 4 — Build the Docker image locally and push the Singularity image

Build the Isaac Lab Docker image locally (only needed once, or after image changes):

```bash
./docker/container.py start
./docker/container.py stop
```

Export to Singularity and upload to the cluster (also only needed once per image version):

```bash
./docker/cluster/cluster_interface.sh push
```

This converts the Docker image to a `.sif` file under `docker/exports/` and copies it to `CLUSTER_SIF_PATH` on Euler. This step can take a while.

---

## Step 5 — Upload the dataset to `data_storage/`

The dataset HDF5 file must be present on the cluster inside the persistent `data_storage/` directory before a training job is submitted. Copy it from your local machine using `scp` or `rsync`:

```bash
# scp — simple single file transfer
scp /path/to/local/dataset.hdf5 \
    username@euler.ethz.ch:/cluster/project/meboldt/student_Lucas_Michael/isaaclab/data_storage/datasets/dataset.hdf5

# rsync — preferred for large files, resumes on interruption
rsync -avP /path/to/local/dataset.hdf5 \
    username@euler.ethz.ch:/cluster/project/meboldt/student_Lucas_Michael/isaaclab/data_storage/datasets/dataset.hdf5
```

The destination path on the cluster is `$CLUSTER_ISAACLAB_DIR/data_storage/`, which is bind-mounted into the container at `/workspace/isaaclab/data_storage/`. Any subdirectory structure you create under `data_storage/` (e.g. `datasets/`) is preserved inside the container.

> **Important — update `DATA_DIR` in `act_copy/constants.py` before submitting.**
>
> `constants.py` defines the path to the dataset file that `imitate_episodes.py` reads via the task config. The path must be **relative to `/workspace/isaaclab/`** (the Isaac Lab root inside the container), because that is the working directory when the job runs.
>
> ```python
> # act_copy/constants.py
>
> # Path is relative to /workspace/isaaclab/ inside the container.
> # /workspace/isaaclab/data_storage/ is bind-mounted from
> # $CLUSTER_ISAACLAB_DIR/data_storage/ on the cluster.
>
> DATA_DIR = 'data_storage/datasets/mimic_800_simplfied_task.hdf5'
> ```
>
> Make sure this path matches exactly where you uploaded the file. For example, if you uploaded to
> `$CLUSTER_ISAACLAB_DIR/data_storage/datasets/my_dataset.hdf5`, set:
> ```python
> DATA_DIR = 'data_storage/datasets/my_dataset.hdf5'
> ```
> Also update `num_episodes` in the corresponding task config block to match the actual number of demos in your HDF5 file.

---

## Step 6 — Submit a training job

From your **local terminal**, run `cluster_interface.sh job` followed by all arguments that will be forwarded to `CLUSTER_PYTHON_EXECUTABLE`. The script automatically syncs your latest local code to a timestamped copy on the cluster and submits a SLURM job.

### ACT training example

```bash
./docker/cluster/cluster_interface.sh job \
    --task_name sim_transfer_cube_scripted \
    --ckpt_dir data_storage/260205_1550_small_ws_simplifiedTask_400_augmented \
    --policy_class ACT \
    --kl_weight 0.1 \
    --chunk_size 64 \
    --hidden_dim 512 \
    --batch_size 16 \
    --dim_feedforward 3200 \
    --num_epochs 5000 \
    --lr 5e-5 \
    --seed 1 \
    --temporal_agg \
    --context_length 4 \
    --velocity_control \
    --image_aug
```

The `--ckpt_dir` path is relative to `/workspace/isaaclab/` inside the container, so writing to `data_storage/...` means the checkpoint lands in `$CLUSTER_ISAACLAB_DIR/data_storage/...` on the cluster and persists after the job.

### Other entry point examples

The `CLUSTER_PYTHON_EXECUTABLE` in `.env.cluster` determines which script runs. To use a different entry point, change that variable and re-submit. Examples:

**Dataset generation (Isaac Lab Mimic):**
```bash
# Set CLUSTER_PYTHON_EXECUTABLE=scripts/imitation_learning/isaaclab_mimic/generate_dataset.py
./docker/cluster/cluster_interface.sh job \
    --device cuda \
    --num_envs 10 \
    --generation_num_trials 5000 \
    --input_file datasets/annotated_16_07.hdf5 \
    --output_file datasets/euler_test_5k.hdf5 \
    --task Isaac-Stack-Cube-Franka-IK-Abs-Mimic-RGB-v0 \
    --enable_cameras \
    --headless
```

**Robomimic training:**
```bash
# Set CLUSTER_PYTHON_EXECUTABLE=scripts/imitation_learning/robomimic/train.py
./docker/cluster/cluster_interface.sh job \
    --task Isaac-Stack-Cube-Franka-IK-Abs-Transformer-RGB-v0 \
    --algo bc \
    --dataset datasets/Datasets_benchmarking/euler__tst_21_12_5000.hdf5
```

---

### SLURM job script — `submit_job_slurm.sh`

The file `docker/cluster/submit_job_slurm.sh` defines the SLURM resource request and calls `run_singularity.sh` inside the job. The version used for this project:

```bash
#!/usr/bin/env bash

# Load the ETH proxy module so the compute node has internet access.
# Required because Isaac Sim loads assets from NVIDIA Nucleus at runtime.
module load eth_proxy

cat <<EOT > job.sh
#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --tmp=120g
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=mschmotzer@ethz.ch
#SBATCH --job-name="Test_aug$(date +"%Y-%m-%dT%H:%M")"

# Pass the cluster code directory ($1) and SIF image name ($2) to run_singularity.sh,
# followed by all script arguments (${@:3}) that were forwarded from cluster_interface.sh.
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
```

### Resource parameters explained

| Parameter | Value | Notes |
|---|---|---|
| `--cpus-per-task` | 4 | CPU cores per job. Increase if data loading becomes a bottleneck. |
| `--gpus` | `rtx_4090:1` | Requests one RTX 4090. Change to e.g. `rtx_3090:1` if unavailable. |
| `--time` | `6:00:00` | Wall-clock limit. ACT training for 5000 epochs typically fits in 6 h. |
| `--mem-per-cpu` | 8000 MB | Memory per CPU core, so 4 × 8 GB = 32 GB total RAM. |
| `--tmp` | 120 GB | Local scratch space on the compute node (`$TMPDIR`). The Singularity image and Isaac Sim cache are copied here at job start — 120 GB is the minimum safe value. |
| `--mail-type` | `END,FAIL,BEGIN` | Email notifications on job start, finish, and failure. |

### How arguments flow from local terminal to the container

```
cluster_interface.sh job [args...]
        │
        ├── rsync latest code → $CLUSTER_ISAACLAB_DIR_<timestamp>/
        │
        └── ssh → submit_job_slurm.sh  $code_dir  $sif_name  [args...]
                        │
                        └── sbatch → run_singularity.sh  $1=$code_dir  $2=$sif_name  ${@:3}=[args...]
                                            │
                                            └── singularity exec ... python $CLUSTER_PYTHON_EXECUTABLE [args...]
```

All arguments passed after `job` on the local command line are forwarded unchanged through this chain and eventually passed to the Python executable inside the container.

---

## Directory layout on the cluster

After a job submission the cluster layout looks like this:

```
/cluster/project/meboldt/student_Lucas_Michael/
│
├── isaaclab/                          # CLUSTER_ISAACLAB_DIR — permanent
│   ├── logs/                          # Training logs, always kept
│   └── data_storage/                  # Checkpoints, datasets, results
│
├── isaaclab_2026-02-05T15:50/         # Timestamped code copy for this job
│   ├── act_copy/
│   ├── scripts/
│   ├── source/
│   └── ...
│
├── docker-isaac-sim/                  # CLUSTER_ISAAC_SIM_CACHE_DIR — Isaac Sim cache
│   └── cache/kit, ov, pip, ...
│
└── scratch/
    └── isaac-lab-base.sif             # CLUSTER_SIF_PATH — Singularity image (shared)
```

Each job submission creates a new timestamped code directory so multiple runs with different code versions can execute simultaneously. The Singularity image and the `data_storage/` contents are shared across all jobs.

---

## Quick reference

| Step | Command | When |
|---|---|---|
| SSH to cluster | `ssh username@euler.ethz.ch` | First time |
| Create data dir | `mkdir -p ~/isaaclab/data_storage` | First time |
| Build Docker image | `./docker/container.py start && ./docker/container.py stop` | Once / after image changes |
| Push Singularity image | `./docker/cluster/cluster_interface.sh push` | Once / after image changes |
| Upload dataset | `rsync -avP dataset.hdf5 username@euler.ethz.ch:.../data_storage/datasets/` | Per dataset |
| Update `DATA_DIR` | Edit `act_copy/constants.py` to match upload path | Per dataset |
| Submit ACT training job | `./docker/cluster/cluster_interface.sh job [args...]` | Every run |
| Check job status | `ssh username@euler.ethz.ch squeue --me` | Any time |
