#!/usr/bin/env bash

echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2 and arguments ${@:3}"

#==
# Helper functions
#==

setup_directories() {
    # Check and create directories
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}

copy_dataset_if_needed() {
    # The command comes as a single string, split it into array
    local cmd_string="$1"
    echo "Command string: $cmd_string"
    # Split into array (unquoted expansion splits on spaces)
    local args=($cmd_string)
    echo "Total arguments after split: ${#args[@]}" >&2
    for ((i=0; i<${#args[@]}; i++)); do
        echo "arg $i = '${args[i]}'" 
    done
    for ((i=0; i<${#args[@]}; i++)); do
        if [[ "${args[i]}" == "--dataset" ]] && [[ $((i+1)) -lt ${#args[@]} ]]; then
            local dataset_path="${args[i+1]}"
            echo "✓ Dataset argument found: $dataset_path" >&2
            # Check if path is not empty
            if [[ "$dataset_path" ]]; then
                local full_dataset_path="$CLUSTER_ISAACLAB_DIR/$dataset_path"
                local tmpdir_dataset_path="$TMPDIR/$dir_name/$dataset_path"
                if [ -f "$full_dataset_path" ]; then
                    mkdir -p "$(dirname "$tmpdir_dataset_path")"
                    echo "Copying dataset from $full_dataset_path to compute node on $tmpdir_dataset_path" >&2
                    stdbuf -oL rsync -h --info=progress2  "$full_dataset_path" "$tmpdir_dataset_path" >&2
                    echo "Dataset copied successfully" >&2
                else
                    echo "✗ Warning: Dataset file not found at $full_dataset_path"
                fi
            fi
            break
        fi
    done
    # Return TMPDIR path on stdout (so it can be captured)
    echo "$tmpdir_dataset_path"
}

# Function to build command string with replaced dataset path
build_modified_command() {
    local new_dataset_path="$1"
    shift
    local result=""
    
    local i=0
    while [ $i -lt $# ]; do
        local arg="${@:$((i+1)):1}"
        if [[ "$arg" == "--dataset" ]]; then
            result="$result $arg"
            # Skip the old path and add new path
            i=$((i+1))
            result="$result $new_dataset_path"
        else
            result="$result $arg"
        fi
        i=$((i+1))
    done
    
    echo "$result"
}

#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the Isaac Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base

# make sure that all directories exists in cache directory
setup_directories
# copy all cache files
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the temporary isaaclab directory with the latest changes to the compute node
cp -r $1 $TMPDIR
# Get the directory name
dir_name=$(basename "$1")

# copy dataset to compute node if specified in arguments and get the local path
local_dataset_path=$(copy_dataset_if_needed "${@:3}")

# Prepare modified arguments with updated dataset path
if [[ -n "$local_dataset_path" ]]; then
    # The local dataset path is relative to /workspace/isaaclab inside the container
    # Remove the TMPDIR/$dir_name prefix to get the relative path
    # relative_local_path="${local_dataset_path#$TMPDIR/$dir_name/}"
    echo "Using local dataset path: $local_dataset_path" >&2
    # modified_command=$(build_modified_command "$relative_local_path" "${@:3}")
    modified_command=$(build_modified_command "$local_dataset_path" "${@:3}")
else
    # No dataset argument found, use original args
    modified_command="${@:3}"
fi

# copy container to the compute node
tar -xf $CLUSTER_SIF_PATH/$2.tar  -C $TMPDIR

# check args
echo "original commands: ${@:3}"
echo "modified commands: $modified_command"
echo "{modified_command}: ${modified_command}"
echo ""{modified_command}": "${modified_command}""

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim
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
    -B $CLUSTER_ISAACLAB_DIR/data_storage:/workspace/isaaclab/data_storage:rw \
    --nv --writable --containall $TMPDIR/$2.sif \
    bash -c "export ISAACLAB_PATH=/workspace/isaaclab && cd /workspace/isaaclab && /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${modified_command}"

# bash -c "export ISAACLAB_PATH=/workspace/isaaclab && cd /workspace/isaaclab && /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}"
# copy resulting cache files back to host
rsync -azPv $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

# if defined, remove the temporary isaaclab directory pushed when the job was submitted
if $REMOVE_CODE_COPY_AFTER_JOB; then
    rm -rf $1
fi

echo "(run_singularity.py): Return"
