#!/bin/bash
WANDB_API_KEY=$(cat ./setup/docker/my_wandb_key)

script_and_args="${@:3}"
gpu=$1
name=$2
echo "Launching container $name on GPU $gpu"
docker run \
    --env CUDA_VISIBLE_DEVICES=$gpu \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e WANDB_AGENT_MAX_INITIAL_FAILURES=1000 \
    -e WANDB_AGENT_DISABLE_FLAPPING=true \
    -v $(pwd):/home/duser/rl \
    --name $name \
    --user $(id -u) \
    -d \
    -t kelvin_docker \
    /bin/bash -c "$script_and_args"

# Check the exit status of the docker run command
if [ $? -eq 0 ]; then
    echo "Docker run command succeeded."
    docker logs -f $name
    echo "Docker container name: $name"
    exit 0
else
    echo "Docker run command failed."
    exit 1
fi