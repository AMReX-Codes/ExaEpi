echo "Running 1 job on the GPU"
start=`date +%s`
srun ./agent ../../examples/inputs.census >run0.log
end=`date +%s`
runtime=$((end-start))
echo "Runtime: $((runtime)) seconds"

srun --ntasks-per-node 1 dcgmi profile --pause
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=33
export SLURM_CPU_BIND="cores"


if [ $SLURM_PROCID -eq 0 ]; then
    nvidia-cuda-mps-control -d
fi

echo "Colocating 3 identical jobs on the GPU"
start=`date +%s`
srun ./agent ../../examples/inputs.census >run1.log &
srun ./agent ../../examples/inputs.census >run2.log &
srun ./agent ../../examples/inputs.census >run3.log 
end=`date +%s`
runtime=$((end-start))
echo "Runtime for all jobs: $((runtime)) seconds"

if [ $SLURM_PROCID -eq 0 ]; then
    echo quit | nvidia-cuda-mps-control
fi







