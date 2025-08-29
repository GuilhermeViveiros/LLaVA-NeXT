submitter scslurm :: COMMANDS 
                  :: TASK REALIZATION CONFIGURATION {
    action run > exit_code {
        # Returns true iff the first parameter 
        # is the name of a defined variable
        function var_defined {
            eval '[[ -n ${'$1'+set} && ${'$1'} != "none" ]]'
        }

        # define script 
        wrapper="ducttape_job.sh"

        echo "#!/bin/bash" > $wrapper
        echo "set -e # stop on errors" >> $wrapper
        echo "set -o pipefail # stop on pipeline errors" >> $wrapper
        echo "set -u # stop on undeclared variables" >> $wrapper

        # setup modules - custom made for ist372485 on bsc
        echo "source /apps/GPP/ANACONDA/2024.02/etc/profile.d/conda.sh" >> $wrapper
        echo "conda activate llava-next-env" >> $wrapper
        echo "export CUDA_HOME=$CONDA_PREFIX" >> $wrapper
        echo "export PATH='$HOME/.conda/envs/llava-next-env/bin:$PATH'" >> $wrapper
        echo "export PATH='/home/ist/ist372485/ducttape-0.3:$PATH'" >> $wrapper
        echo "export PYTHONPATH='/home/ist/ist372485/LLaVA-NeXT/lmms-eval:$PYTHONPATH'" >> $wrapper

        # print actual jobs 
        echo "$COMMANDS" >> $wrapper

        # execute in background
        SLURM_ARGS="--job-name=$TASK"
        #if (var_defined C); then
        #    SLURM_ARGS+=" -C $C"
        #fi
        if (var_defined account); then
            SLURM_ARGS+=" -A $account"
        fi
        if (var_defined config); then
            SLURM_ARGS+=" -C ${config}"
        fi
        if (var_defined reservation); then
            SLURM_ARGS+=" --reservation=$reservation"
        fi
        if (var_defined nodes); then
            SLURM_ARGS+=" --nodes=$nodes --ntasks-per-node=1"
        fi
        if (var_defined cpus); then
            SLURM_ARGS+=" --cpus-per-task=$cpus"
        fi
        if (var_defined mem); then
            SLURM_ARGS+=" --mem=$mem"
        fi
        if (var_defined gres); then
            SLURM_ARGS+=" --gres=$gres"
        fi
        if (var_defined time); then
            SLURM_ARGS+=" --time=$time"
        fi
        if (var_defined partition); then
            SLURM_ARGS+=" --partition=$partition"
        fi
        if (var_defined qos); then
            SLURM_ARGS+=" --qos=$qos"
        fi

        SLURM_ARGS+=" --hint=nomultithread" 

        echo $SLURM_ARGS
        if (var_defined restart_on_timeout) && [ $restart_on_timeout == "true" ]; then
            echo "Will restart on timeout!"
            set +e # don't stop on errors
            # loop until the job completes successfully
            while true; do
                # submit the job and capture its exit code
                srun $SLURM_ARGS bash $wrapper
                exit_code=$?

                # if the job completed successfully, break the loop
                if [ $exit_code -eq 0 ]; then
                    break
                # else if code is not 143 (timeout) propagate error
                elif [ $exit_code -ne 143 ]; then
                    echo "Slurm job failed with exit code $exit_code"
                    exit $exit_code
                else
                    echo "Job timed out, resubmitting..."
                fi

                # otherwise, wait a bit and then continue the loop to resubmit the job
                sleep 60
            done
        else
            echo "Submitting job..."
            srun $SLURM_ARGS --hint=nomultithread bash $wrapper
        fi
  }
}