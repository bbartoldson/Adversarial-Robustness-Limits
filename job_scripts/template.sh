#!/bin/sh
echo "START"
echo `which python`
date

job_type=$$1

firsthost=`jsrun --nrs 1 -r 1 /bin/hostname`
export MASTER_ADDR=$$firsthost
export MASTER_PORT=12321
export NUM_PROCESS_PER_NODE=${PROC_PER_NODE}

# Check if we need to submit one extra job if yes, submit it.
python submit.py --config $CONFIG_FILE chain --job-config $JOB_CONFIG_FILE

if [[ "$$job_type" == "Train" ]]; then
jsrun --smpiargs="-disable_gpu_hooks" -r ${PROC_PER_NODE} python train_wa_distributed.py --config $CONFIG_FILE
elif [[ "$$job_type" == "AutoAttack" ]]; then
  python eval-aa.py --config $CONFIG_FILE
fi


echo "END"
