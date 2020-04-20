#!/bin/bash -e
# A script running in a container

## Set enviroment variables related to profiling
if [ "${BYTEPS_TRACE_ON}" = "1" ]; then
    echo "BYTEPS_TRACE_DIR: ${BYTEPS_TRACE_DIR}"
    echo "BYTEPS_TRACE_START_STEP: ${BYTEPS_TRACE_START_STEP}"
    echo "BYTEPS_TRACE_END_STEP: ${BYTEPS_TRACE_END_STEP}"
    mkdir -p ${BYTEPS_TRACE_DIR}
fi

### GPU info
if [ "$NVIDIA_VISIBLE_DEVICES" = "all" ]; then
    export WORKER_GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    export GPUS=$(seq -s "," 0 $(expr ${WORKER_GPU_NUM} - 1))
else
    IFS=, read -a strarr <<<"$NVIDIA_VISIBLE_DEVICES"
    export WORKER_GPU_NUM=${#strarr[*]}
    export GPUS=$NVIDIA_VISIBLE_DEVICES
fi 
echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
echo "WORKER_GPU_NUM: $WORKER_GPU_NUM"
echo "GPUS: $GPUS"

### root directory
export ROOT_DIR=/root
cd ${ROOT_DIR} 

# ---------------------- start to run ----------------------
DATA="/tmp/wiki_en_uncased_data/wiki_en_uncased_0*"
OPTIMIZER="bertadam"
## other evnvironment variables
export DMLC_ROLE="${DMLC_ROLE:-worker}"
# optimizer parameters
export LR=0.00354;   
export OPTIONS=--synthetic_data\ --eval_use_npz; 
export WARMUP_RATIO=0.1;          
export NUMSTEPS=281250;   
export CKPTDIR=ckpt_stage1_lamb_16k-682a361-c5fd6fc-0412-cu90; 
export ACC=1;  
# start
export TRUNCATE_NORM="${TRUNCATE_NORM:-1}"
export LAMB_BULK="${LAMB_BULK:-30}"
export EPS_AFTER_SQRT="${EPS_AFTER_SQRT:-1}"
export NUMSTEPS="${NUMSTEPS:-900000}"
export DTYPE="${DTYPE:-float16}"
export ACC="${ACC:-1}"
export MODEL="${MODEL:-bert_24_1024_16}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
export MAX_PREDICTIONS_PER_SEQ="${MAX_PREDICTIONS_PER_SEQ:-20}"
export LR="${LR:-0.000625}"
export LOGINTERVAL="${LOGINTERVAL:-10}"
export CKPTDIR="${CKPTDIR:-ckpt_stage1_lamb}"
export CKPTINTERVAL="${CKPTINTERVAL:-300000000}"
export OPTIMIZER="${OPTIMIZER:-lamb}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.003125}"
export BYTEPS_PARTITION_BYTES="${BYTEPS_PARTITION_BYTES:-4096000}"
export BYTEPS_NCCL_GROUP_SIZE="${BYTEPS_NCCL_GROUP_SIZE:-16}"
# export NVIDIA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
export DMLC_WORKER_ID="${DMLC_WORKER_ID:-0}"
export DMLC_NUM_WORKER="${DMLC_NUM_WORKER:-1}"
export NCCL_MIN_NRINGS="${NCCL_MIN_NRINGS:-16}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD:-120}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD:-120}"
export MXNET_SAFE_ACCUMULATION="${MXNET_SAFE_ACCUMULATION:-1}"
export OPTIONS="${OPTIONS:- }"
export DATA="${DATA:-/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train}"
export DATAEVAL="${DATAEVAL:-/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test}"

# Get the batch size
if [ "$1" = "" ]; then
    BATCH_PER_GPU=10
else
    BATCH_PER_GPU=$1
fi
TOTAL_BATCH_SIZE=$(($WORKER_GPU_NUM*$DMLC_NUM_WORKER*$BATCH_PER_GPU))
TOTAL_GPU_NUM=$(($WORKER_GPU_NUM*$DMLC_NUM_WORKER))
echo "Total batch size is $TOTAL_BATCH_SIZE"
echo "Total GPU num is $TOTAL_GPU_NUM"

### RDMA
# export RDMA_DEVICE="${RDMA_DEVICE:-mlx5_0}"
if [ "$RDMA_DEVICE" != "" ]; then
    RDMA_COMMAND="-x NCCL_IB_DISABLE=0 \
                    -x NCCL_IB_HCA=mlx5_0:1 \
                    -x NCCL_IB_GID_INDEX=3 \
                    -x HOROVOD_MPI_THREADS_DISABLE=1 "
    echo "Ebable RDMA!"
else
    RDMA_COMMAND="-x NCCL_IB_DISABLE=1"
    echo "Disable RDMA!"
fi

### for horovod
export HOST_LIST="${HOST_LIST:-localhost:${WORKER_GPU_NUM}}"
LISTEN_PORT=12345
echo "HOST_LIST:$HOST_LIST, PORT:$LISTEN_PORT"

### Create Trace directories
if [ ! -s $BYTEPS_TRACE_DIR ]; then
    mkdir -p $BYTEPS_TRACE_DIR
fi
rm -rf $BYTEPS_TRACE_DIR/*
for(( id=0; id < ${WORKER_GPU_NUM}; id++ ))
do
    GPU_PATH=$BYTEPS_TRACE_DIR/$id
    mkdir -p $GPU_PATH
done

### take different actions for different hosts
if [ "${DMLC_WORKER_ID}" = "0" ]; then
    IFS=, read -a HOST_IP_NP_LIST <<<"$HOST_LIST"
    # readarray -d , -t HOST_IP_NP_LIST <<<"$HOST_LIST"

    IFS=':'
    for(( id=1; id < ${#HOST_IP_NP_LIST[@]}; id++ ))
    do
        HOST_IP_NP=${HOST_IP_NP_LIST[$id]}
        HOST_INFO=($HOST_IP_NP)
        while true; do
            test_worker=`head -n 1 2>/dev/null < /dev/tcp/${HOST_INFO[0]}/${LISTEN_PORT}`
            if [ "$test_worker" == "" ]; then
                echo "Waiting for worker ${HOST_INFO[0]}:${LISTEN_PORT} to be ready"
            else
                break
            fi
            sleep 1m
        done
    done
    unset IFS

    mpirun -np ${TOTAL_GPU_NUM} -H ${HOST_LIST} \
        ${RDMA_COMMAND} \
        -x HOROVOD_FUSION_THRESHOLD=0 \
        -x HOROVOD_CYCLE_TIME=0 \
        -x HOROVOD_TIMELINE=$BYTEPS_TRACE_DIR/comm.json \
        -x BYTEPS_TRACE_ON \
        -x BYTEPS_TRACE_DIR \
        -x BYTEPS_TRACE_START_STEP \
        -x BYTEPS_TRACE_END_STEP \
        -x NCCL_DEBUG=INFO \
        -x NCCL_DEBUG_SUBSYS=INIT \
        -x NCCL_ALGO=Tree \
        -bind-to none -map-by slot -mca plm_rsh_args '-p 12345' \
        -x LD_LIBRARY_PATH -x PATH \
        -mca pml ob1 -mca btl ^openib --allow-run-as-root \
        python3 ${ROOT_DIR}/gluon-nlp/scripts/bert/run_pretraining.py \
            --data=$DATA \
            --data_eval=$DATAEVAL \
            --optimizer $OPTIMIZER \
            --warmup_ratio $WARMUP_RATIO \
            --num_steps $NUMSTEPS \
            --ckpt_interval $CKPTINTERVAL \
            --dtype $DTYPE \
            --ckpt_dir $CKPTDIR \
            --lr $LR \
            --accumulate $ACC \
            --model $MODEL \
            --max_seq_length $MAX_SEQ_LENGTH \
            --max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
            --num_data_workers 4 \
            --no_compute_acc \
            --comm_backend horovod \
            --log_interval $LOGINTERVAL \
            --total_batch_size $TOTAL_BATCH_SIZE \
            --total_batch_size_eval $TOTAL_BATCH_SIZE \
            --gpus $GPUS \
            --synthetic_data \
            $OPTIONS 
else
    /usr/sbin/sshd -p ${LISTEN_PORT}
    echo "start to listen to port ${LISTEN_PORT}"
    sleep infinity
fi

