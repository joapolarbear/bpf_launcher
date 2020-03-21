#!/bin/bash -e
# A script running in a container

## Set enviroment variables related to profiling
export BYTEPS_TRACE_ON="${BYTEPS_TRACE_ON:-1}"
export BYTEPS_TRACE_END_STEP="${BYTEPS_TRACE_END_STEP:-30}"
export BYTEPS_TRACE_START_STEP="${BYTEPS_TRACE_START_STEP:-10}"
export BYTEPS_TRACE_DIR="${BYTEPS_TRACE_DIR:-${HOME}/traces}"
# export BYTEPS_TRACE_DEBUG=1
export MXNET_GPU_WORKER_NTHREADS=1
export MXNET_EXEC_BULK_EXEC_TRAIN=0

## install dependencies
if [ "$BPF_INSTALL" = "1" ]; then
    apt-get update && apt-get install -y librdmacm-dev
    apt install -y libopenmpi-dev
fi

## other evnvironment variables
export USE_CUDA_PATH=/usr/local/cuda:/usr/local/cudnn/lib64 \
    PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/nccl/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/usr/local/lib:/usr/local/cudnn/lib64:/usr/local/cuda/lib64:/usr/local/nccl/lib/:$LIBRARY_PATH \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

export TRUNCATE_NORM="${TRUNCATE_NORM:-1}"
export BYTEPS_BRANCH="${BYTEPS_BRANCH:-byteprofile}"
export GLUON_NLP_BRANCH="${GLUON_NLP_BRANCH:-bert-byteprofile}"
export DMLC_ROLE="${DMLC_ROLE:-worker}"


### GPU info
echo "NVIDIA_VISIBLE_DEVICES: $NVIDIA_VISIBLE_DEVICES"
if [ "$NVIDIA_VISIBLE_DEVICES" = "all" ]; then
    export WORKER_GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    export GPUS=$(seq -s "," 0 $(expr ${WORKER_GPU_NUM} - 1))
else
    readarray -d , -t strarr <<<"$NVIDIA_VISIBLE_DEVICES"
    export WORKER_GPU_NUM=${#strarr[*]}
    export GPUS=$NVIDIA_VISIBLE_DEVICES
fi 

### NCCL
if [ "$NCCL_REINSTALL" = "1" ]; then
    cd /root
    git clone --recurse-submodules -b byteprofile https://github.com/joapolarbear/nccl.git
    cd nccl && \
    make -j src.build && make pkg.txz.build && \
    mkdir -p /usr/local/nccl && \
    tar -Jxf ./build/pkg/txz/nccl*.txz -C /usr/local/nccl/ --strip-components 1 && \
    echo "/usr/local/nccl/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig
    ln -sf /usr/local/nccl/include/* /usr/include/
fi

if [ "$NCCL_REINSTALL" = "1" ]; then
    cd /root
    if [ ! -s "nccl-tests" ]; then
        git clone --recurse-submodules https://github.com/NVIDIA/nccl-tests.git
    fi
    cd nccl-tests
    make clean && make
    ./build/all_reduce_perf -b 8 -e 256M -f 2 -g ${WORKER_GPU_NUM}
fi 

# ----------------- re-install horovod and gluon-nlp -----------------
if [ "$BPF_INSTALL" = "1" ]; then 
    cd /usr/local/horovod 
    pip3 uninstall -y horovod
    python3 setup.py clean --all

    cd /usr/local 
    rm -rf horovod
    git clone --recurse-submodules https://github.com/joapolarbear/horovod

    cd /usr/local/horovod
    python3 setup.py sdist
    HOROVOD_NCCL_HOME=/usr/local/nccl \
    HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_GPU_BROADCAST=NCCL \
    HOROVOD_WITH_MPI=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_MXNET=1 pip3 install --no-cache-dir dist/horovod*

    cd /root
    rm -rf gluon-nlp
    git clone -b $GLUON_NLP_BRANCH https://github.com/joapolarbear/gluon-nlp.git
    cd gluon-nlp
    python3 setup.py install
fi

### download dataset
mkdir -p /root/.mxnet/models
cd /root/.mxnet/models 
wget https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/vocab/book_corpus_wiki_en_uncased-a6607397.zip
apt-get install -y zip
unzip -o *.zip


# ---------------------- start to run ----------------------

DATA="/tmp/wiki_en_uncased_data/wiki_en_uncased_0*"
OPTIMIZER="bertadam"
cd /root    

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
echo "total batch size is $TOTAL_BATCH_SIZE"

if [ "${BYTEPS_TRACE_ON}" = "1" ]; then
    echo "BYTEPS_TRACE_DIR: ${BYTEPS_TRACE_DIR}"
    echo "BYTEPS_TRACE_START_STEP: ${BYTEPS_TRACE_START_STEP}"
    echo "BYTEPS_TRACE_END_STEP: ${BYTEPS_TRACE_END_STEP}"
    mkdir -p ${BYTEPS_TRACE_DIR}
fi

### RDMA
RDMA_DEVICE=mlx5_0 \
if [ "$RDMA_DEVICE" != "" ]; then
        export NCCL_IB_DISABLE=0 
        export NCCL_IB_HCA=mlx5_0:1 
        export NCCL_IB_GID_INDEX=3 
        export HOROVOD_MPI_THREADS_DISABLE=1
else
        export NCCL_IB_DISABLE=1 
fi

### Nvprof
export NVPROF="${NVPROF:-0}"
if [ "$NVPROF" = "1" ]; then
    NVPROF_COMMAND="nvprof --print-gpu-trace -o ${BYTEPS_TRACE_DIR}/nvprof/simpleMPI.%q{OMPI_COMM_WORLD_RANK}.nvvp"
else
    NVPROF_COMMAND=""
fi

### for horovod
export HOST_LIST="${HOST_LIST:-localhost:${WORKER_GPU_NUM}}"
LISTEN_PORT=12345

### take different actions for different hosts
if [ "${DMLC_WORKER_ID}" = "0" ]; then
    readarray -d , -t HOST_IP_NP_LIST <<<"$HOST_LIST"

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

    horovodrun --disable-cache -np ${TOTAL_GPU_NUM} \
                --timeline-filename $BYTEPS_TRACE_DIR/comm.json \
                --fusion-threshold-mb 0 \
                -H ${HOST_LIST} \
                -p ${LISTEN_PORT} \
                ${NVPROF_COMMAND} \
                --cycle-time-ms 0 \
        python3 /root/gluon-nlp/scripts/bert/run_pretraining.py \
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

