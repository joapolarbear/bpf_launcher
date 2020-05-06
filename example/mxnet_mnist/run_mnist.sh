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
if [ "$HOME" = "/root" ]; then
    export ROOT_DIR=/root
else
    export ROOT_DIR=$HOME/hphu
    mkdir -p $ROOT_DIR
fi

### RDMA
export RDMA_DEVICE="${RDMA_DEVICE:-mlx5_0}"
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

# Get the batch size
if [ "$1" = "" ]; then
    BATCH_PER_GPU=10
else
    BATCH_PER_GPU=$1
fi
export DMLC_NUM_WORKER="${DMLC_NUM_WORKER:-1}"
TOTAL_BATCH_SIZE=$(($WORKER_GPU_NUM*$DMLC_NUM_WORKER*$BATCH_PER_GPU))
TOTAL_GPU_NUM=$(($WORKER_GPU_NUM*$DMLC_NUM_WORKER))
echo "Total batch size is $TOTAL_BATCH_SIZE"
echo "Total GPU num is $TOTAL_GPU_NUM"

### for horovod
export HOST_LIST="localhost:${WORKER_GPU_NUM}"
LISTEN_PORT=12345
echo "HOST_LIST:$HOST_LIST, PORT:$LISTEN_PORT"

########################## re-install something ##########################
# ----------------- re-install nccl -----------------
if [ "$NCCL_REINSTALL" = "1" ]; then
    cd ${ROOT_DIR}/nccl && git pull
    make -j src.build && make pkg.txz.build
    mkdir -p /usr/local/nccl 
    tar -Jxf ./build/pkg/txz/nccl*.txz -C /usr/local/nccl/ --strip-components 1 
    echo "/usr/local/nccl/lib" >> /etc/ld.so.conf.d/nvidia.conf 
    ldconfig
    ln -sf /usr/local/nccl/include/* /usr/include/
fi

if [ "$NCCL_REINSTALL" = "1" ]; then
    cd ${ROOT_DIR}
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

    git pull
    python3 setup.py sdist
    HOROVOD_NCCL_HOME=/usr/local/nccl \
    HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_GPU_BROADCAST=NCCL \
    HOROVOD_WITH_MPI=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_MXNET=1 pip3 install --no-cache-dir dist/horovod*

    ### install horovod
    cd /root/gluon-nlp
    python3 setup.py install
fi
########################################################################

### Create Trace directories and clear it
if [ ! -s $BYTEPS_TRACE_DIR ]; then
    mkdir -p $BYTEPS_TRACE_DIR
fi
rm -rf $BYTEPS_TRACE_DIR/*
for(( id=0; id < ${WORKER_GPU_NUM}; id++ ))
do
    GPU_PATH=$BYTEPS_TRACE_DIR/$id
    mkdir -p $GPU_PATH
done

# ---------------------- start to run ----------------------

### take different actions for different hosts
if [ "${DMLC_WORKER_ID}" = "0" ]; then
    IFS=, read -a HOST_IP_NP_LIST <<<"$HOST_LIST"
    # readarray -d , -t HOST_IP_NP_LIST <<<"$HOST_LIST"

    IFS=':'
    for(( id=0; id < ${#HOST_IP_NP_LIST[@]}; id++ ))
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
        -x HOROVOD_TIMELINE=$BYTEPS_TRACE_DIR \
        -x HOROVOD_LOG_LEVEL=warning \
        -x BYTEPS_TRACE_ON \
        -x BYTEPS_TRACE_DIR \
        -x BYTEPS_TRACE_START_STEP \
        -x BYTEPS_TRACE_END_STEP \
        -x NCCL_DEBUG=INFO \
        -x NCCL_DEBUG_SUBSYS=INIT \
        -x NCCL_ALGO=Ring \
        -bind-to none -map-by slot -mca plm_rsh_args '-p 12345' \
        -x LD_LIBRARY_PATH -x PATH -x MXNET_EXEC_BULK_EXEC_TRAIN \
        -mca pml ob1 -mca btl ^openib --allow-run-as-root \
        python3 /root/horovod_examples/mxnet_mnist.py
        
else
    /usr/sbin/sshd -p ${LISTEN_PORT}
    echo "start to listen to port ${LISTEN_PORT}"
    sleep infinity
fi

