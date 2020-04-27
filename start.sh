#!/bin/bash -e

### Get all ip address of hosts
all_ip=($(python3 utils.py --option readcfg_host))
### Get the number of hosts
host_num=${#all_ip[@]}

## Get the number of visible devices and host list
DOCKER_VISIBLE_DEVICES=$(python3 utils.py --option readcfg_visible_device)
IFS=, read -a strarr <<<"$DOCKER_VISIBLE_DEVICES"
process_num=${#strarr[*]}
all_ip_str=${all_ip[0]}:${process_num}
for(( id=1; id < ${#all_ip[@]}; id++ ))
do
	all_ip_str=${all_ip_str},${all_ip[$id]}:${process_num}
done

### Read username and prompt info for log in
HOST_USERNAME=$(python3 utils.py --option readcfg_username)
HOST_PROMPT=$(python3 utils.py --option readcfg_prompt)
HOST_HOME_PATH=/home/net/hphu
HOST_TRACE_PATH=${HOST_HOME_PATH}/traces

DOCKER_HOME_PATH=/root/hphu
LOCAL_HOME_PATH=/home/net/hphu/local

HOROVOD_IMAGE_V=cuda10.0_mx1.5.0-v1.1_bpf_v1.0.10


function dockerStop {
	# $1 is the container name
	if [ "$(docker ps -a | grep $1)" ]; then
		docker stop $1 && docker rm $1
		echo "Stop container $1"
	fi
}
function dockerRemove {
	# $1 is the container name
	if [ "$(docker ps -a | grep $1)" ]; then
		docker rm $1
		echo "Remove container $1"
	fi
}

## To avoid integrating multiple operators into one single events
# \TODO: may influence the performance
function getDockerCmd {
	DOCKER_CMD="nohup nvidia-docker run \
                            -v ${HOST_HOME_PATH}:${DOCKER_HOME_PATH} \
                            -v ${HOST_TRACE_PATH}:/root/traces \
                            -v /root/.ssh:/root/.ssh \
                            -v /run/sshd:/run/sshd \
                            --shm-size 32768m \
                            --name host$1 \
                            --net=host \
                            -e NVIDIA_VISIBLE_DEVICES=${DOCKER_VISIBLE_DEVICES} \
                            -e DMLC_NUM_WORKER=${host_num}  \
                            -e DMLC_WORKER_ID=$1 \
                            -e BYTEPS_TRACE_END_STEP=30 \
                            -e HOST_LIST=${all_ip_str} \
                            -e BPF_INSTALL=${BPF_INSTALL} \
                            ${DOCKER_IMAGE} ${COMMAND} \
                            > ${HOST_HOME_PATH}/log/host$1.txt 2>&1 &"
}

function launchHost {
	/usr/bin/expect launcher.sh ${HOST_USERNAME} ${HOST_PROMPT} $@
}

function checkStatus {
	retstr=$(launchHost "$1" "docker ps | grep $2")
	python3 utils.py --option status --retstr "${retstr}" --target "$2" --command "docker ps | grep $2"
}

########################################################################
#######         Execute different commands accroding to $1        ###### 
########################################################################

if [ "$1" = "start" ]; then
	COMMAND="bash ${DOCKER_HOME_PATH}/byteps_launcher/example/mxnet-gluon/run_gluon_bert.sh 10"
	if [ "$2" = "install" ]; then
		DOCKER_IMAGE=haaanpeng/byteprofile:cuda10.0_mx1.5.0-v1.1_bpf_v1.0.7
		BPF_INSTALL=1
	else
		# no need to re-install horovod and gluon-nlp
		DOCKER_IMAGE=haaanpeng/byteprofile:${HOROVOD_IMAGE_V}
		BPF_INSTALL=0
	fi
	
	# launch workers
	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		getDockerCmd ${id}
		launchHost "${all_ip[$id]}" "mkdir -p ${HOST_HOME_PATH}/log && cd ${HOST_HOME_PATH} && ${DOCKER_CMD}"
	done

elif [ "$1" = "stop" ]; then
	# launch workers
	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		launchHost "${all_ip[$id]}" "docker stop host${id} && docker rm host${id} &"
	done

elif [ "$1" = "status" ]; then
	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		checkStatus "${all_ip[$id]}" "host${id}"
	done

elif [ "$1" = "collect" ]; then
	if [ -s "${LOCAL_HOME_PATH}/traces" ]; then
		rm -rf ${LOCAL_HOME_PATH}/traces/*
	else
		mkdir -p ${LOCAL_HOME_PATH}/traces
	fi
	cd ${LOCAL_HOME_PATH}

	### Remote copy trace files
	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		echo "Copying traces from ${HOST_USERNAME}@${all_ip[$id]}:${HOST_TRACE_PATH} to ${LOCAL_HOME_PATH}/traces/host$id"
		scp -r ${HOST_USERNAME}@${all_ip[$id]}:${HOST_TRACE_PATH} ${LOCAL_HOME_PATH}/traces/host$id
	done

	### Compress
	TAR_NAME="traces.tar"
	if [ -s "${TAR_NAME}" ]; then
        rm ${TAR_NAME}
	fi
	tar -cf ${TAR_NAME} traces/
	# scp ${TAR_NAME} huhanpeng@10.0.243.54:/Users/huhanpeng/

### Set the configuration of physical machines
elif [ "$1" = "gpu" ]; then
	if [ "$2" = "info" ]; then
		GPU_COMMAND="nvidia-smi --query-gpu=clocks.default_applications.memory --format=csv && \
                 nvidia-smi --query-gpu=clocks.applications.graphics --format=csv"
	elif [ "$2" = "ac" ]; then
		python3 utils.py --option "$1" --bash_arg "$1,$2,$3"
		GPU_COMMAND="nvidia-smi -ac 877,$3"
	elif [ "$2" = "reset" ]; then
		GPU_COMMAND="nvidia-smi -ac 877,1380"
	else
		echo "Argument Error!: unexpected \$2: '$2'"
		exit 1
	fi
	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		retstr=$(launchHost "${all_ip[$id]}" "${GPU_COMMAND}")
		python3 utils.py --option "status" --retstr "${retstr}"
	done
elif [ "$1" = "tc" ]; then
	if [ "$2" = "info" ]; then
		GPU_COMMAND="tc qdisc show dev eth0 | grep root"
	elif [ "$2" = "add" ]; then
		python3 utils.py --option "$1" --bash_arg "$1,$2,$3"
		LIMIT_SOURCES=$(python -c 'import sys; print(",".join(sys.argv[1:]))' "${all_ip[@]}")
		GPU_COMMAND="tc qdisc add dev eth0 root tbf rate $3gbit burst 100mbit latency 400ms"
	elif [ "$2" = "reset" ]; then
		GPU_COMMAND="tc qdisc del dev eth0 root"
	else
		echo "Argument Error!: unexpected \$2: '$2'"
		exit 1
	fi
	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		retstr=$(launchHost "${all_ip[$id]}" "${GPU_COMMAND}")
		python3 utils.py --option "status" --retstr "${retstr}"
	done
elif [ "$1" = "mlnx" ]; then
	if [ "$2" = "info" ]; then
		GPU_COMMAND="mlnx_qos -i eth10"
	elif [ "$2" = "add" ]; then
		python3 utils.py --option "$1" --bash_arg "$1,$2,$3"
		LIMIT_SOURCES=$(python -c 'import sys; print(",".join(sys.argv[1:]))' "${all_ip[@]}")
		GPU_COMMAND="mlnx_qos -i eth10 -r 0,0,0,$3,0,0,0,0"
	elif [ "$2" = "reset" ]; then
		GPU_COMMAND="mlnx_qos -i eth10 -r 0,0,0,0,0,0,0,0"
	else
		echo "Argument Error!: unexpected \$2: '$2'"
		exit 1
	fi
	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		retstr=$(/usr/bin/expect launcher.sh "${all_ip[$id]}" "${GPU_COMMAND}")
		python3 utils.py --option "status" --retstr "${retstr}"
	done
elif [ "$1" = "ip" ]; then
	if [ "$2" = "info" ]; then
		GPU_COMMAND="iptables -L -nv --line-numbers"
	elif [ "$2" = "rate" ]; then
		python3 utils.py --option "$1" --bash_arg "$1,$2,$3"
		LIMIT_SOURCES=$(python -c 'import sys; print(",".join(sys.argv[1:]))' "${all_ip[@]}")
		GPU_COMMAND="cd ${HOST_HOME_PATH} && iptables-restore < rules.v4 && \
                     iptables --new-chain RATE-LIMIT && \
                     iptables --append INPUT --match conntrack --ctstate NEW --jump RATE-LIMIT && \
                     iptables --append RATE-LIMIT -s ${LIMIT_SOURCES} --match limit --limit $3/sec --limit-burst $3 --jump ACCEPT && \
                     iptables --append RATE-LIMIT -s ${LIMIT_SOURCES} --jump DROP"
	elif [ "$2" = "reset" ]; then
		GPU_COMMAND="cd ${HOST_HOME_PATH} && iptables-restore < rules.v4"
	else
		echo "Argument Error!: unexpected \$2: '$2'"
		exit 1
	fi
	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		retstr=$(launchHost "${all_ip[$id]}" "${GPU_COMMAND}")
		python3 utils.py --option "status" --retstr "${retstr}"
	done
elif [ "$1" = "backup" ]; then
	### launch an image
	nvidia-docker run -it --shm-size 8G --net host --name byteprofile \
		-v /home/net/hphu/traces:/root/traces \
        -v /root/.ssh:/root/.ssh \
        -v /run/sshd:/run/sshd \
        -v /etc/libibverbs.d:/etc/libibverbs.d \
        haaanpeng/byteprofile:cuda10.0_mx1.5.0-v1.1_bpf_v1.0.10 /bin/bash

    ### reinstall nccl and horovod
    cd /root/nccl && make clean && make -j src.build && make pkg.txz.build && \
    mkdir -p /usr/local/nccl && \
    tar -Jxf ./build/pkg/txz/nccl*.txz -C /usr/local/nccl/ --strip-components 1 && \
    echo "/usr/local/nccl/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    ldconfig && ln -sf /usr/local/nccl/include/* /usr/include/

    cd /usr/local/horovod && python3 setup.py sdist && \
    HOROVOD_NCCL_HOME=/usr/local/nccl \
    HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_GPU_BROADCAST=NCCL \
    HOROVOD_WITH_MPI=1 \
    HOROVOD_WITHOUT_TENSORFLOW=1 \
    HOROVOD_WITHOUT_PYTORCH=1 \
    HOROVOD_WITH_MXNET=1 pip3 install --no-cache-dir dist/horovod* && \
    cp -r /usr/local/horovod/examples /root/horovod_examples


	docker run --rm -it --shm-size 32768m --runtime=nvidia --net host --name byteprofile haaanpeng/byteprofile:cuda10.0_mx1.5.0-v1.1_bpf_v1.0.1 /bin/bash
else
	echo "Argument Error!: unexpected '$1'"
	exit 1
fi

cd ${LOCAL_HOME_PATH}/bpf_launcher
