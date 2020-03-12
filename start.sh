#!/bin/bash -e
all_ip=$(python3 utils.py --option readcfg_host)

# get the number of visible devices and host list
DOCKER_VISIBLE_DEVICES=0,1,2,3
readarray -d , -t strarr <<<"$DOCKER_VISIBLE_DEVICES"
process_num=${#strarr[*]}
all_ip_str=${all_ip[0]}:${process_num}
for(( id=1; id < ${#all_ip[@]}; id++ ))
do
	all_ip_str=${all_ip_str},${all_ip[$id]}:${process_num}
done

HOROVOD_IMAGE_V=cuda10.0_mx1.5.1_byteps
# get the number of hosts
host_num=${#all_ip[@]}

HOME_PATH=/root/hphu
LOCAL_HOME_PATH=/home/huhanpeng/hphu

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
function prePrepare {
	mkdir -p ${HOME_PATH}/log
	if [ ! -d "${HOME_PATH}/byteprofile_benchmark" ]; then
		exit 1
	fi
}
## To avoid integrating multiple operators into one single events
# \TODO: may influence the performance
function getDockerCmd {
	DOCKER_CMD="nohup nvidia-docker run \
                            -v ${HOME_PATH}:${HOME_PATH} \
                            -v /root/traces:/root/traces \
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
                            > ${HOME_PATH}/log/host$1.txt 2>&1 &"
}

function checkStatus {
	retstr=$(/usr/bin/expect launcher.sh "$1" "docker ps | grep $2")
	python3 utils.py --option status --retstr "${retstr}" --target "$2" --command "docker ps | grep $2"
}

if [ "$1" = "start" ]; then
	COMMAND="bash ${HOME_PATH}/byteps_launcher/example/mxnet-gluon/run_gluon_bert.sh 10"
	if [ "$2" = "install" ]; then
		DOCKER_IMAGE=haaanpeng/byteprofile:cuda10.0_mx1.5.1_byteps
		BPF_INSTALL=1
	else
		# no need to re-install horovod and gluon-nlp
		DOCKER_IMAGE=haaanpeng/byteprofile:${HOROVOD_IMAGE_V}
		BPF_INSTALL=0
	fi
	
	# launch workers
	for(( id=1; id < ${#all_ip[@]}; id++ ))
	do
		getDockerCmd ${id}
		/usr/bin/expect launcher.sh "${all_ip[$id]}" "mkdir -p ${HOME_PATH}/log && cd ${HOME_PATH} && ${DOCKER_CMD}"
	done
	# launch the first worker
	id=0
	getDockerCmd ${id}
	/usr/bin/expect launcher.sh "${all_ip[$id]}" "mkdir -p ${HOME_PATH}/log && cd ${HOME_PATH} && ${DOCKER_CMD}"
elif [ "$1" = "stop" ]; then
	# launch workers
	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		/usr/bin/expect launcher.sh "${all_ip[$id]}" "docker stop host${id} && docker rm host${id} &"
	done
elif [ "$1" = "status" ]; then
	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		checkStatus "${all_ip[$id]}" "host${id}"
	done
elif [ "$1" = "collect" ]; then
	cd ${LOCAL_HOME_PATH}
	if [ -s "${LOCAL_HOME_PATH}/traces" ]; then
		rm -rf ${LOCAL_HOME_PATH}/traces/*
	else
		mkdir -p ${LOCAL_HOME_PATH}/traces
	fi

	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		scp -r root@${all_ip[$id]}:/root/traces ${LOCAL_HOME_PATH}/traces/host$id
	done
elif [ "$1" = "tar" ]; then
	cd ${LOCAL_HOME_PATH}
	TAR_NAME="traces.tar"
	if [ -s "${TAR_NAME}" ]; then
        rm ${TAR_NAME}
	fi
	tar -cf ${TAR_NAME} traces/
	# scp ${TAR_NAME} huhanpeng@10.0.243.54:/Users/huhanpeng/
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
		retstr=$(/usr/bin/expect launcher.sh "${all_ip[$id]}" "${GPU_COMMAND}")
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
		retstr=$(/usr/bin/expect launcher.sh "${all_ip[$id]}" "${GPU_COMMAND}")
		python3 utils.py --option "status" --retstr "${retstr}"
	done
elif [ "$1" = "ip" ]; then
	if [ "$2" = "info" ]; then
		GPU_COMMAND="iptables -L -nv --line-numbers"
	elif [ "$2" = "rate" ]; then
		python3 utils.py --option "$1" --bash_arg "$1,$2,$3"
		LIMIT_SOURCES=$(python -c 'import sys; print(",".join(sys.argv[1:]))' "${all_ip[@]}")
		GPU_COMMAND="cd ${HOME_PATH} && iptables-restore < rules.v4 && \
                     iptables --new-chain RATE-LIMIT && \
                     iptables --append INPUT --match conntrack --ctstate NEW --jump RATE-LIMIT && \
                     iptables --append RATE-LIMIT -s ${LIMIT_SOURCES} --match limit --limit $3/sec --limit-burst $3 --jump ACCEPT && \
                     iptables --append RATE-LIMIT -s ${LIMIT_SOURCES} --jump DROP"
	elif [ "$2" = "reset" ]; then
		GPU_COMMAND="cd ${HOME_PATH} && iptables-restore < rules.v4"
	else
		echo "Argument Error!: unexpected \$2: '$2'"
		exit 1
	fi
	for(( id=0; id < ${#all_ip[@]}; id++ ))
	do
		retstr=$(/usr/bin/expect launcher.sh "${all_ip[$id]}" "${GPU_COMMAND}")
		python3 utils.py --option "status" --retstr "${retstr}"
	done
elif [ "$1" = "retrive" ]; then
	export BYTEPS_SERVER_MXNET_PATH=/root/incubator-mxnet
	export MXNET_GPU_WORKER_NTHREADS=1
	export BYTEPS_FORCE_DISTRIBUTED=1

	# profiling env
	export BYTEPS_TRACE_ON='1' 
	export BYTEPS_TRACE_END_STEP=20
	export BYTEPS_TRACE_START_STEP=10
	export BYTEPS_TRACE_DIR='./traces'

	# branch env
	export BYTEPS_BRANCH="${BYTEPS_BRANCH:-byteprofile}"
	export GLUON_NLP_BRANCH="${GLUON_NLP_BRANCH:-bert-byteprofile}"

	# ---- retrive 

	unset DMLC_ROLE
	unset DMLC_PS_ROOT_URI
	unset DMLC_PS_ROOT_PORT
	unset DMLC_NUM_WORKER 
	unset DMLC_NUM_SERVER
	unset DMLC_WORKER_ID 


	unset BYTEPS_TRACE_ON
	unset BYTEPS_TRACE_END_STEP
	unset BYTEPS_TRACE_START_STEP
	unset BYTEPS_TRACE_DIR
	unset MXNET_GPU_WORKER_NTHREADS

	unset https_proxy
	unset http_proxy
	unset no_proxy

	unset TRUNCATE_NORM
	unset BYTEPS_BRANCH
	unset GLUON_NLP_BRANCH
	unset DMLC_ROLE

	cd /usr/local/byteps 
	pip3 uninstall -y byteps
	python3 setup.py clean --all
	cd /usr/local 
	rm -rf byteps

	docker commit -c "ENV BYTEPS_TRACE_DIR='' BYTEPS_TRACE_START_STEP='' BYTEPS_TRACE_END_STEP='' BYTEPS_TRACE_ON='' " test hub.byted.org/arnold/lab.hphu.mxnet_byteps:${HOROVOD_IMAGE_V}
	docker push hub.byted.org/arnold/lab.hphu.mxnet_byteps:${HOROVOD_IMAGE_V}
	docker rmi hub.byted.org/arnold/lab.hphu.mxnet_byteps:${HOROVOD_IMAGE_V}
else
	echo "Argument Error!: unexpected '$1'"
	exit 1
fi

cd ${LOCAL_HOME_PATH}/byteps_launcher
