#!/usr/bin/env bash

set -e

ROS_DISTRO=${1:-"none"}
BASE_IMAGE=$2
# NEMO_VERSION="1.0.0rc1"
NEMO_VERSION="1.12.0"

# find container tag from os version
source docker/tag.sh

if [ $ARCH = "aarch64" ]; then
	if [ -z $BASE_IMAGE ]; then
		if [ $L4T_VERSION = "35.1.0" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r35.1.0-py3"
		else
			echo "cannot build jetson-voice docker container for L4T R$L4T_VERSION"
			echo "please upgrade to the latest JetPack, or build jetson-voice natively"
			exit 1
		fi
	fi
elif [ $ARCH = "x86_64" ]; then
	BASE_IMAGE=${BASE_IMAGE:-"nvcr.io/nvidia/nemo:$NEMO_VERSION"}
fi

VOICE_CONTAINER="$CONTAINER_NAME:$TAG"

echo "CONTAINER=$VOICE_CONTAINER"
echo "BASE_IMAGE=$BASE_IMAGE"

# build the container
sudo nvidia-docker build -t $VOICE_CONTAINER -f Dockerfile.$ARCH \
          --build-arg BASE_IMAGE=$BASE_IMAGE \
		--build-arg NEMO_VERSION=$NEMO_VERSION \
		.

# build ROS version of container
if [[ "$ROS_DISTRO" != "none" ]] ; then
	ROS_CONTAINER="$VOICE_CONTAINER-ros-$ROS_DISTRO"
	ROS_CONTAINER_BASE="$ROS_CONTAINER-base"
	
	# copy files needed to build ROS container
	if [ ! -d "packages/" ]; then
		cp -r docker/containers/packages packages
	fi
	
	# opencv.csv mounts files that preclude us installing different version of opencv
	# temporarily disable the opencv.csv mounts while we build the container
	CV_CSV="/etc/nvidia-container-runtime/host-files-for-container.d/opencv.csv"

	if [ -f "$CV_CSV" ]; then
		sudo mv $CV_CSV $CV_CSV.backup
	fi
	
	echo "CONTAINER=$ROS_CONTAINER_BASE"
	echo "BASE_IMAGE=$VOICE_CONTAINER"

	sudo nvidia-docker build -t $ROS_CONTAINER_BASE -f docker/containers/Dockerfile.ros.$ROS_DISTRO \
          --build-arg BASE_IMAGE=$VOICE_CONTAINER \
		.
	
	# restore opencv.csv mounts
	if [ -f "$CV_CSV.backup" ]; then
		sudo mv $CV_CSV.backup $CV_CSV
	fi
fi
