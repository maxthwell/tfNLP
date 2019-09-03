#!/bin/bash
echo $PWD

build_tfjob_base(){
docker build \
        --network host \
        --build-arg http_proxy=$http_proxy \
        --build-arg https_proxy=$https_proxy \
        --build-arg ftp_proxy=$ftp_proxy \
        --build-arg no_proxy=$no_proxy \
        -f Dockerfile.base \
         -t tfjob:base ../
}

build_tfjob(){
docker build \
        --network host \
        --build-arg http_proxy=$http_proxy \
        --build-arg https_proxy=$https_proxy \
        --build-arg ftp_proxy=$ftp_proxy \
        --build-arg no_proxy=$no_proxy \
        -f Dockerfile \
         -t dist-tfnlp-job:cpu ../
}

case $1 in
	base)
		build_tfjob_base
	;;
	*)
		build_tfjob
	;;
esac
