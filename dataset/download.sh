#!/bin/bash

function main() {
    public_server="http://lmflow.org:5000"
    if [ $# -lt 1 -o "$1" = "-h" -o "$1" = "--help" ]; then
        echo "Usage: bash $(basename $0) dataset_name"
        echo "Example: bash $(basename $0) coco"
        echo "Example: bash $(basename $0) all"
    fi

    if [ "$1" = "coco" -o "$1" = "all" ]; then
        echo "downloading coco"
        filename='coco_data.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi

#    if [ "$1" = "coco_data" -o "$1" = "all" ]; then
#        echo "downloading coco_task_annotation"
#        filename='coco_data.tar.gz'
#        wget ${public_server_detgpt}/${filename}
#    fi

    if [ "$1" = "cc_sbu_align" -o "$1" = "all" ]; then
        echo "downloading cc_sbu_align"
        filename='cc_sbu_align.tar.gz'
        wget ${public_server}/${filename}
        tar zxvf ${filename}
        rm ${filename}
    fi
}

main "$@"