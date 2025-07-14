#!/bin/bash
clear
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

echo "=> ================="
rm -rfv build
echo "=> ================="

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

echo "=> ================="
cp -rv ../image_test ./
cp -rv ../lg_v2.bin ./
cp -rv ../dfeat_640_640.bin ./
mkdir -v ./image_vis
echo "=> ================="

