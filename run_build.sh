#!/bin/bash
clear
cd $(dirname "$0")
echo "=> curr dir: $(pwd)"

echo "=> ================="
rm -rfv build
echo "=> ================="

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. \
  -DPLATFORM_X5=ON \
  -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++
make -j$(nproc)

echo "=> ================="
cp -r ../image_test ./
mkdir ./image_vis
cp -r ../model ./
mkdir -p ./3rdparty/lib_opencv4.5.4/
cp -r ../3rdparty/lib_opencv4.5.4/ ./3rdparty/
cp -r ../make_ln.sh ./
tar -zcf DFMatch_X5.tar.gz \
--transform 's,^,DFMatch/,' \
./image_test ./image_vis ./model ./3rdparty ./make_ln.sh ./dfmatch_infer ./test_dfmatch
echo "=> output tar file: DFMatch_X5.tar.gz"
echo "=> ================="
md5sum dfmatch_infer
md5sum test_dfmatch
echo "=> ================="

