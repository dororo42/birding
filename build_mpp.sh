#!/bin/bash
# Tier B v2: mpp(develop) + ffmpeg(rkmpp) 一键构建（cmake 模块由 b64 解码安装）
LOG=/opt/birding_build/build2.log
exec > "$LOG" 2>&1
set -x
cd /opt/birding_build

echo "=== [1/5] install cmake modules ==="
mkdir -p cmakefix
base64 -d b64_mo.txt > cmakefix/merge_objects.cmake
base64 -d b64_vi.txt > cmakefix/version.in
wc -c cmakefix/merge_objects.cmake cmakefix/version.in

echo "=== [2/5] build mpp (develop) ==="
MPPDIR=mpp-develop
mkdir -p "$MPPDIR/build/cmake"
cp cmakefix/merge_objects.cmake cmakefix/version.in "$MPPDIR/build/cmake/"
cd "$MPPDIR/build"
rm -rf CMakeCache.txt CMakeFiles
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DVERSION_INFO:STRING="\"libmpp-develop-birding\"" \
  -DVERSION_CNT:STRING="1" \
  -DVERSION_HISTORY_0:STRING="\"develop-birding\""
CM_RC=$?
echo "mpp cmake rc=$CM_RC"
if [ $CM_RC -ne 0 ]; then
  echo "=== MPP CMAKE FAILED ==="
  exit 1
fi
nice -n 19 make -j2 > /opt/birding_build/mpp_make.log 2>&1
MK_RC=$?
echo "mpp make rc=$MK_RC"
if [ $MK_RC -ne 0 ]; then
  tail -20 /opt/birding_build/mpp_make.log
  exit 1
fi
make install > /dev/null 2>&1
ldconfig
ls -la /usr/local/lib/librockchip_mpp.so* 2>/dev/null || { echo "=== MPP INSTALL MISSING ==="; exit 1; }
cd /opt/birding_build

echo "=== [3/5] extract ffmpeg ==="
if [ ! -d ffmpeg-6.1.2 ]; then
  tar xJf ffmpeg.tar.xz
fi
ls -d ffmpeg-6.1.2 || { echo "FFMPEG SOURCE MISSING"; exit 1; }

echo "=== [4/5] configure ffmpeg ==="
cd ffmpeg-6.1.2
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:$PKG_CONFIG_PATH
./configure --prefix=/usr/local \
  --disable-everything --disable-doc --disable-ffplay --disable-ffprobe --disable-debug \
  --enable-pthreads --enable-network --enable-swscale --enable-hwaccels \
  --enable-protocol=tcp,udp,rtp,rtsp,file \
  --enable-demuxer=rtsp,sdp,h264,hevc --enable-parser=h264,hevc \
  --enable-decoder=h264_rkmpp,hevc_rkmpp \
  --enable-filter=hwdownload,format,null,buffer,scale \
  --enable-libdrm --enable-rkmpp \
  > /opt/birding_build/ffmpeg_conf.log 2>&1
CF_RC=$?
echo "ffmpeg configure rc=$CF_RC"
tail -15 /opt/birding_build/ffmpeg_conf.log
if [ $CF_RC -ne 0 ]; then
  echo "=== FFMPEG CONFIGURE FAILED ==="
  exit 1
fi

echo "=== [5/5] make ffmpeg ==="
nice -n 19 make -j2 > /opt/birding_build/ffmpeg_make.log 2>&1
MK_RC=$?
echo "ffmpeg make rc=$MK_RC"
if [ $MK_RC -ne 0 ]; then
  tail -20 /opt/birding_build/ffmpeg_make.log
  exit 1
fi
make install > /dev/null 2>&1
ldconfig
echo "=== RESULT ==="
which ffmpeg
ffmpeg -hide_banner -decoders 2>/dev/null | grep -i rkmpp || echo "no rkmpp decoder listed"
echo "=== ALL DONE ==="
