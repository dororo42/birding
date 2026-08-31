#!/bin/bash
LOG=/opt/birding_build/build3.log
exec > "$LOG" 2>&1
set -x
cd /opt/birding_build/ffmpeg-6.1.2
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:$PKG_CONFIG_PATH
./configure --prefix=/usr/local \
  --disable-everything --disable-doc --disable-ffplay --disable-ffprobe --disable-debug \
  --enable-pthreads --enable-network --enable-swscale --enable-hwaccels \
  --enable-version3 \
  --enable-protocol=tcp,udp,rtp,rtsp,file \
  --enable-demuxer=rtsp,sdp,h264,hevc --enable-parser=h264,hevc \
  --enable-decoder=h264_rkmpp,hevc_rkmpp \
  --enable-filter=hwdownload,format,null,buffer,scale \
  --enable-libdrm --enable-rkmpp \
  > /opt/birding_build/ffmpeg_conf.log 2>&1
CF_RC=$?
echo "ffmpeg configure rc=$CF_RC"
tail -8 /opt/birding_build/ffmpeg_conf.log
if [ $CF_RC -ne 0 ]; then
  echo "=== FFMPEG CONFIGURE FAILED ==="
  exit 1
fi
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
ffmpeg -hide_banner -decoders 2>/dev/null | grep -i rkmpp || echo "no rkmpp decoder"
echo "=== ALL DONE ==="
