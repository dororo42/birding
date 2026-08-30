#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RGA 2D 加速封装（可选依赖，缺失/故障时自动回退 cv2）。

依赖：
  /usr/local/lib/librga.so    —— Rockchip 官方预编译（rga_api 1.10.6）
  /usr/local/lib/rga_shim.so  —— 本项目编译的薄封装（避免 ctypes 复刻 rga_buffer_t 结构体）
"""
import ctypes
import os

# rga.h 中的格式枚举（<<8 为 legacy 编码，im2d 直接使用）
RK_FORMAT_RGBA_8888 = 0x0 << 8
RK_FORMAT_RGB_888 = 0x2 << 8
RK_FORMAT_BGR_888 = 0x7 << 8
RK_FORMAT_YCbCr_420_SP = 0xA << 8   # NV12
RK_FORMAT_YCrCb_420_SP = 0xE << 8   # NV21

_SHIM = "/usr/local/lib/rga_shim.so"


class RGAAccel:
    """加载失败或调用失败时 ok=False，调用方回退 cv2。"""

    def __init__(self):
        self.ok = False
        self.err = None
        self.lib = None
        try:
            lib = ctypes.CDLL(_SHIM)
            lib.shim_resize.restype = ctypes.c_int
            lib.shim_resize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            self.lib = lib
            self.ok = True
        except OSError as e:
            self.err = str(e)

    def _addr(self, arr):
        import numpy as np
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr.ctypes.data, arr

    def resize(self, src, sw, sh, sfmt, dst, dw, dh, dfmt):
        """整帧 resize + 格式转换（同步）。返回 0 成功。"""
        if not self.ok:
            return -1
        sa, _ = self._addr(src)
        da, _ = self._addr(dst)
        return int(self.lib.shim_resize(sa, sw, sh, sfmt, da, dw, dh, dfmt))

    def letterbox(self, src, sw, sh, sfmt, dst, dw, dh, dfmt, gray, dx, dy, rw, rh):
        """letterbox：numpy 填边 + RGA 缩放到子区域尺寸 + numpy 拼合。
        src/dst 均传 numpy 数组（dst 为 640×640×3 画布）。返回 0 成功。"""
        if not self.ok:
            return -1
        import numpy as np
        dst[...] = gray
        tmp = np.empty((rh, rw, 3), dtype=np.uint8)
        rc = self.resize(src, sw, sh, sfmt, tmp, rw, rh, dfmt)
        if rc != 0:
            return rc
        dst[dy:dy + rh, dx:dx + rw] = tmp
        return 0
