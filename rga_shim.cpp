#include <cstdio>
#include <cstring>
#include <im2d.h>
#include <rga.h>

extern "C" {

/* 整帧 resize + 格式转换（同步，im2d 1.10.x API）。
   返回 0 成功，否则 IM_STATUS 错误码。 */
int shim_resize(void* src, int sw, int sh, int sfmt,
                void* dst, int dw, int dh, int dfmt) {
    rga_buffer_t s = wrapbuffer_virtualaddr(src, sw, sh, sfmt);
    rga_buffer_t d = wrapbuffer_virtualaddr(dst, dw, dh, dfmt);
    IM_STATUS st = imresize(s, d, 0, 0, IM_INTERP_DEFAULT, 1);
    /* 注意：im2d 1.10 有两个成功码 —— IM_STATUS_NOERROR=2 与 IM_STATUS_SUCCESS=1 */
    return (st == IM_STATUS_NOERROR || st == IM_STATUS_SUCCESS) ? 0 : (int)st;
}

} /* extern "C" */
