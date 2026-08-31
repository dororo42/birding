/* mppdec — birding VPU 解码守护进程（方案A，v2）
 * stdin : h264 Annex-B 流（由 ffmpeg -c:v copy 从 RTSP 转出）
 * stdout: 帧序列，每帧 [magic "FRM1"][width u32le][height u32le][NV12 payload w*h*3/2，紧凑无 stride]
 * stderr: 日志
 *
 * 调用序列复刻 mpp-develop/test/mpi_dec_test.c 的成功路径：
 *   mpp_create → mpp_init(MPP_CTX_DEC, CodingAVC) → info_change 时
 *   mpp_buffer_group_get_internal(ION) + MPP_DEC_SET_EXT_BUF_GROUP + MPP_DEC_SET_INFO_CHANGE_READY
 *
 * v2 主循环（修复喂包速率不足导致的流卡顿）：
 *   每轮：非阻塞 select stdin（0 超时）→ 有数据则喂一个 chunk（put 失败时边 drain 边重试）
 *        → drain 一轮输出 → 若既无输入也无输出则 msleep(5)
 *   喂包失败（输出队列满）时丢弃该 chunk 而非退出（直播语义：迟到数据无用，等下一个 IDR）
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/select.h>

#include "rk_mpi.h"
#include "mpp_frame.h"
#include "mpp_packet.h"
#include "mpp_buffer.h"

#define HDR_SIZE 12
#define CHUNK_SZ (256 * 1024)
#define PUT_RETRY_MS 1000     /* put 重试上限 1s，失败丢弃该 chunk */

static void msleep(unsigned int ms) { usleep(ms * 1000); }

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int sig) { (void)sig; g_stop = 1; }

static void put_u32(unsigned char *p, unsigned int v)
{
    p[0] = (unsigned char)(v & 0xff);
    p[1] = (unsigned char)((v >> 8) & 0xff);
    p[2] = (unsigned char)((v >> 16) & 0xff);
    p[3] = (unsigned char)((v >> 24) & 0xff);
}

static long write_frame(FILE *out, MppFrame frame)
{
    RK_U32 w, h, hs, vs, i;
    MppFrameFormat fmt;
    MppBuffer buf;
    RK_U8 *base, *cbase;
    unsigned char hdr[HDR_SIZE];

    if (mpp_frame_get_errinfo(frame) || mpp_frame_get_discard(frame))
        return 0;
    buf = mpp_frame_get_buffer(frame);
    if (NULL == buf)
        return 0;

    w = mpp_frame_get_width(frame);
    h = mpp_frame_get_height(frame);
    hs = mpp_frame_get_hor_stride(frame);
    vs = mpp_frame_get_ver_stride(frame);
    fmt = mpp_frame_get_fmt(frame) & MPP_FRAME_FMT_MASK;
    if (fmt != MPP_FMT_YUV420SP)
        return 0;
    base = (RK_U8 *)mpp_buffer_get_ptr(buf);
    if (NULL == base)
        return 0;

    memcpy(hdr, "FRM1", 4);
    put_u32(hdr + 4, w);
    put_u32(hdr + 8, h);
    fwrite(hdr, 1, HDR_SIZE, out);

    for (i = 0; i < h; i++)
        fwrite(base + i * hs, 1, w, out);
    cbase = base + (size_t)hs * vs;
    for (i = 0; i < h / 2; i++)
        fwrite(cbase + i * hs, 1, w, out);
    fflush(out);
    return 1;
}

int main(void)
{
    MppCtx ctx = NULL;
    MppApi *mpi = NULL;
    MppPacket packet = NULL;
    MppBufferGroup grp = NULL;
    unsigned char *inbuf = NULL;
    MPP_RET ret;
    long frames = 0;
    int eos_sent = 0, eos_got = 0;
    unsigned long drops = 0;

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);
    signal(SIGPIPE, SIG_IGN);
    setvbuf(stdout, NULL, _IOFBF, 2 * 1024 * 1024);

    ret = mpp_create(&ctx, &mpi);
    if (ret) {
        fprintf(stderr, "mpp_create failed %d\n", ret);
        return 1;
    }
    ret = mpp_init(ctx, MPP_CTX_DEC, MPP_VIDEO_CodingAVC);
    if (ret) {
        fprintf(stderr, "mpp_init failed %d\n", ret);
        return 1;
    }
    ret = mpp_packet_init(&packet, NULL, 0);
    if (ret) {
        fprintf(stderr, "mpp_packet_init failed %d\n", ret);
        return 1;
    }
    inbuf = (unsigned char *)malloc(CHUNK_SZ);
    if (!inbuf)
        return 1;

    fprintf(stderr, "mppdec ready\n");

    while (!g_stop && !eos_got) {
        fd_set rfds;
        struct timeval tv = {0, 0};
        int rdy;
        int fed = 0;
        int idle = 0;
        int drained = 0;

        /* 1) 非阻塞喂入：stdin 有数据则喂一个 chunk */
        FD_ZERO(&rfds);
        FD_SET(0, &rfds);
        rdy = select(1, &rfds, NULL, NULL, &tv);
        if (rdy > 0 && !eos_sent) {
            size_t n = fread(inbuf, 1, CHUNK_SZ, stdin);
            if (n > 0) {
                mpp_packet_set_data(packet, inbuf);
                mpp_packet_set_size(packet, n);
                mpp_packet_set_pos(packet, inbuf);
                mpp_packet_set_length(packet, n);
                for (int t = 0; t < PUT_RETRY_MS / 10; t++) {
                    ret = mpi->decode_put_packet(ctx, packet);
                    if (ret == MPP_OK) { fed = 1; break; }
                    if (ret != MPP_ERR_TIMEOUT)
                        break;
                    /* 输出队列满：边 drain 边重试 */
                    MppFrame f = NULL;
                    if (mpi->decode_get_frame(ctx, &f) == MPP_OK && f) {
                        if (mpp_frame_get_eos(f))
                            eos_got = 1;
                        mpp_frame_deinit(&f);
                    }
                    msleep(5);
                }
                if (!fed) {
                    drops++;
                    fprintf(stderr, "put timeout, drop chunk #%lu (%zu bytes)\n", drops, n);
                }
            } else if (feof(stdin)) {
                mpp_packet_set_eos(packet);
                mpp_packet_set_length(packet, 0);
                mpi->decode_put_packet(ctx, packet);
                eos_sent = 1;
                fprintf(stderr, "input EOS sent\n");
            }
        }

        /* 2) drain 一轮输出（有帧就全部写完） */
        while (!g_stop) {
            MppFrame frame = NULL;
            RK_S32 times = 10;

        try_again:
            ret = mpi->decode_get_frame(ctx, &frame);
            if (MPP_ERR_TIMEOUT == ret && times > 0) {
                times--;
                msleep(1);
                goto try_again;
            }
            if (ret)
                break;
            if (NULL == frame) {
                idle = 1;
                break;
            }
            idle = 0;
            drained = 1;

            if (mpp_frame_get_info_change(frame)) {
                RK_U32 w = mpp_frame_get_width(frame);
                RK_U32 h = mpp_frame_get_height(frame);
                RK_U32 hs = mpp_frame_get_hor_stride(frame);
                RK_U32 vs = mpp_frame_get_ver_stride(frame);
                RK_U32 buf_size = mpp_frame_get_buf_size(frame);

                fprintf(stderr, "info change: w=%u h=%u hs=%u vs=%u buf_size=%u\n",
                        w, h, hs, vs, buf_size);
                if (NULL == grp) {
                    ret = mpp_buffer_group_get_internal(&grp, MPP_BUFFER_TYPE_ION);
                    if (ret) {
                        fprintf(stderr, "buffer group get failed %d\n", ret);
                        mpp_frame_deinit(&frame);
                        goto out;
                    }
                } else {
                    mpp_buffer_group_clear(grp);
                }
                ret = mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, grp);
                if (ret) {
                    fprintf(stderr, "set ext buf group failed %d\n", ret);
                    mpp_frame_deinit(&frame);
                    goto out;
                }
                ret = mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
                if (ret) {
                    fprintf(stderr, "info change ready failed %d\n", ret);
                    mpp_frame_deinit(&frame);
                    goto out;
                }
            } else {
                if (write_frame(stdout, frame))
                    frames++;
                if (mpp_frame_get_eos(frame)) {
                    fprintf(stderr, "eos frame got\n");
                    eos_got = 1;
                }
            }
            mpp_frame_deinit(&frame);
            if (eos_got)
                break;
        }
        if (eos_got)
            break;

        /* 3) 既无输入也无输出时小睡，避免空转 */
        if (idle && !fed && !drained)
            msleep(5);
    }

out:
    fprintf(stderr, "mppdec exit: frames=%ld drops=%lu eos_sent=%d eos_got=%d\n",
            frames, drops, eos_sent, eos_got);
    if (packet)
        mpp_packet_deinit(&packet);
    if (grp)
        mpp_buffer_group_put(grp);
    if (ctx)
        mpp_destroy(ctx);
    free(inbuf);
    fflush(stdout);
    return 0;
}
