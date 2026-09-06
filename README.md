# 观鸟识别服务（Birding Service）— 项目说明与轻量迁移指南

一个运行在 ARM 单板（Armbian / RK356x aarch64）上的实时鸟类识别服务：
通过 RTSP 拉取摄像头视频流，用 YOLOv8n（COCO 类别 14 = 鸟）做目标检测，
命中后自动截图并按日期归档，同时提供 Web 界面实时预览、照片浏览与参数调节。

> 本目录为**轻量迁移包**：仅含源代码、前端、模型与部署脚本，**不含** Python 虚拟环境（venv）与缓存。
> 环境重建步骤见下文「部署 / 环境重建」。

***

## 1. 运行环境

| 项目     | 说明                                                                                                 |
| ------ | -------------------------------------------------------------------------------------------------- |
| 操作系统   | Armbian（Debian 11，内核 6.1.57，RK356x / aarch64）                                                      |
| Python | 3.x，使用独立虚拟环境 `/home/birding/venv`                                                                  |
| 推理框架   | Ultralytics YOLOv8n；**可插拔后端**：CPU（`device='cpu'`，默认）或 Rockchip NPU（float16 RKNN，需 `rknnlite`）      |
| Web 框架 | FastAPI + Uvicorn                                                                                  |
| 模型文件   | `yolov8n.pt`（YOLOv8 nano 预训练权重，本包已附带）；NPU 另需 `yolov8n_fp16.rknn`（float16 RKNN，约 7.7 MB，需自行转换，见 §7） |
| 服务端口   | `8000`（HTTP / MJPEG 流）                                                                             |
| 摄像头    | RTSP 网络摄像头（见第 3 节配置）                                                                               |

> 依赖清单见 `requirements.txt`（来自设备 venv 的 `pip freeze` 完整导出）。
> 实际运行只需其中一部分，最小安装集合见第 5 节。

***

## 2. 目录结构

```
birding/
├── README.md                      # 本文档
├── birding_service.py             # 【主程序】FastAPI + YOLO 检测主逻辑（含 __main__ 入口）
├── frontend/
│   ├── index.html                 # Web 主界面（结构 + 样式，逻辑外链 script.js）
│   └── script.js                  # 前端交互逻辑（已抽离，原内联代码）
├── install_birding_autostart.sh   # 一键安装脚本：建 venv、装依赖、写 service、开机自启
├── birding.service                # systemd 单元文件（部署时复制到 /etc/systemd/system/）
├── requirements.txt               # 完整依赖清单（pip freeze，用于精确复现）
├── requirements-min.txt           # 最小运行依赖（仅服务运行所需）
├── archive/                       # 历史/参考版本（v3-kimi、早期 backup、旧界面），不纳入版本库
├── yolov8n.pt                     # YOLOv8n 模型权重（约 6.3 MB，运行时加载；已 gitignore，首跑可自动下载）
└── yolov8n_fp16.rknn              # NPU 后端模型（float16 RKNN，约 7.7 MB；由 convert_rknn_fp16.py 生成，不纳入版本库）
```

**本包刻意不包含**（按需求「环境配置不下载」）：

- `/home/birding/venv/`（约 1.2 GB 的虚拟环境）

- `__pycache__/`、`*.pyc`

- `/home/bird_image/`（运行期产生的鸟类截图，体积随时间增长）

- 系统级配置（ssh、ufw、NetworkManager 连接等）

***

## 3. 运行前配置（修改 `birding_service.py` 顶部常量）

| 常量                            | 默认值                               | 含义                                                                    |
| ----------------------------- | --------------------------------- | --------------------------------------------------------------------- |
| `CAMERA_IP`                   | `192.168.2.222`                   | 摄像头 IP（config.json 可覆盖；当前现场为 `192.168.2.162`）             |
| `CAMERA_IP_FALLBACKS`         | 空（逗号分隔 IP 列表）            | 摄像头候选地址：主地址连续 4 次连接失败自动轮换（应对 DHCP 换址）        |
| `USERNAME` / `PASSWORD`       | `root` / `1234qwer`               | 摄像头 RTSP 鉴权账号密码                                                       |
| `CONF_THRESHOLD`              | `0.3`                             | 检测置信度阈值（也可 Web 实时调）                                                   |
| `IOU_THRESHOLD`               | `0.3`                             | 连续帧匹配 IoU（去重匹配用）                                                      |
| `NMS_IOU_THRESHOLD`           | `0.5`                             | 检测 NMS 阈值（CPU / NPU 后端统一使用）                                           |
| `DEBUG_MODE`                  | `False`                           | 调试日志开关；**生产默认 False，仅入库鸟类**（True 时非鸟类别也会存盘，仅调试用）                      |
| `SAVE_COOLDOWN`               | `2`                               | 同位置再次保存冷却（秒）                                                          |
| `DETECT_EVERY_N_FRAMES`       | `2`                               | 每 N 帧做一次推理                                                            |
| `CONSECUTIVE_FRAMES_REQUIRED` | `1`                               | 连续确认帧数                                                                |
| `WORK_START` / `WORK_END`     | `05:00` / `17:00`                 | 观察时段，时段外休眠不检测                                                         |
| `IMAGE_ROOT`                  | `/home/bird_image`                | 截图归档根目录（按 `YYYYMMDD/` 与 `thumb/` 分存）                                  |
| `BIRD_CLASS_ID`               | `14`                              | COCO 类别 14 = 鸟（YOLOv8n 沿用 COCO 80 类）                                  |
| `RTSP_TRANSPORT`              | `tcp`                             | RTSP 传输协议（tcp 抗丢包；udp 易花屏）                                            |
| `RTSP_STREAM`                 | `ch0_0.h264`                      | RTSP 路径；可改 `ch0_1.h264` 子码流降低解码压力与延迟                                  |
| `RTSP_READ_TIMEOUT_SEC`       | `10`                              | RTSP 读超时（秒），死流时自动断开重连                                                 |
| `OFFLINE_ALARM_SEC`           | `600`                             | OFFLINE 持续超过该秒数 `/healthz` 返回 503（配合外部巡检自愈）                                        |
| `CONFIG_PATH`                 | `/home/birding/config.json`       | 运行时参数持久化文件（Web 修改的参数重启后自动恢复）                                          |
| `DETECT_BACKEND`              | `cpu`                             | 检测后端：`cpu`（默认，兼容性强）或 `npu`（Rockchip NPU / float16 RKNN，约 12× 更快、精度一致） |
| `NPU_MODEL_PATH`              | `/home/birding/yolov8n_fp16.rknn` | NPU 后端模型路径（float16 RKNN）                                              |
| `CPU_MODEL_PATH`              | `/home/birding/yolov8n.pt`        | CPU 后端模型路径（YOLOv8n PT）                                                |

**RTSP 地址拼接规则**：`rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/ch0_0.h264`

> 摄像头 IP / 阈值 / 时段 / 检测参数均可通过 Web 界面（POST 接口）在运行时动态修改，无需改代码重启。
>
> **环境变量覆盖（可选）**：除摄像头凭据外，其余运行参数均可用环境变量覆盖默认值，便于容器/不同场地部署：
> `CONF_THRESHOLD`、`IOU_THRESHOLD`、`NMS_IOU_THRESHOLD`、`DEBUG_MODE`、`SAVE_COOLDOWN`、`DETECT_EVERY_N_FRAMES`、
> `CONSECUTIVE_FRAMES_REQUIRED`、`WORK_START_HOUR`、`WORK_END_HOUR`、`IMAGE_ROOT`、
> `DETECT_BACKEND`、`NPU_MODEL_PATH`、`CPU_MODEL_PATH`、`RTSP_TRANSPORT`、`RTSP_STREAM`、`RTSP_READ_TIMEOUT_SEC`、`CONFIG_PATH`。
> 摄像头凭据（`USERNAME`/`PASSWORD`/`CAMERA_IP`）按需求保留为源码内的硬编码默认值。
>
> **参数持久化**：Web 界面修改的参数（IP/阈值/时段/检测参数/后端）会自动写入 `CONFIG_PATH`，重启/断电后自动恢复；删除该文件即回退默认值。

***

## 4. HTTP 接口

| 方法   | 路径                                                          | 说明                                         |
| ---- | ----------------------------------------------------------- | ------------------------------------------ |
| GET  | `/`                                                         | Web 主界面（实时预览 + 控制台）                        |
| GET  | `/video_feed`                                               | MJPEG 实时视频流（带检测框）                          |
| GET  | `/healthz`                                                  | 健康检查分级：ERROR/DEAD、采集线程死亡、OFFLINE 超过 `OFFLINE_ALARM_SEC` 均返回 503（SLEEPING 休眠不算不健康） |
| GET  | `/status`                                                   | 服务状态 JSON（state / conf / camera / 统计 + `frame_age_sec` / `last_error` / `camera_candidates`） |
| GET  | `/api/photos?date=YYYYMMDD&limit=50`                        | 某日截图列表（含缩略图/原图 URL）                        |
| GET  | `/api/dates`                                                | 有截图的日期列表及数量                                |
| POST | `/update_camera` (form: `ip`)                               | 切换摄像头 IP 并自动重连                             |
| POST | `/update_yolo` (form: `conf`)                               | 调整置信度阈值（0.1–0.9）                           |
| POST | `/update_worktime` (form: `start`,`end`)                    | 调整观察时段（小时）                                 |
| POST | `/update_detection_params` (form: `interval`,`consecutive`) | 调整检测间隔/确认帧数                                |
| POST | `/update_backend` (form: `backend`)                         | 运行时切换检测后端（`cpu` / `npu`，无需重启进程）            |
| POST | `/snapshot`                                                 | 手动保存最近一帧（仅当检测到鸟）                           |
| POST | `/snapshot_simple`                                          | 手动保存当前画面（无条件）                              |
| 静态   | `/images/...`                                               | 截图与缩略图目录（由 `IMAGE_ROOT` 挂载）                |
| 静态   | `/static`                                                   | 前端目录（`frontend/`）                          |

***

## 5. 部署 / 环境重建（轻量迁移步骤）

### 5.1 准备目录

```bash
APP_DIR=/home/birding
mkdir -p "$APP_DIR" /home/bird_image
cp -r ./frontend "$APP_DIR"/
cp birding_service.py birding_service-v3-kimi.py birding_service.py.backup "$APP_DIR"/
cp yolov8n.pt "$APP_DIR"/
chmod 755 /home/bird_image
```

### 5.2 创建虚拟环境并安装依赖

```bash
python3 -m venv "$APP_DIR/venv"
source "$APP_DIR/venv/bin/activate"

# 最小运行集合（推荐先装这些）：
pip install --upgrade pip
pip install fastapi uvicorn jinja2 opencv-python-headless ultralytics numpy pillow python-multipart

# 若需复现设备上的完整环境（含 matplotlib/polars/psutil/scipy 等开发依赖）：
# pip install -r requirements.txt
```

> ⚠️ `torch` / `torchvision` 在 aarch64 上体积较大，`ultralytics` 会自动拉取；
> 在 RK356x 等 ARM 板上请用 CPU 版（本服务推理默认 `device='cpu'`）。
>
> 若启用 NPU 后端（`DETECT_BACKEND=npu`），需额外安装 `rknn-toolkit-lite2`（提供 `rknnlite`），
> 且系统须自带 Rockchip NPU 驱动（DRM `/dev/dri/renderD12x`）。NPU 后端加载失败会自动回退 CPU，缺失依赖不会崩溃。

### 5.3 安装为 systemd 服务（开机自启）

复制单元文件并启用（也可直接运行 `install_birding_autostart.sh`）：

```bash
cp birding.service /etc/systemd/system/birding.service
systemctl daemon-reload
systemctl enable --now birding.service
journalctl -u birding.service -f   # 查看日志
```

服务单元关键项（与设备一致）：

```
WorkingDirectory=/home/birding
# 主程序已内置 __main__ 入口（uvicorn.run），直接运行即可启动 Web 服务
ExecStart=/home/birding/venv/bin/python3 /home/birding/birding_service.py
User=root
Restart=on-failure
Environment=DEBUG_MODE=false
Environment=RGA_ENABLED=0        # 2026-09-06 假死排查后回退（见 §11）
Environment=RTSP_BACKEND=cv2     # 同上：MPP 硬解与内核页表损坏相关，回退 cv2 软解
Environment=PYTHONUNBUFFERED=1
Environment=RTSP_IDLE_SEC=30
Environment=OFFLINE_ALARM_SEC=600
TimeoutStopSec=15
```

设备侧另已布置（不在仓库内，见 §11）：
硬件看门狗（`system.conf.d/watchdog.conf` → `RuntimeWatchdogSec=10`）、
`birding-health.timer`（每 5 分钟探活 /healthz，无响应自动重启服务）、
`crash-watch.service`（每 5 秒落盘 `/root/crash_watch.log`：uptime/温度/内存/负载/网关探针）。

### 5.4 访问

- Web 界面：`http://<设备IP>:8000`

- 视频流：`http://<设备IP>:8000/video_feed`

***

## 6. 轻量迁移清单（checklist）

- [ ] `birding_service.py` 主程序

- [ ] `frontend/`（界面，JS 已内联）

- [ ] `yolov8n.pt` 模型权重（或首次运行时由 ultralytics 自动下载）

- [ ] 目标机已装 Python3 + `venv`，按 5.2 重建环境

- [ ] 摄像头 RTSP 地址/账号密码（按 5.1 / 第 3 节核对 `CAMERA_IP` 等）

- [ ] 放行端口 `8000`（如启用防火墙，仅对局域网开放）

- [ ] 写入 `birding.service` 并 `systemctl enable`

***

## 7. 注意事项与已知问题

1. **设备网络**：原部署机使用 NetworkManager 管理 `eth0`（有线）与 `wlan1`（USB 网卡）。
   本服务的可访问性依赖设备网络配置，迁移到新板时请单独配置网络。
2. **模型类别**：当前仅识别 COCO 类别 14（鸟）。**生产默认** **`DEBUG_MODE=False`，仅入库鸟类**；
   设 `True` 时其它类别也会存盘与绘制，仅调试用。检测 NMS 统一用 `NMS_IOU_THRESHOLD=0.5`（CPU/NPU 一致）。
3. **图片落盘**：`/home/bird_image` 会随运行持续增长，建议定期清理或挂载大容量存储。
4. **安全性**：`birding.service` 以 `root` 运行、监听 `0.0.0.0:8000` 且无鉴权。
   局域网调试可接受；若暴露到公网，务必加反向代理鉴权并改为非 root 用户运行（`birding.service` 已留 `User=birding` 注释示例）。

   - 摄像头账号密码仍**硬编码在源码** `birding_service.py` 顶部（按需求保留），部署到新场地请直接改源码或后续改为 `.env` 外部化。

   - CORS 已关闭 `allow_credentials`（`allow_origins=["*"]` + credentials 原组合对纯内网无意义且冲突），如需跨域再按需放开。
5. **`frontend/script.js`**：交互逻辑已从 `index.html` 内联代码抽离到该文件，由 `<script src="/static/script.js">` 加载；照片墙改为增量刷新（集合无变化不重渲染）。
6. **历史版本**：`birding_service-v3-kimi.py`、`birding_service.py.backup`、`index-v3-kimi.html` 已移入 `archive/` 仅作参考，未纳入版本库；当前线上运行以 `birding_service.py` + `frontend/index.html` 为准。
7. **NPU 后端（Rockchip RK356x）**：通过 `DETECT_BACKEND=npu` 启用，复用板载 NPU 加速推理。

   - **模型须为 float16 RKNN**：`yolov8n_fp16.rknn`（约 7.7 MB），由本包 `convert_rknn_fp16.py`（PT → ONNX → RKNN float16）生成，转换链见下。
     ⚠️ **切勿用 int8（w8a8）量化**——会把分类头权重归零，导致所有锚框统一判为某一类、检测全空；若出现「NPU 有输出但全是同一类」即为此故，请改用 float16。

   - **性能（Panther X2 / RK3566，RKNPU \~0.8 TOPS）**：CPU（通用 aarch64 torch）约 **7190 ms/帧（0.1 FPS）**，
     NPU float16 约 **591 ms/帧（1.7 FPS）**，即 **约 12× 加速**，且检测结果与 CPU / ONNX 完全一致（同一图 5 个目标、置信度一致）。

   - **解码要点**：YOLOv8n 的 RKNN 输出 `(1,84,8400)` 为 **box\_first** 布局（`out[:4]`=框 xywh，`out[4:]`=80 类分数），
     且类分数**已是 sigmoid 后概率（0\~1）**，无需再 sigmoid；直接按阈值 + NMS 解码即可（脚本里 `rknnlite.inference` 已自动反量化）。

   - **运行时热切换**：`POST /update_backend`（form `backend=cpu|npu`）可在不重启进程的情况下切换后端。

   - **转换脚本**：`python3 convert_rknn_fp16.py`（依赖 `rknn-toolkit2` 与 `ultralytics`；ONNX 用 `ultralytics export` 生成，imgsz=640、opset=13、simplify=True）。

   - **前端控制**：Web 控制台「检测后端」面板可一键切换 CPU / NPU（调用 `POST /update_backend`，状态随 `/status` 同步）；「优化推荐参数」面板按当前后端给出推荐值（CPU 降频省算力 / NPU 每帧检测）并可一键应用（数据来自 `GET /recommended_params`）。

***

## 8. 2026-08-30 修复记录（对应审查报告）

- **P0-1 拉流重试**：取消 100 次上限，改为无限重试 + 指数退避（2s→60s 封顶），摄像头中途恢复即可自动续拍；此前重试耗尽后采集线程会永久死亡。

- **P0-2 DEBUG\_MODE 默认 False**：非鸟类别不再入库；单元文件显式 Environment=DEBUG\_MODE=false 兜底。

- **P1-1 日志脱敏**：连接失败仅打印掩码地址（`rtsp://user:***@ip`），不再泄露密码到 journald。

- **P1-2 参数持久化**：Web 修改的参数自动写入 config.json（原子写），重启/断电自动恢复，不再静默回退默认值。

- **P1-3 优雅停机**：lifespan shutdown 置位 stop\_event，MJPEG/采集线程随之退出；TimeoutStopSec=15；带流重启从 10s+SIGKILL 降为秒级。

- **P1-4 接口加固**：/api/photos 的 date 仅接受 YYYYMMDD（堵路径穿越），limit 上限 200。

- **P1-5 拉流架构**：独立 FrameReader 线程只保留最新帧，推理不再阻塞读流，延迟不再累积；RTSP 强制 TCP + 读超时；RTSP\_STREAM 可切子码流。

- **P2-1 去重语义统一**：移除永不过期的 confirmed\_hashes，同位置冷却统一由 SAVE\_COOLDOWN 控制（默认 2s）。

- **P2-2 healthz 分级**：ERROR 状态返回 503，不再假活。

- **P2-3 时段校验**：开始小时必须早于结束小时（暂不支持跨夜）。

- **P3**：NMS 阈值 CPU/NPU 统一、冷却字典自动清理、IP 格式校验、日期高亮保留、关闭 access\_log 等。

## 9. V5 前端升级（2026-08-30）

- 版本号 v4 → **v5**（页面标题 / logo / FastAPI title / README 同步）。

- 摄像头设置：`IP地址 [输入框] [切换]` 合并为紧凑一行。

- 检测后端：删除按钮下方"推理后端：…"状态文字（与悬停提示重复），当前后端由激活按钮心跳/脉冲高亮表达。

- 悬停提示防遮挡：CPU/NPU 按钮的提示气泡改为向侧栏内侧对齐（CPU 左对齐、NPU 右对齐），不再溢出被裁。

- 优化推荐参数：删除说明行；置信度/间隔/确认三项并排一行；加载失败提示改显示在应用按钮上。

- script.js 缓存破坏 `?v=4`。

***

## 10. 硬件加速状态（2026-09-06 修订：MPP/RGA 已回退）

- **Tier 0 保持生效**：MJPEG 预览限帧 10fps + q50（预览体积减半）。

- **Tier A（RGA letterbox）与 Tier B（MPP 硬解）已于 2026-09-06 回退**：假死专项排查（§11）通过二分实验确认
  二者的内核路径与"内核页表损坏 → 整机死亡"强相关——MPP+RGA 配置下带载 1.2~4.5 分钟内必死；
  回退为 `RTSP_BACKEND=cv2`（软解拉流）+ `RGA_ENABLED=0` 后带载稳定运行 1 小时+。
  **NPU 检测保留**（12× 加速不受影响）。如需重测硬件路径：`RGA_ENABLED=1 RTSP_BACKEND=mpp`
  并 `systemctl daemon-reload && systemctl restart birding`（建议先读 §11 的取证方法）。

- **journald 修复（保持）**：SystemMaxUse=100M / SystemKeepFree=200M（/etc/systemd/journald.conf.d/birding-fix.conf）。

***

## 11. 2026-09-06 假死专项排查与修复记录

**现象**：板子（Panther X2 / RK3566）反复"假死"——典型模式为第 1 天正常、第 2 天发现无画面无照片；
当天实测更恶化为开机 1~30 分钟内必死。

**根因（三层，全部实锤）**：

1. **主因：MPP/RGA 内核路径缺陷**。死亡瞬间崩溃监视器抓到内核页表损坏转储
   （`page dumped because: still mapped when deleted` + `el0_ia` 用户态指令中止），且二分实验定位：
   服务停用→稳定；MPP+RGA 开→必死；cv2+NPU→稳定 1 小时+。应用层代码不可能写坏内核页表。
2. **叠加：摄像头 DHCP 换址**。摄像头由 192.168.2.222 迁至 192.168.2.162，服务指向旧地址导致
   "进程活着但永远离线"的业务假死。已改 IP 并新增 `CAMERA_IP_FALLBACKS` 候选自动轮换。
3. **关联：双网卡同网段双活**（eth0 + wlan1 各拿 DHCP）对低端路由器形成 ARP 双口应答/MAC 漂移与
   DHCP 风暴压力（用户曾多次需重启路由器恢复网络）。已在崩溃监视器加网关探针，今后断网可直接归因
   （`gw=DEAD` 且板子心跳连续 → 路由器死；心跳中断 → 板子死）。**建议只保留一个网络接口**。

**已落地的修复**（代码提交 `48f377e` / `ff1c281`）：

- 采集监护线程：采集线程意外退出自动重启（退避），杜绝静默假死；
- `/healthz` 分级：ERROR/DEAD、线程死亡、长时间 OFFLINE 返回 503（不再假活）；
- `/status` 新增 `frame_age_sec` / `last_error` / `camera_candidates`；
- RTSP 超时参数 FFmpeg 6.1+ 兼容（stimeout+timeout 双设）；torch 线程上限（cpu-2）；
- mppdec 日志改每连接截断；摄像头候选 IP 自动轮换。

**设备侧防护体系**（不在仓库）：

- 硬件看门狗 `RuntimeWatchdogSec=10`：内核挂死 ~15 秒自动重启（实测死亡→恢复仅 34 秒，全程无人值守）；
- `birding-health.timer`：每 5 分钟探活 /healthz，无响应自动重启服务（对付"用户态冻死"形态）；
- `crash-watch.service`：每 5 秒向 `/root/crash_watch.log`（eMMC）落盘 uptime/温度/内存/负载/网关探针，
  充当"黑匣子"——每次死亡都有完整曲线可查。

**遗留建议**：路由器后台绑定静态租约（板子/相机 MAC↔IP）；板子只保留一个网络接口；
无 RTC 电池，务必保证 NTP 可用（否则照片归档日期与日志时间会错乱）；
如需进一步定位 MPP 与 RGA 哪个是元凶，可单独重开 MPP（保持 RGA 关闭）做对照。

