#!/bin/bash
set -e

echo "===== Birding Service AutoStart Installer ====="

# -----------------------------
# 基本路径配置
# -----------------------------
APP_DIR="/home/birding"
VENV_DIR="$APP_DIR/venv"
SERVICE_FILE="/etc/systemd/system/birding.service"
IMAGE_DIR="/home/bird_image"
PY_FILE="birding_service.py"

# -----------------------------
# 1. 检查运行用户
# -----------------------------
if [ "$EUID" -ne 0 ]; then
  echo "❌ 请使用 root 用户运行该脚本"
  exit 1
fi

# -----------------------------
# 2. 检查主程序是否存在
# -----------------------------
if [ ! -f "$APP_DIR/$PY_FILE" ]; then
  echo "❌ 未找到 $APP_DIR/$PY_FILE"
  exit 1
fi

# -----------------------------
# 3. 创建图片目录
# -----------------------------
echo "📁 创建图片目录 $IMAGE_DIR"
mkdir -p "$IMAGE_DIR"
chmod 755 "$IMAGE_DIR"

# -----------------------------
# 4. 创建 Python 虚拟环境
# -----------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "🐍 创建 Python venv"
  python3 -m venv "$VENV_DIR"
else
  echo "🐍 venv 已存在，跳过"
fi

# -----------------------------
# 5. 安装依赖
# -----------------------------
echo "📦 安装 Python 依赖"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

pip install \
  fastapi \
  uvicorn \
  jinja2 \
  opencv-python-headless \
  ultralytics \
  numpy \
  pillow \
  python-multipart

deactivate

# -----------------------------
# 6. 写入 systemd 服务
# -----------------------------
echo "⚙️ 写入 systemd 服务"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Birding Service (YOLO RTSP)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
ExecStart=$VENV_DIR/bin/python3 $APP_DIR/$PY_FILE
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# -----------------------------
# 7. 启动并设置开机自启
# -----------------------------
echo "🔄 重载 systemd"
systemctl daemon-reload

echo "🚀 启动 birding.service"
systemctl start birding.service

echo "📌 设置开机自启"
systemctl enable birding.service

# -----------------------------
# 8. 状态输出
# -----------------------------
echo "✅ 当前服务状态："
systemctl status birding.service --no-pager

echo "🎉 安装完成！"
echo "👉 Web 界面访问：http://<设备IP>:8000"
echo "👉 查看日志：journalctl -u birding.service -f"
