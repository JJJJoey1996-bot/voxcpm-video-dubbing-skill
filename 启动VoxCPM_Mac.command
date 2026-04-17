#!/bin/bash

# VoxCPM2 Mac One-Click Starter
# This script will automatically:
# 1. Set the working directory
# 2. Check and activate the virtual environment
# 3. Launch the WebUI with Mac-optimized settings

# Get the script directory and cd into it
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Clear terminal screen for a clean start
clear

echo "=========================================================="
echo "          VoxCPM2 Mac Studio - 一键启动脚本"
echo "=========================================================="
echo " 正在初始化环境..."

# Check if .venv exists, if not, try to run setup
if [ ! -f ".venv/bin/activate" ]; then
    echo ""
    echo "[!] 未发现虚拟环境 (.venv)。"
    echo "[?] 是否现在运行 scripts/setup_mac.sh 进行环境初始化？(y/n)"
    read -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "[*] 正在启动初始化脚本..."
        chmod +x scripts/setup_mac.sh
        ./scripts/setup_mac.sh
        if [ $? -ne 0 ]; then
            echo "[ERROR] 环境初始化失败，请检查上方报错。"
            read -p "按回车键退出..."
            exit 1
        fi
    else
        echo "[!] 操作已取消。请手动运行 scripts/setup_mac.sh 后再启动。"
        read -p "按回车键退出..."
        exit 1
    fi
fi

# Activate virtual environment
source .venv/bin/activate

# Optional: Sync dependencies only when explicitly requested
if [ "${VOXCPM_SYNC_ON_START:-0}" = "1" ] && command -v uv >/dev/null 2>&1; then
    echo "[*] 检测到 uv，正在同步依赖项..."
    uv sync > /dev/null 2>&1
fi

echo "[*] 环境激活成功。"
echo "[*] 正在启动 WebUI (端口: 8808)..."
echo "[*] Apple Silicon 优化: 已启用 (PyTorch + MPS)"
echo "[*] 若本地缺少模型，将自动从 Hugging Face 或 ModelScope 下载。"
echo "[*] 默认优先使用本地 ./models/VoxCPM2，以获得更快的重复启动速度。"
echo "[*] 启动前会做一次 MPS 自检；若当前环境拿不到 MPS，将自动提示并回退。"
echo "----------------------------------------------------------"
echo " 访问地址: http://127.0.0.1:8808"
echo " 启动后请勿关闭此窗口。"
echo "----------------------------------------------------------"

./scripts/run_webui_mac.sh

# Handle exit
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] 程序异常退出。"
else
    echo ""
    echo "[INFO] 服务已停止。"
fi

echo "=========================================================="
read -n 1 -s -r -p "按任意键退出窗口..."
echo ""
