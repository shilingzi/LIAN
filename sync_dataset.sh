#!/bin/bash

# 使用方法说明
usage() {
    echo "使用方法:"
    echo "  $0 -h <服务器地址> [-u <用户名>] [-p <端口号>] [-d <本地数据集路径>]"
    echo ""
    echo "参数说明:"
    echo "  -h: 服务器地址（必需）"
    echo "  -u: SSH用户名（可选，默认为当前用户）"
    echo "  -p: SSH端口号（可选，默认为22）"
    echo "  -d: 本地数据集路径（可选，默认为./data/datasets/DIV2K）"
    echo ""
    echo "示例:"
    echo "  $0 -h example.com -u username -p 22 -d /path/to/dataset"
    exit 1
}

# 默认值
PORT=22
USERNAME=$(whoami)
DATASET_PATH="./data/datasets/DIV2K"

# 解析命令行参数
while getopts "h:u:p:d:" opt; do
    case $opt in
        h) HOST="$OPTARG";;
        u) USERNAME="$OPTARG";;
        p) PORT="$OPTARG";;
        d) DATASET_PATH="$OPTARG";;
        ?) usage;;
    esac
done

# 检查必需参数
if [ -z "$HOST" ]; then
    echo "错误: 必须指定服务器地址"
    usage
fi

# 确保本地数据集目录存在
if [ ! -d "$DATASET_PATH" ]; then
    echo "错误: 本地数据集目录不存在: $DATASET_PATH"
    exit 1
fi

# 远程服务器上的目标路径
REMOTE_PATH="~/LIAN/data/datasets/DIV2K"

# 创建远程目录
echo "创建远程目录..."
ssh -p $PORT $USERNAME@$HOST "mkdir -p $REMOTE_PATH/{train/HR,val/HR}"

# 同步训练集
echo "同步训练集..."
rsync -avz --progress -e "ssh -p $PORT" \
    "$DATASET_PATH/train/HR/" \
    "$USERNAME@$HOST:$REMOTE_PATH/train/HR/"

# 同步验证集
echo "同步验证集..."
rsync -avz --progress -e "ssh -p $PORT" \
    "$DATASET_PATH/val/HR/" \
    "$USERNAME@$HOST:$REMOTE_PATH/val/HR/"

# 检查远程文件数量
echo "检查文件传输结果..."
echo "训练集文件数量:"
ssh -p $PORT $USERNAME@$HOST "ls -1 $REMOTE_PATH/train/HR/ | wc -l"
echo "验证集文件数量:"
ssh -p $PORT $USERNAME@$HOST "ls -1 $REMOTE_PATH/val/HR/ | wc -l"

echo "数据集同步完成！" 