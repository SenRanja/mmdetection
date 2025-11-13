#!/bin/bash

# 安装daemon进程
apt install screen

cd /workspace/

# 安装依赖
pip install -U pip setuptools wheel
pip install kagglehub
pip install torch torchvision torchaudio
pip install -U openmim
mim install mmengine
mim install mmdet
git clone https://github.com/SenRanja/mmdetection.git

# 安装mmcv
# 此过程有点费时间，因为我调研过程中此步骤一直出错，严格按照此处bash执行
cd /workspace/
pip uninstall -y mmcv mmcv-full
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0   # 这个版本与 mmdetection 最新版最兼容
MMCV_WITH_OPS=1 FORCE_CUDA=1 python setup.py build_ext --inplace
pip install -e .
python -c "import mmcv; print(mmcv.__version__); import mmcv.ops; print('✅ mmcv._ext loaded successfully')"

cd /workspace/
git clone https://github.com/SenRanja/Aug.git
pip install -U pip setuptools wheel
pip install albumentations kagglehub

# 替换
NEW_PATH=$(python /workspace/Aug/download.py | tail -n 1)

# 数据清洗
cd /workspace/Aug/
python /workspace/Aug/main.py "$NEW_PATH"
cd /workspace/


# 进行yolo2coco转换
python /workspace/mmdetection/yolo2coco.py



FILE="/venv/main/lib/python3.10/site-packages/mmengine/runner/checkpoint.py"

# 备份
cp "$FILE" "$FILE.bak"

# 删除旧的函数内容（包含装饰器）
sed -i "/@CheckpointLoader.register_scheme(prefixes='')/,/return checkpoint/d" "$FILE"

# 追加新的函数（包含装饰器）
cat << 'EOF' >> "$FILE"

@CheckpointLoader.register_scheme(prefixes='')
def load_from_local(filename: str, map_location: str = 'cpu') -> dict:
    import torch
    import mmengine.logging

    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f"{filename} can not be found.")

    # 信任 mmengine 的日志对象，避免 UnpicklingError
    torch.serialization.add_safe_globals([mmengine.logging.history_buffer.HistoryBuffer])

    # 允许完整反序列化
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)

    return checkpoint

EOF

echo "✔ 替换完成：$FILE"


