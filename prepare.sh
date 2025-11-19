#!/bin/bash

# Install daemon process
apt install screen

cd /workspace/

# Install dependencies
pip install -U pip setuptools wheel
pip install kagglehub seaborn
pip install torch torchvision torchaudio
pip install -U openmim
mim install mmengine
mim install mmdet
git clone https://github.com/SenRanja/mmdetection.git

# Install mmcv
# This process is a bit time-consuming because I kept encountering errors during my research. Please strictly follow the bash instructions here.
cd /workspace/
pip uninstall -y mmcv mmcv-full
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0   # This version is most compatible with the latest version of mmdetection
MMCV_WITH_OPS=1 FORCE_CUDA=1 python setup.py build_ext --inplace
pip install -e .
python -c "import mmcv; print(mmcv.__version__); import mmcv.ops; print('✅ mmcv._ext loaded successfully')"

cd /workspace/
git clone https://github.com/SenRanja/Aug.git
pip install -U pip setuptools wheel
pip install albumentations kagglehub

# Replace
NEW_PATH=$(python /workspace/Aug/download.py | tail -n 1)

# Data Cleaning
cd /workspace/Aug/
python /workspace/Aug/main.py "$NEW_PATH"
cd /workspace/


# Perform yolo2coco conversion
python /workspace/mmdetection/yolo2coco.py



FILE="/venv/main/lib/python3.10/site-packages/mmengine/runner/checkpoint.py"

# Backup
cp "$FILE" "$FILE.bak"

# Remove old function content (including decorators)
sed -i "/@CheckpointLoader.register_scheme(prefixes='')/,/return checkpoint/d" "$FILE"

# Add new functions (including decorators)
cat << 'EOF' >> "$FILE"

@CheckpointLoader.register_scheme(prefixes='')
def load_from_local(filename: str, map_location: str = 'cpu') -> dict:
    import torch
    import mmengine.logging

    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f"{filename} can not be found.")

    # Trust mmengine's log objects to avoid UnpicklingError
    torch.serialization.add_safe_globals([mmengine.logging.history_buffer.HistoryBuffer])

    # Allow full deserialization
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)

    return checkpoint

EOF

echo "✔ Replacement complete: $FILE"


