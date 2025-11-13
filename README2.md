

Doc文档：`https://docs.google.com/document/d/1H8y2cItitBfidH8BwDBNAL4e1RJpIFDKc4HgqvQaMic/edit?tab=t.9ud2au53lm5g`

Git repo: `https://github.com/open-mmlab/mmdetection`

# 文件结构说明

kagglehub下载的数据是直接的yolov5 v8的数据格式，但是 MMD(即MMDetection) 不直接支持此格式。

故而有写`yolo2coco.py`脚本，进行格式转换，从yolo的`data.yaml`变为coco的三个json。Coco格式的json包含文件路径，但是yolo的label坐标数据会被coco写到json中，这是yolo格式和coco格式差别。

我的脚本直接讲coco的三个json文件写到目录`/root/.cache/kagglehub/datasets/rupankarmajumdar/crop-pests-dataset/versions/2/`之下，舍弃了`annotations`之类的目录名包装。

# 环境准备

Python should be **3.10.x**, there is something wrong if **3.11.x** or **3.12.x**.

```bash
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

```

然后手动部分：

```bash
cd /workspace/
# 【到此不要复制！需要手动操作】
# 手动复制文件进去
# download.py 和 yolo2coco.py 复制到 /workspace/
# faster_rcnn_crop_pest.py 复制到 /workspace/mmdetection/configs/crop_pest/

# 下载 kagglehub 数据集
# 数据集默认路径：
# /root/.cache/kagglehub/datasets/rupankarmajumdar/crop-pests-dataset/versions/2/
python /workspace/mmdetection/download.py
```

TODO: 进行数据清洗

```bash
cd /workspace/
git clone https://github.com/SenRanja/Aug.git
pip install -U pip setuptools wheel
pip install albumentations kagglehub
...
```


```
# 进行yolo2coco转换
python /workspace/mmdetection/yolo2coco.py
```

训练

    python tools/train.py configs/crop_pest/faster_rcnn_crop_pest.py > train.log 2>&1

验证与测试

因为python新版本针对反序列化的安全机制，和这个库本身使用的这种办法，导致生成图片预测打标比对的功能无法实现，此处使用我的办法可以运行这个模型。

1. 先修改如下的py代码，替换这个`load_from_local`函数

    vi /venv/main/lib/python3.10/site-packages/mmengine/runner/checkpoint.py

```python
def load_from_local(filename: str, map_location: str = 'cpu') -> dict:
    import torch
    import mmengine.logging

    # ✅ 信任 mmengine 的日志对象，避免 UnpicklingError
    torch.serialization.add_safe_globals([mmengine.logging.history_buffer.HistoryBuffer])

    # ✅ 关键：允许完整反序列化（PyTorch 2.6+ 默认是 True）
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)

    return checkpoint
```

然后自己运行这个命令就成功了，会在`/workspace/output1_show/`生成比对的图像打标

    python tools/test.py \
    configs/crop_pest/faster_rcnn_crop_pest.py \
    /workspace/output/epoch_46.pth \
    --show-dir /workspace/output2_show/ > eval_50.log 2>&1



