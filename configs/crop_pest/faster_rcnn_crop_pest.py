# ----------------------------------------------------------
# 基于 Faster R-CNN 的虫害检测 (COCO 格式)
# ----------------------------------------------------------

path = r"/root/.cache/kagglehub/datasets/rupankarmajumdar/crop-pests-dataset/versions/2/"

_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 数据集类型和路径
dataset_type = 'CocoDataset'

# data_root = os.path.join(path, 'coco')
data_root = path

classes = (
    "Ants",
    "Bees",
    "Beetles",
    "Caterpillars",
    "Earthworms",
    "Earwigs",
    "Grasshoppers",
    "Moths",
    "Slugs",
    "Snails",
    "Wasps",
    "Weevils",
)

# 数据加载
train_dataloader = dict(
    batch_size=18,                # 可根据显存调整
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='instances_train.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=classes)
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=18,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='instances_val.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=classes)
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=18,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='instances_test.json',
        data_prefix=dict(img=''),
        metainfo=dict(classes=classes)
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# 验证指标
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'instances_val.json',
    metric='bbox'
)

test_evaluator = val_evaluator

# 模型类别数要匹配你的数据集（12类）
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes))
    )
)

# 训练设置
train_cfg = dict(max_epochs=150, val_interval=1)

# 输出目录
work_dir = '/workspace/output/'
