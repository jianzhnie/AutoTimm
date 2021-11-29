# AutoTimm 使用教程

## 1  数据集加载

我们将使用图像分类任务来说明如何使用 AutoDL API。 

 

第一步导入需要的包和函数

```python
import autotimm
from from autotimm.auto import ImagePredictor, TorchImageClassificationDataset
```

 

第二步：导入数据集

我们使用 Kaggle 的 Shopee-IET 数据集的一个子集。 该数据中的每幅图像都描述了一件服装，相应的标签指定了它的服装类别。 我们的数据子集包含以下可能的标签：BabyPants、BabyShirt、womencasualshoes、womenchiffontop。我们可以通过自动下载 url 数据来加载数据集：

（1）直接从文件夹读取

```python
train_dataset, _, test_dataset = TorchImageClassificationDataset.from_folders('https://autotimm.s3.amazonaws.com/datasets/shopee-iet.zip')
print(train_dataset)
```

（2）从 csv 文件读取

```python
csv_file = autotimm.utils.download('https://autogluon.s3-us-west-2.amazonaws.com/datasets/petfinder_example.csv')
image_data = TorchImageClassificationDataset.from_csv(csv_file)
```

 

下面是数据集列表

```python
data/
├── test/
└── train/
                                                 image  label
0    /var/lib/jenkins/.gluoncv/datasets/shopee-iet/...      0
1    /var/lib/jenkins/.gluoncv/datasets/shopee-iet/...      0
2    /var/lib/jenkins/.gluoncv/datasets/shopee-iet/...      0
3    /var/lib/jenkins/.gluoncv/datasets/shopee-iet/...      0
4    /var/lib/jenkins/.gluoncv/datasets/shopee-iet/...      0
..                                                 ...    ...
795  /var/lib/jenkins/.gluoncv/datasets/shopee-iet/...      3
796  /var/lib/jenkins/.gluoncv/datasets/shopee-iet/...      3
797  /var/lib/jenkins/.gluoncv/datasets/shopee-iet/...      3
798  /var/lib/jenkins/.gluoncv/datasets/shopee-iet/...      3
799  /var/lib/jenkins/.gluoncv/datasets/shopee-iet/...      3
 
[800 rows x 2 columns]
```

 

## 2  模型训练

使用 ImagePredictor() 自动进行模型训练和评估，下面是模型训练接口。

 
```python
predictor = ImagePredictor()
predictor.fit(train_dataset, hyperparameters={'epochs': 2}) 
```

 

下面是模型训练的输出：

```python
time_limit=auto set to time_limit=7200.
Reset labels to [0, 1, 2, 3]
Randomly split train_data into train[720]/validation[80] splits.
The number of requested GPUs is greater than the number of available GPUs.Reduce the number to 1
Starting fit without HPO
modified configs(<old> != <new>): {
root.img_cls.model   resnet101 != resnet50
root.train.early_stop_baseline 0.0 != -inf
root.train.early_stop_patience -1 != 10
root.train.epochs    200 != 2
root.train.early_stop_max_value 1.0 != inf
root.train.batch_size 32 != 16
root.misc.num_workers 4 != 8
root.misc.seed       42 != 762
}
Saved config to /var/lib/jenkins/workspace/workspace/autotimm-tutorial-image-classification-v3/docs/_build/eval/tutorials/image_prediction/c730a589/.trial_0/config.yaml
Model resnet50 created, param count:                                         23516228
AMP not enabled. Training in float32.
Disable EMA as it is not supported for now.
Start training from [Epoch 0]
[Epoch 0] training: accuracy=0.306944
[Epoch 0] speed: 91 samples/sec     time cost: 7.728239
[Epoch 0] validation: top1=0.425000 top5=1.000000
[Epoch 0] Current best top-1: 0.425000 vs previous -inf, saved to /var/lib/jenkins/workspace/workspace/autotimm-tutorial-image-classification-v3/docs/_build/eval/tutorials/image_prediction/c730a589/.trial_0/best_checkpoint.pkl
[Epoch 1] training: accuracy=0.523611
[Epoch 1] speed: 95 samples/sec     time cost: 7.392515
[Epoch 1] validation: top1=0.750000 top5=1.000000
[Epoch 1] Current best top-1: 0.750000 vs previous 0.425000, saved to /var/lib/jenkins/workspace/workspace/autotimm-tutorial-image-classification-v3/docs/_build/eval/tutorials/image_prediction/c730a589/.trial_0/best_checkpoint.pkl
Applying the state from the best checkpoint...
Finished, total runtime is 22.83 s
{ 'best_config': { 'batch_size': 16,
                   'dist_ip_addrs': None,
                   'early_stop_baseline': -inf,
                   'early_stop_max_value': inf,
                   'early_stop_patience': 10,
                   'epochs': 2,
                   'final_fit': False,
                   'gpus': [0],
                   'log_dir': '/var/lib/jenkins/workspace/workspace/autotimm-tutorial-image-classification-v3/docs/_build/eval/tutorials/image_prediction/c730a589',
                   'lr': 0.01,
                   'model': 'resnet50',
                   'ngpus_per_trial': 8,
                   'nthreads_per_trial': 128,
                   'num_trials': 1,
                   'num_workers': 8,
                   'problem_type': 'multiclass',
                   'scheduler': 'local',
                   'search_strategy': 'random',
                   'searcher': 'random',
                   'seed': 762,
                   'time_limits': 7200,
                   'wall_clock_tick': 1630452470.4286814},
  'total_time': 16.54758381843567,
  'train_acc': 0.5236111111111111,
  'valid_acc': 0.75}
```

 

## 3  模型测试

在完成模型训练之后，在测试集上对模型进行评估。在 fit 中，数据集会自动拆分为训练集和验证集。 根据其在验证集上的性能选择具有最佳超参数配置的模型。 最好的模型最终使用最佳配置在我们的整个数据集上重新训练（即合并训练+验证）。

在验证集上获得的最佳 Top-1 准确率如下：

```python
test_acc = predictor.evaluate(test_dataset)
print('Top-1 test acc: %.3f' % test_acc['top1'])
```

```
Top-1 test acc: 0.688
```

### 3.1 单张图片预测

给出一个示例图像，我们可以很容易地使用最终模型预测标签：

```python
image_path = test_dataset.iloc[0]['image']
result = predictor.predict(image_path)
print(result)
0    1
Name: label, dtype: int64
```

如果需要所有类别的概率，可以调用predict_Probability：

```python
proba = predictor.predict_proba(image_path)
print(proba)
          0         1         2         3
0  0.230975  0.350888  0.260955  0.157182
```

### 3.2 批量预测

还可以同时输入多个图像，得到预测结果的列表：

```python
bulk_result = predictor.predict(test_dataset)
print(bulk_result)
0     1
1     1
2     2
3     1
4     1
     ..
75    3
76    3
77    2
78    2
79    3
Name: label, Length: 80, dtype: int64
```

### 3.3 提取图像特征

从模型学习提取整个图像表示，我们提供predict_特征函数，允许predictor返回N维图像特征，其中N取决于模型（通常为512到2048长度向量）

```python
image_path = test_dataset.iloc[0]['image']
feature = predictor.predict_feature(image_path)
print(feature)
```

image_feature 0 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...

## 4  保存和加载模型

模型训练完成可以很方便的将模型保存下来方便后续使用：

```python
filename = 'predictor.pkl'
predictor.save(filename)
predictor_loaded = ImagePredictor.load(filename)
# use predictor_loaded as usual
result = predictor_loaded.predict(image_path)
print(result)
0    1
Name: label, dtype: int64
```