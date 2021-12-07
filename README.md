<h1 align="center">
    <p>AutoTimm</p>
</h1>

<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/jianzhnie/AutoTimm/blob/main/README_zh.md">简体中文</a>
    <p>
</h4>

<h4 align="center">
    <p>State-of-the-art Automatic PyTorch image models</p>
</h4>

AutoTimm automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications.  With just a few lines of code, you can train and deploy high-accuracy machine learning and deep learning models on image.


##  Documents for  AutoTimm Benchmark

This tutorial demonstrates how to use AutoTimm with your own custom datasets.
As an example, we use a dataset from Kaggle to show the required steps to format image data properly for AutoTimm.


### Step 1: Organizing the dataset into proper directories

After completing this step, you will have the following directory structure on your machine:

```sh
   Your_Dataset/
    ├──train/
        ├── class1/
        ├── class2/
        ├── class3/
        ├── ...
    ├──test/
        ├── class1/
        ├── class2/
        ├── class3/
        ├── ...
```

Here `Your_Dataset` is a folder containing the raw images categorized into classes. For example, subfolder `class1` contains all images that belong to the first class, `class2` contains all images belonging to the second class, etc.

We generally recommend at least 100 training images per class for reasonable classification performance, but this might depend on the type of images in your specific use-case.

Under each class, the following image formats are supported when training your model:

```sh
- JPG
- JPEG
- PNG
```

In the same dataset, all the images should be in the same format. Note that in image classification, we do not require that all images have the same resolution.

You will need to organize your dataset into the above directory structure before using AutoTimm.


#### For kaggle datasets

Sometimes dataset needs additional data preprocessing by script [data_processing](./autotimm/utils/data_processing.py).

```sh
  data
    ├──XXXX/images_all
    ├         ├── img1.jpg
    ├         ├── img2.jpg
    ├──XXXX/test
    ├         ├── ...

python data_processing.py --dataset <aerial\dog\> --data-dir data
```

Finally, we have the desired directory structure under `./data/XXXX/train/`, which in this case looks as follows:

```sh
  data
    ├──XXXX/train
    ├         ├── classA
    ├         ├── classb
    ├         ├── ...
    ├──XXXX/test
    ├         ├── ...
    ├
    ├
    ├──ZZZZ/train
    ├         ├── classA
    ├         ├── classb
    ├         ├── ...
    ├──ZZZZ/test
              ├── ...
```

#### For Paperwithcode datasets

#### TODO
```sh
python data_processing.py --dataset <aerial\dog\> --data-dir data
```

### Step 2: Split the original dataset into train_data and test_data

Sometimes dataset needs additional data_split by Script [data_split](./autotimm/utils/data_split.py).


```sh
dataset__name
    ├──train
        ├──split/train
        ├         ├── classA
        ├         ├── classb
        ├         ├── ...
        ├──split/test
        ├         ├── classA
        ├         ├── classb
        ├         ├── ...
    ├──test
        ├── img1.jpg
        ├── img2.jpg
        ├── ...
```

```sh
python data_split.py --data-dir /data/AutoML_compete/Store-type-recognition/
```


### Step 3: Use AutoTimm fit to generate a classification model

Now that we have a `Dataset` object, we can use AutoTimm's default configuration to obtain an image classification model using the [fit]() function.


#### AutoTimm Benchmark

```shell script
python -m torch.distributed.launch --nproc_per_node=1 train.py \
--data_name hymenoptera \
--data_path /media/robin/DATA/datatsets/image_data/hymenoptera/split/ \
--output-dir /media/robin/DATA/datatsets/image_data/hymenoptera \
--model resnet18 \
--epochs 10 \
--lr 0.01 \
--batch-size 16 \
--pretrained > output.txt 2>&1 &
```

## Step 4:  fit to generate a classification model

Bag of tricks are used on image classification dataset.

Customize parameter configuration according your data as follow:

```python
lr_config = ag.space.Dict(
            lr_mode='cosine',
            lr_decay=0.1,
            lr_decay_period=0,
            lr_decay_epoch='40,80',
            warmup_lr=0.0,
            warmup_epochs=5)

tricks = ag.space.Dict(
            last_gamma=True,
            use_pretrained=True,
            use_se=False,
            mixup=False,
            mixup_alpha=0.2,
            mixup_off_epoch=0,
            label_smoothing=True,
            no_wd=True,
            teacher_name=None,
            temperature=20.0,
            hard_weight=0.5,
            batch_norm=False,
            use_gn=False)
```
## Experiment Result
We test our framwork on 41 open source datasets and compared with ModelArts、EasyDL、AutoGluon and BiT. Our framwork shows the best performance on more than 80% datasets. Hyperparameter  tuning was used on AutoGluon、BiT and AutoTimm. The model、learning-rate and betchsize be chosen in the same framework might be different. Here are specific accuracy on each test dataset of 41 datasets.

 

|                   dataset                   | ModelArts |  EasyDL   |    BiT    | AutoGluon | AutoTimm  |
| :-----------------------------------------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|                 dog-vs-cat                  |   0.99    |   0.994   |   0.995   |   0.996   | **0.997** |
|                  sport-70                   |   0.934   |   0.967   |   0.961   |   0.976   | **0.988** |
|                Chinese MNIST                |   0.994   | **0.999** |   0.998   |   0.998   |   0.997   |
|               casting product               |   0.762   | **0.997** |   0.991   |   0.995   |   0.996   |
|         A-Large-Scale-Fish-Dataset          |   0.953   |   0.988   |     1     |     1     |   **1**   |
|             Flowers Recognition             |   0.943   |   0.956   |   0.964   |   0.947   | **0.968** |
|              Emotion Detection              |   0.605   |   0.706   |   0.671   |   0.705   | **0.721** |
|          Dog Breed Identification           |   0.758   |   0.76    |   0.808   |   0.924   | **0.958** |
|             Leaf Classification             |   0.662   |   0.611   | **0.95**  |   0.848   |   0.919   |
|       APTOS 2019 Blindness Detection        |   0.789   |   0.836   | **0.852** |   0.829   |   0.85    |
|            Cassava Leaf Disease             |   0.828   |   0.843   |   0.86    |   0.866   | **0.887** |
| The Nature Conservancy Fisheries Monitoring |   0.945   |   0.947   |   0.965   |   0.951   | **0.965** |
|       Plant Seedlings Classification        |   0.909   | **0.991** |   0.965   |   0.964   |   0.974   |
|              Oxford Flower-102              |   0.977   |   0.985   |   0.982   |   0.993   | **0.996** |
|         National Data Science Bowl          |   0.523   |   0.747   |   0.76    |   0.745   | **0.782** |
|                  Food-101                   |   0.828   |     /     |   0.859   |   0.889   | **0.928** |
|         Caltech-UCSD Birds-200-2011         |   0.674   |   0.802   |   0.844   |   0.857   |  **0.9**  |
|                  CINIC-10                   |   0.386   |     /     |   0.934   |   0.923   | **0.965** |
|                 Caltech-101                 |   0.883   |   0.933   | **0.968** |   0.958   |   0.963   |
|                     DTD                     |   0.661   |   0.668   |   0.753   |   0.715   | **0.781** |
|                FGVC-Aircraft                |   0.857   |   0.827   |   0.942   |   0.964   | **0.969** |
|             Weather-recognition             |   0.83    |   0.836   |   0.866   |   0.838   |   0.867   |
|           Store-type-recognition            |   0.624   |   0.702   |   0.697   |   0.686   | **0.729** |
|                    MURA                     |   0.786   |   0.817   |   0.808   |   0.833   | **0.834** |
|                   CIFAR10                   |   0.637   |   0.971   |   0.968   |   0.976   | **0.991** |
|                  CIFAR100                   |   0.142   |   0.844   |   0.83    |   0.848   | **0.928** |
|                UKCarsDataset                |   0.672   |   0.864   |   0.681   |   0.831   | **0.898** |
|           garbage classification            |   0.972   |   0.979   |   0.977   |   0.974   | **0.983** |
|                flying plane                 |   0.689   |   0.729   |   0.766   |   0.75    | **0.785** |
|                  Satellite                  |   0.986   |   0.996   |   0.997   |   0.993   | **0.998** |
|                    MAMe                     |   0.814   |   0.869   |   0.836   |   0.863   | **0.879** |
|                 Road damage                 |   0.966   |   0.973   |   0.969   |   0.969   | **0.978** |
|           Boat types recognition            |   0.875   |   0.92    |   0.92    |   0.944   | **0.958** |
|            Scene Classification             |   0.921   |   0.944   |   0.95    |   0.937   | **0.957** |
|                    coins                    |   0.725   |   0.83    |   0.901   |   0.909   | **0.919** |
|             Bald Classification             |   0.989   |     /     |   0.989   |   0.99    | **0.99**  |
|              Vietnamese Foods               |   0.854   |   0.913   |   0.891   |   0.923   | **0.936** |
|                  Yoga Pose                  |   0.53    |   0.673   | **0.75**  |   0.667   |   0.73    |
|                Green Finder                 |   0.989   |   0.994   |   0.999   |   0.998   | **0.999** |
|              MIT Indoor Scene               |   0.764   |   0.82    |   0.846   |   0.852   | **0.889** |
|        Google Scraped Image Dataset         |   0.835   |   0.788   |   0.802   | **0.926** |   0.918   |

