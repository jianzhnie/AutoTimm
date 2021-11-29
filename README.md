# AutoTimm

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
python benchmark.py \
    --data_path /media/robin/DATA/datatsets/image_data/dog-breed-identification \
    --output_path /home/robin/jianzh/automl/autodl/benchmark \
    --dataset dog-breed-identification \
    --train_framework autogluon
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
