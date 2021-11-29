"""Dataset implementation for specific task(s)"""
# pylint: disable=consider-using-generator
import logging
import os
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from autotimm.auto.utils.auto_data import is_url, url_data

try:
    import torch
    TorchDataset = torch.utils.data.Dataset
except ImportError:
    TorchDataset = object
    torch = None

logger = logging.getLogger()

__all__ = ['TorchImageClassificationDataset']


def _absolute_pathify(df, root=None, column='image'):
    """Convert relative paths to absolute."""
    if root is None:
        return df
    assert column in df.columns
    assert isinstance(root, str), 'Invalid root path: {}'.format(root)
    root = os.path.abspath(os.path.expanduser(root))
    for i, _ in df.iterrows():
        path = df.at[i, 'image']
        if not os.path.isabs(path):
            df.at[i, 'image'] = os.path.join(root, os.path.expanduser(path))
    return df


class TorchImageClassificationDataset(pd.DataFrame):
    """ImageClassification dataset as DataFrame.

    Parameters
    ----------
    data : the input for pd.DataFrame
        The input data.
    classes : list of str, optional
        The class synsets for this dataset, if `None`, it will infer from the data.
    image_column : str, default is 'image'
        The name of the column for image paths.
    label_column : str, default is 'label'
        The name for the label column, leave it as is if no label column is available. Note that
        in such case you won't be able to train with this dataset, but can still visualize the images.
    """
    # preserved properties that will be copied to a new instance
    _metadata = ['classes', 'IMG_COL', 'LABEL_COL']

    def __init__(self,
                 data,
                 classes=None,
                 image_column='image',
                 label_column='label',
                 **kwargs):
        root = kwargs.pop('root', None)
        no_class = kwargs.pop('no_class', False)
        if isinstance(data, str) and data.endswith('csv'):
            data = self.from_csv(
                data,
                root=root,
                image_column=image_column,
                label_column=label_column,
                no_class=no_class)
        if no_class:
            self.classes = []
        else:
            self.classes = classes
        self.IMG_COL = image_column
        self.LABEL_COL = label_column
        super().__init__(data, **kwargs)

    @property
    def _constructor(self):
        return TorchImageClassificationDataset

    @property
    def _constructor_sliced(self):
        return pd.Series

    def random_split(self, test_size=0.1, val_size=0, random_state=None):
        r"""Randomly split the dataset into train/val/test sets.
        Note that it's perfectly fine to set `test_size` or `val_size` to 0, where the
        returned splits will be empty dataframes.

        Parameters
        ----------
        test_size : float
            The ratio for test set, can be in range [0, 1].
        val_size : float
            The ratio for validation set, can be in range [0, 1].
        random_state : int, optional
            If not `None`, will set the random state of numpy.random engine.

        Returns
        -------
        train, val, test - (DataFrame, DataFrame, DataFrame)
            The returned dataframes for train/val/test

        """
        assert 0 <= test_size < 1.0
        assert 0 <= val_size < 1.0
        assert (val_size +
                test_size) < 1.0, 'val_size + test_size must less than 1.0!'
        if random_state:
            np.random.seed(random_state)
        test_mask = np.random.rand(len(self)) < test_size
        test = self[test_mask]
        trainval = self[~test_mask]
        val_mask = np.random.rand(len(trainval)) < val_size
        val = trainval[val_mask]
        train = trainval[~val_mask]
        return train, val, test

    def show_images(self,
                    indices=None,
                    nsample=16,
                    ncol=4,
                    shuffle=True,
                    resize=224,
                    fontsize=20):
        r"""Display images in dataset.

        Parameters
        ----------
        indices : iterable of int, optional
            The image indices to be displayed, if `None`, will generate `nsample` indices.
            If `shuffle` == `True`(default), the indices are random numbers.
        nsample : int, optional
            The number of samples to be displayed.
        ncol : int, optional
            The column size of ploted image matrix.
        shuffle : bool, optional
            If `shuffle` is False, will always sample from the begining.
        resize : int, optional
            The image will be resized to (resize, resize) for better visual experience.
        fontsize : int, optional
            The fontsize for the title
        """
        df = self.reset_index(drop=True)
        if indices is None:
            if not shuffle:
                indices = range(nsample)
            else:
                indices = list(range(len(df)))
                np.random.shuffle(indices)
                indices = indices[:min(nsample, len(indices))]
        images = [cv2.cvtColor(cv2.resize(cv2.imread(df.at[idx, df.IMG_COL]), (resize, resize), \
            interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for idx in indices if idx < len(df)]
        titles = None
        if df.LABEL_COL in df.columns:
            if df.classes:
                titles = [df.classes[int(df.at[idx, df.LABEL_COL])] + ': ' + str(df.at[idx, df.LABEL_COL]) \
                    for idx in indices if idx < len(df)]
            else:
                titles = [
                    str(df.at[idx, df.LABEL_COL]) for idx in indices
                    if idx < len(df)
                ]
        _show_images(images, cols=ncol, titles=titles, fontsize=fontsize)

    def to_pytorch(self, transform):
        """Return a pytorch based iterator that returns ndarray and labels."""
        df = self.rename(
            columns={
                self.IMG_COL: 'image',
                self.LABEL_COL: 'label'
            },
            errors='ignore')
        df = df.reset_index(drop=True)
        return _TorchImageClassificationDataset(df, transform=transform)

    @classmethod
    def from_csv(cls,
                 csv_file,
                 root=None,
                 image_column='image',
                 label_column='label',
                 no_class=False):
        r"""Create from csv file.

        Parameters
        ----------
        csv_file : str
            The path for csv file.
        root : str
            The relative root for image paths stored in csv file.
        image_column : str, default is 'image'
            The name of the column for image paths.
        label_column : str, default is 'label'
            The name for the label column, leave it as is if no label column is available. Note that
            in such case you won't be able to train with this dataset, but can still visualize the images.
        """
        if is_url(csv_file):
            csv_file = url_data(csv_file, disp_depth=0)
        df = pd.read_csv(csv_file)
        assert image_column in df.columns, f'`{image_column}` column is required, used for accessing the original images'
        if not label_column in df.columns:
            logger.info('label not in columns, no access to labels of images')
            classes = None
        else:
            classes = df[label_column].unique().tolist()
        df = _absolute_pathify(df, root=root, column=image_column)
        return cls(
            df,
            classes=classes,
            image_column=image_column,
            label_column=label_column,
            no_class=no_class)

    @classmethod
    def from_folder(cls, root, exts=('.jpg', '.jpeg', '.png'), no_class=False):
        r"""A dataset for loading image files stored in a folder structure.
        like::
            root/car/0001.jpg
            root/car/xxxa.jpg
            root/car/yyyb.jpg
            root/bus/123.png
            root/bus/023.jpg
            root/bus/wwww.jpg

        Parameters
        -----------
        root : str or pathlib.Path
            The root folder
        exts : iterable of str
            The image file extensions
        """
        if is_url(root):
            root = url_data(root)
        synsets = []
        items = {'image': [], 'label': []}
        if isinstance(root, Path):
            assert root.exists(), '{} not exist'.format(str(root))
            root = str(root.resolve())
        assert isinstance(root, str)
        root = os.path.abspath(os.path.expanduser(root))

        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                logger.debug(
                    'Ignoring %s, which is not a directory.',
                    path,
                    stacklevel=3)
                continue
            label = len(synsets)
            synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in exts:
                    logger.debug('Ignoring %s of type %s. Only support %s',
                                 filename, ext, ', '.join(exts))
                    continue
                items['image'].append(filename)
                items['label'].append(label)
        return cls(items, classes=synsets, no_class=no_class)

    @classmethod
    def from_folders(cls,
                     root,
                     train='train',
                     val='val',
                     test='test',
                     exts=('.jpg', '.jpeg', '.png'),
                     no_class=False):
        """Method for loading (already) splited datasets under root.
        like::
            root/train/car/0001.jpg
            root/train/car/xxxa.jpg
            root/train/car/yyyb.jpg
            root/val/bus/123.png
            root/test/bus/023.jpg
            root/test/bus/wwww.jpg
        will be loaded into three splits, with 3/1/2 images, respectively.
        You can specify the sub-folder names of `train`/`val`/`test` individually. If one particular sub-folder is not
        found, the corresponding returned dataset will be `None`.
        Note: if your existing dataset isn't split into such format, please use `from_folder` function and apply
        random splitting using `random_split` function afterwards.

        Example:
        >>> train_data, val_data, test_data = ImageClassificationDataset.from_folders('./data', val='validation')
        >> assert len(train_data) == 3


        Parameters
        ----------
        root : str or pathlib.Path or url
            The root dir for the entire dataset, if url is provided, the data will be downloaded and extracted.
        train : str
            The sub-folder name for training images.
        val : str
            The sub-folder name for training images.
        test : str
            The sub-folder name for training images.
        exts : iterable of str
            The supported image extensions when searching for sub-sub-directories.

        Returns
        -------
        (train_data, val_data, test_data) of type tuple(ImageClassificationDataset, )
            splited datasets, can be `None` if no sub-directory found.

        """
        if is_url(root):
            root = url_data(root)
        if isinstance(root, Path):
            assert root.exists(), '{} not exist'.format(str(root))
            root = str(root.resolve())
        assert isinstance(root, str)
        root = os.path.abspath(os.path.expanduser(root))
        train_root = os.path.join(root, train)
        val_root = os.path.join(root, val)
        test_root = os.path.join(root, test)
        empty = cls({'image': [], 'label': []})
        train_data, val_data, test_data = empty, empty, empty
        # train
        if os.path.isdir(train_root):
            train_data = cls.from_folder(
                train_root, exts=exts, no_class=no_class)
        else:
            raise ValueError('Train split does not exist: {}'.format(train))
        # val
        if os.path.isdir(val_root):
            val_data = cls.from_folder(val_root, exts=exts, no_class=no_class)
        # test
        if os.path.isdir(test_root):
            test_data = cls.from_folder(
                test_root, exts=exts, no_class=no_class)

        # check synsets, val/test synsets can be subsets(order matters!) or exact matches of train synset
        if len(val_data) and not _check_synsets(train_data.classes,
                                                val_data.classes):
            warnings.warn('Train/val synsets does not match: {} vs {}'.format(
                train_data.classes, val_data.classes))
        if len(test_data) and not _check_synsets(train_data.classes,
                                                 test_data.classes):
            warnings.warn('Train/val synsets does not match: {} vs {}'.format(
                train_data.classes, test_data.classes))

        return train_data, val_data, test_data

    @classmethod
    def from_name_func(cls, im_list, fn, root=None, no_class=False):
        """Short summary.

        Parameters
        ----------
        cls : type
            Description of parameter `cls`.
        im_list : type
            Description of parameter `im_list`.
        fn : type
            Description of parameter `fn`.
        root : type
            Description of parameter `root`.

        Returns
        -------
        type
            Description of returned object.
        """
        # create from a function parsed from name
        synsets = []
        items = {'image': [], 'label': []}
        for im in im_list:
            if isinstance(im, Path):
                path = str(im.resolve())
            else:
                assert isinstance(im, str)
                if root is not None and not os.path.isabs(im):
                    path = os.path.abspath(
                        os.path.join(root, os.path.expanduser(im)))
            items['image'].append(path)
            label = fn(Path(path))
            if isinstance(label, (int, bool, str)):
                label = str(label)
            else:
                raise ValueError(
                    'Expect returned label to be (str, int, bool), received {}'
                    .format(type(label)))
            if label not in synsets:
                synsets.append(label)
            items['label'].append(synsets.index(label))  # int label id
        return cls(items, classes=synsets, no_class=no_class)

    @classmethod
    def from_name_re(cls, im_list, fn, root=None):
        # create from a re parsed from name
        raise NotImplementedError

    @classmethod
    def from_label_func(cls, label_list, fn):
        # create from a function parsed from labels
        raise NotImplementedError


def _check_synsets(ref_synset, other_synset):
    """Check if other_synset is part of ref_synsetself. Not that even if
    other_synset is a subset, still be careful when comparing them.

    E.g., ref: ['apple', 'orange', 'melon'], other: ['apple', 'orange'] is OK
          ref: ['apple', 'orange', 'melon'], other: ['orange', 'melon'] is not!
    """
    if ref_synset == other_synset:
        return True
    if len(other_synset) < len(ref_synset):
        if ref_synset[:len(other_synset)] == other_synset:
            return True
    return False


def _show_images(images, cols=1, titles=None, fontsize=20):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(int(np.ceil(n_images / float(cols))), cols, n + 1)
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title, fontsize=fontsize)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


class _TorchImageClassificationDataset(TorchDataset):
    """Internal wrapper read entries in pd.DataFrame as images/labels.

    Parameters
    ----------
    dataset : ImageClassificationDataset
        DataFrame as ImageClassificationDataset.
    transform : Torchvision Transform function
        torch function for image transformation
    """

    def __init__(self, dataset, transform=None):
        if torch is None:
            raise RuntimeError('Unable to import pytorch which is required.')
        assert isinstance(dataset, TorchImageClassificationDataset)
        assert 'image' in dataset.columns
        self._has_label = 'label' in dataset.columns
        self._dataset = dataset
        self.classes = self._dataset.classes
        self._imread = Image.open
        self.transform = transform

    def __len__(self):
        return self._dataset.shape[0]

    def __getitem__(self, idx):
        im_path = self._dataset['image'][idx]
        img = self._imread(im_path).convert('RGB')
        # img = mmcv.imread(im_path, flag='color', channel_order='rgb')
        # img = Image.fromarray(img.astype('uint8')).convert('RGB')
        label = None
        if self._has_label:
            label = self._dataset['label'][idx]
        else:
            label = torch.tensor(-1, dtype=torch.long)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


if __name__ == '__main__':
    import autogluon.core as ag
    import pandas as pd
    csv_file = ag.utils.download(
        'https://autogluon.s3-us-west-2.amazonaws.com/datasets/petfinder_example.csv'
    )
    df = pd.read_csv(csv_file)
    df.head()
    df = TorchImageClassificationDataset.from_csv(csv_file)
    df.head()
    print(df)
    image_dir = ''
    train_data, _, _, = TorchImageClassificationDataset.from_folders(image_dir)
    print(train_data)
