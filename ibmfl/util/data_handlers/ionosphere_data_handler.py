"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
import logging
import pandas as pd


from ibmfl.data.pandas_data_handler import PandasDataHandler
from ibmfl.util.datasets import load_ionosphere

logger = logging.getLogger(__name__)


class IonosphereDataHandler(PandasDataHandler):
    """
    Data handler for nursery dataset.
    """
    def __init__(self, data_config=None):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'txt_file' in data_config:
                self.file_name = data_config['txt_file']

        # load dataset
        self.dataset = self.load_dataset()
        # pre-process the data
        if 'class' in self.dataset.columns:
            self.y = self.dataset['class'].replace({'b': 0, 'g': 1})
            self.x = self.dataset.drop(['class'], axis=1)
        else:
            self.y = None
            self.x = self.dataset

        split_idx = int(len(self.dataset) * 0.8)
        self.x_train = self.x.iloc[:split_idx, :]
        self.x_test = self.x.iloc[split_idx:, :]
        if 'class' in self.dataset.columns:
            self.y_train = self.y.iloc[:split_idx]
            self.y_test = self.y.iloc[split_idx:]
        else:
            self.y_train = None
            self.y_test = None

    def get_data(self):
        """
        Read nursery.data.txt from a given dir.

        :return: A dataset structure
        :rtype: `tuple` of `pandas.core.frame.DataFrame`
        """
        return (self.x_train, self.y_train), (self.x_test, self.y_test)

    def get_dataset_info(self):
        """
        Read ionosphere.data.txt and extract data information

        :return: spec, a dictionary that contains list_of_features, \
        feature_values and list_of_labels.
        :rtype: `dict`
        """
        spec = {'list_of_features': list(range(self.dataset.shape[1]))}

        feature_values = []
        for feature in range(self.dataset.shape[1]):
            if self.dataset.columns[feature] != 'class':
                new_feature = self.dataset[self.dataset.columns[feature]].cat.categories
                feature_values.append(new_feature.tolist())
        spec['feature_values'] = feature_values

        list_of_labels = self.dataset['class'].cat.categories
        spec['list_of_labels'] = list_of_labels.tolist()

        return spec

    def load_dataset(self):
        """
        Loads the local dataset from a given local path. \
        If no local path is provided, it will download the original nursery \
        dataset from UCI.

        :return: raw dataset
        :rtype: `pandas.core.frame.DataFrame`
        """
        if self.file_name is None:
            dataset = load_ionosphere()
        else:
            try:
                logger.info('Loaded training data from {}'.format(self.file_name))
                dataset = pd.read_csv(self.file_name, dtype='category')
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' + self.file_name)
        return dataset
