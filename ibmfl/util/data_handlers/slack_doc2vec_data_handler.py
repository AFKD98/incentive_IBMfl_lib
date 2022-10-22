"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
import logging
import pickle
from ibmfl.data.data_handler import DataHandler
from ibmfl.util.data_handlers.slack_reader import SlackReader
from ibmfl.util.datasets import load_slack

logger = logging.getLogger(__name__)


class SlackDoc2VecDataHandler(DataHandler):
    def __init__(self, data_config=None):
        super().__init__()
        self.user_path = None
        self.channel_path = None
        self.pfile = None

        if data_config is not None:
            if 'user_path' in data_config and 'channel_path' in data_config:
                self.user_path = data_config['user_path']
                self.channel_path = data_config['channel_path']

            elif 'pickled_file' in data_config:
                self.pfile = data_config['pickled_file']

    def get_data(self):
        """
        Gets pre-process slack training and testing data.

        :return: training data
        :rtype: 'tuple' in for the form of (docs, doc_ids)
        """
        if self.user_path is not None and self.channel_path is not None:
            try:
                logger.info('Loaded training data from ' + str(self.user_path))
                reader = SlackReader(self.user_path, self.channel_path)
                training_set = reader.create_doc2vec_training_set()
            except Exception:
                raise IOError('Unable to load training data from path '
                              'provided in config file: ' +
                              self.user_path)

        elif self.pfile is not None:
            with open(self.pfile, "rb") as pickled_file:
                training_set = pickle.load(pickled_file)

        else:
            training_set = load_slack()

        return training_set
