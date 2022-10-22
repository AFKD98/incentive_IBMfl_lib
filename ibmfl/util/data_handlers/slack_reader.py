"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
import json
import os


class SlackReader(object):
    """
    Utility to process Slack channel/user data
    """

    def __init__(self, user_path=None, channel_path=None):
        """
        Initializes info from slack user metadata

        :param user_path: The path to the metadata file
        :type path: str
        :param channel_path: The path to the channels directory
        :type channel_path: str
        """
        self.slack_users = {}
        if user_path is not None and channel_path is not None:
            self.add_slack_users(user_path)
            self.add_slack_user_posts(channel_path)

    def add_slack_users(self, path):
        """
        Reads slack user info from a metadata file

        :param path: The path to the metadata file
        :type path: str
        :param filename: The name of the metadata file
        :type filename: str
        :return: None
        """
        self.slack_users = self.get_users(path)

    def add_slack_user_posts(self, path):
        """
        Retrieves all slack users' posts from slack channel metadata files

        :param path: Root dir to all slack channel metadata files
        :type path: str
        :return: None
        """
        self.slack_users = self.get_user_posts(path, self.slack_users)

    def add_slack_user_info(self, path):
        """
        Adds additional user info to a SlackUser

        :param path: The path to the user_info file
        :type path: str
        :param filename: The name of the user info file
        :type filename: str
        :param users: A dictionary of SlackUser objects, where the key is their slack user id
        :type dict
        :return: The updated dictionary of slack users, with additional user info added
        :rtype: dict
        """
        self.slack_users = self.get_user_info(path, self.slack_users)

    def add_slack_connections(self, path, threshold=5):
        """

        :param path: Root dir to all slack channel metadata files
        :type path: str
        :param threshold: threshold count for co-occurrence of other users
        :type threshold: int
        :return: None
        """
        self.slack_users = self.get_users_connections(path, self.slack_users, threshold)

    def remove_slack_users(self, threshold=100):
        """
        Removes all slack users who have posted less than some threshold

        :param threshold: The number to set the minimum amount of posts a user should have before removal
        :type threshold: int
        :return: None
        """
        self.slack_users = self.remove_users(self.slack_users, threshold)

    def get_slack_user(self, user_id):
        """
        Returns a SlackUser object specified by the provided user id
        :param user_id: The slack user id of the user to retrieve
        :type user_id: str
        :return: The SlackUser object determined by the user id. None if user id not found
        :rtype: semantic_search.utils.SlackUser
        """
        if user_id in self.slack_users:
            return self.slack_users[user_id]

        return None

    def create_doc2vec_training_set(self):
        """
        Creates a data set from slack users' post in a format expected for Doc2vec
        :return: A tuple in the form of (documents, document_ids)
        :rtype: tuple
        """
        docs = []
        doc_ids = []
        for user_id, user in self.slack_users.items():
            full_text = ""
            for message in user.posts:
                full_text += message + '\n'

            docs.append(full_text)
            doc_ids.append(user_id)

        return docs, doc_ids

    @staticmethod
    def get_users(path):
        """
        Reads slack user info from a metadata file

        :param path: The path to the metadata file
        :type path: str
        :param filename: The name of the metadata file
        :type filename: str
        :return: A dictionary of SlackUser objects, where they key is their slack user id
        :rtype: dict
        """
        users = {}
        file = path
        with open(file) as metadata_file:
            metadata = json.load(metadata_file)
            for user in metadata['users']:
                users[user] = SlackUser(user, metadata['users'][user])

        return users

    @staticmethod
    def get_user_posts(path, users):
        """
        Retrieves all slack users' posts from slack channel metadata files

        :param path: Root dir to all slack channel metadata files
        :type path: str
        :param users: A dictionary of SlackUser objects, where they key is their slack user id
        :type users: dict
        :return: The updated dictionary of slack users, with their posts from all slack channels appended
        :rtype: dict
        """
        for file in os.listdir(path):
            with open(os.path.join(path, file)) as channel_metadata:
                channel = json.load(channel_metadata)
                for message in channel['messages']:
                    if 'subtype' not in message and 'text' in message and 'user' in message:
                        if message['user'] in users:
                            users[message['user']].add_post(message['text'])

        return users

    @staticmethod
    def get_user_info(path, users):
        """
        Adds additional user info to a SlackUser

        :param path: The path to the user_info file
        :type path: str
        :param filename: The name of the user info file
        :type filename: str
        :param users: A dictionary of SlackUser objects, where the key is their slack user id
        :type dict
        :return: The updated dictionary of slack users, with additional user info added
        :rtype: dict
        """
        file = path
        email = ""
        first_name = ""
        last_name = ""

        with open(file) as user_info_file:
            user_info = json.load(user_info_file)

            for user in user_info:
                if user in users:
                    profile = user_info[user]['profile']
                    if 'first_name' in profile:
                        first_name = profile['first_name']
                    if 'last_name' in profile:
                        last_name = profile['last_name']
                    if 'email' in profile:
                        email = profile['email']

                users[user].add_user_info(email, first_name, last_name)

        return users

    @staticmethod
    def get_users_connections(path, users, threshold):
        """
        Creates a dictionary of co occurrers with co occurrence counts for each user

        :return: The updated dictionary of slack users, with their co-occurrers from slack conversations appended

        """
        for file in os.listdir(path):
            with open(os.path.join(path, file)) as channel_file:
                channel = json.load(channel_file)
                channel_members = []
                try:
                    channel_members = channel['channel_info']['members']
                except KeyError:
                    pass

                for member in channel_members:
                    for co_user in channel_members:
                        users[member].add_co_occurrer(co_user)
                        # try:
                        #     users[member].add_co_occurrer(co_user)
                        # except KeyError:
                        #     users[member].add_co_occurrer(co_user)

        for id_1, user in users.items():
            user_cooccurrers = user.co_occurrers
            user_id = user.user_id
            for cooc, count in user_cooccurrers.items():
                if count > threshold:
                    pair = (user_id, cooc)
                    user.connections.append(pair)
        return users

    @staticmethod
    def remove_users(users, threshold=100):
        """
        Removes all slack users who have posted less than some threshold

        :param users: A dictionary of SlackUser objects, where they key is their slack user id
        :type users: dict
        :param threshold: The number to set the minimum amount of posts a user should have before removal
        :type threshold: int
        :return: The updated dictionary with slack users removed
        :rtype: dict
        """
        updated_users = dict(users)
        for slack_id in users:
            if len(users[slack_id].posts) < threshold:
                del updated_users[slack_id]

        return updated_users


class SlackUser(object):
    """
    Represents a Slack user
    """

    def __init__(self, user_id, user_name):
        """
        Initializes a Slack user

        :param user_id: The user's Slack ID
        :type user_id: str
        :param user_name: The individual's Slack user name
        :type user_name: str
        """
        self.user_id = user_id
        self.user_name = user_name
        self.first_name = ""
        self.last_name = ""
        self.email = ""
        self.posts = []
        self.co_occurrers = {}
        self.connections = []

    def add_post(self, message):
        """
        Adds post associated with a Slack user

        :param message: A post made by a user on Slack
        :type message: str
        :return: None
        """
        self.posts.append(message)

    def add_user_info(self, email, first_name, last_name):
        """
        Adds additional info associated with a Slack user
        :param email:
        :param first_name:
        :param last_name:
        :return:
        """
        self.email = email
        self.first_name = first_name
        self.last_name = last_name

    def add_co_occurrer(self, co_user_id):
        if co_user_id != self.user_id:
            if co_user_id not in self.co_occurrers.keys():
                self.co_occurrers[co_user_id] = 1
            else:
                self.co_occurrers[co_user_id] += 1