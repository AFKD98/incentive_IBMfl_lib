"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20190891
© Copyright IBM Corp. 2021 All Rights Reserved.
"""
"""
Module to where fusion algorithms are implemented.
"""
import copy
import logging
import numpy as np

from ibmfl.model.model_update import ModelUpdate
from ibmfl.aggregator.fusion.iter_avg_fusion_handler import IterAvgFusionHandler
import ibmfl.util.fl_metrics as fl_metrics
from itertools import groupby
import torch as th
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance
from ibmfl.message.message import Message
from ibmfl.message.message_type import MessageType

logger = logging.getLogger(__name__)


class TierIterAvgFusionHandler(IterAvgFusionHandler):
    """
    Class for iterative averaging based fusion algorithms.
    An iterative fusion algorithm here referred to a fusion algorithm that
    sends out queries at each global round to registered parties for
    information, and use the collected information from parties to update
    the global model.
    The type of queries sent out at each round is the same. For example,
    at each round, the aggregator send out a query to request local model's
    weights after parties local training ends.
    The iterative algorithms can be terminated at any global rounds.

    In this class, the aggregator requests local model's weights from all
    parties at each round, and the averaging aggregation is performed over
    collected model weights. The global model's weights then are updated by
    the mean of all collected local models' weights.
    """

    def __init__(self, hyperparams,
                 protocol_handler,
                 data_handler=None,
                 fl_models=None,
                 shapley_value_test_model=None,
                 **kwargs):
        """
        Initializes an IterAvgFusionHandler object with provided information,
        such as protocol handler, fl_model, data_handler and hyperparams.

        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param protocol_handler: Protocol handler used for handling learning \
        algorithm's request for communication.
        :type protocol_handler: `ProtoHandler`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param kwargs: Additional arguments to initialize a fusion handler.
        :type kwargs: `Dict`
        :return: None
        """
        super().__init__(hyperparams,
                         protocol_handler,
                         data_handler,
                         fl_models[0],
                         **kwargs)
        self.name = "TierIterative-Weight-Average"
        self.params_global = hyperparams.get('global') or {}
        self.params_local = hyperparams.get('local') or None
        self.rounds = self.params_global.get('rounds') or 1
        self.curr_round = 0
        self.global_accuracy = -1
        self.previous_model_updates = []
        self.tier_client_idx = {}
        self.pre_training_rounds = 3
        self.termination_accuracy = self.params_global.get(
            'termination_accuracy')
        self.current_tiers = self.params_global.get(
            'tiers')
        self.shapley_value_test_model = shapley_value_test_model
        self.model_updates = []
        self.fl_models = fl_models
        # print(f'fl_models: {fl_models}')
        for fl_model in fl_models:
            if fl_model and fl_model.is_fitted():
                self.model_updates.append(fl_model.get_model_update())
            else:
                self.model_updates = None

        self.current_model_weights_per_tier = []
        for model_update in self.model_updates:
            self.current_model_weights_per_tier.append(model_update.get('weights') if model_update else None)
        logger.info('[FARAZ] len(self.current_model_weights_per_tier): ' + str(len(self.current_model_weights_per_tier)))
        self.current_model_weights = \
            self.model_updates[0].get('weights') if self.model_updates[0] else None

        if self.evidencia:
            from ibmfl.evidencia.util.hashing import hash_model_update

    # def get_cosine_scores():
    def get_model_logits(self, model):
        '''Returns the softmax layer logits of the model'''
        if isinstance(model, ModelUpdate):
            model = model.get('weights')

        return model[-1]
    
    def get_cosine_similarities(self):
        '''Returns the cosine similarities between the client model and the server model'''

        logger.info('[FARAZ] Calculating cosine similarities')
        
        similarities = {}
        for i in range(len(self.previous_model_updates)):
            models = self.previous_model_updates[i][0]
            parties = self.previous_model_updates[i][1]
            logger.info('[FARAZ] clients: ' + str(parties) + 'in tier: ' + str(i))
            tier_global_model_logits = self.get_model_logits(self.current_model_weights_per_tier[i])
            
            for j in range(len(parties)):
                client_id = parties[j]
                
                client_model_logits = self.get_model_logits(models[j])

                if client_id not in similarities.keys():
                    similarities[client_id] = [(1. - distance.cosine(client_model_logits, tier_global_model_logits), i)]
                else:
                    # print('client_model_logits: ', client_model_logits, 'Global tier model logits: ', tier_global_model_logits)
                    # print('cosine similarities: ', 1. - distance.cosine(client_model_logits, tier_global_model_logits))
                    cosine_similarity = 1. - distance.cosine(client_model_logits, tier_global_model_logits)
                    # if math.isnan(cosine_similarity):
                    #     cosine_similarity = 1.0
                    similarities[client_id].append((cosine_similarity, i))
            
        logger.info('[FARAZ] similarities: ' + str(similarities))
        return similarities

    def get_f1_scores_on_IID_data(self):
        """
        Returns f1 scores on IID data

        :return: f1 scores
        :rtype: `Dcit`
        """
        f1_scores = {}
        
        test_dataset = []
        x_test = self.data_handler.x_test
        y_test = self.data_handler.y_test
        for data_id in range(1, len(y_test)):
            test_dataset.append((x_test[data_id], y_test[data_id]))
        
        res = dict()
        # logger.info('[FARAZ] test_dataset: ' + str(test_dataset))
        # forming equal groups
        res = {key: [v[0] for v in val] for key, val in groupby(
            sorted(test_dataset, key=lambda ele: ele[1]), key=lambda ele: ele[1])}
        for party_no in range(0, len(self.previous_model_updates[0][1])):
            self.fl_models[1].update_model(self.previous_model_updates[0][0][party_no])
            f1_scores_per_class = []
            for key, value in res.items():
                
                y_preds = np.argmax(self.fl_models[1].predict(th.tensor(value)), axis=1)
                y_test_per_class = [key for i in range(len(y_preds))]
                f1_scores_per_class.append(fl_metrics.get_multi_label_classification_metrics(y_preds, y_test_per_class)['f1 weighted'])
            f1_scores[self.previous_model_updates[0][1][party_no]] = f1_scores_per_class
        logger.info('[FARAZ] f1_scores: {}'.format(f1_scores))

        return f1_scores
    
    def select_clients_on_basis_of_f1_Scores(self, f1_scores):
        """
        Selects clients on basis of f1 scores

        :param f1_scores: f1 scores
        :type f1_scores: `dict`
        :return: selected clients
        :rtype: `list`
        """
        # [FARAZ] Doing: Select clients on basis of f1 scores
        df  = pd.DataFrame.from_dict(f1_scores, orient='index')
        variance_df = df.var(axis=0).nlargest(2)
        top_variant_features = df.loc[:, variance_df.keys().tolist()].values
        tier_idx = KMeans(n_clusters=2, random_state=0).fit_predict(top_variant_features)
        # logger.info('[FARAZ] tier_idx: ' + str(tier_idx))
        
        k = 0
        for client_id in f1_scores.keys():
            
            if self.tier_client_idx.get(tier_idx[k]) is not None:
                self.tier_client_idx[tier_idx[k]].append(client_id)
            else:
                self.tier_client_idx[tier_idx[k]] = [client_id]
            k += 1
        
        #[FARAZ] Doing: Updating the tiers based on their client averages
        
        for tier_id, party_ids in self.tier_client_idx.items():
            #[FARAZ] Doing: get corresponding model updates from previous_model_updates using party ids
            idx = []
            for party_id in party_ids:
                idx.append(self.previous_model_updates[0][1].index(party_id))
            party_lst_per_tier = []
            for id in idx:
                party_lst_per_tier.append(self.previous_model_updates[0][0][id].get('weights'))
            #[FARAZ] Doing: get the average of the model updates for parties in a tier
            if len(party_lst_per_tier) > 1:
                self.current_model_weights_per_tier[tier_id] = party_lst_per_tier[0]
            else:
                self.current_model_weights_per_tier[tier_id] = np.mean(party_lst_per_tier, axis=0)
            
    
    def get_party_preferences(self):
        """
        Returns party preferences

        :return: party preferences
        :rtype: `dict`
        """
        registered_parties = self.ph.get_registered_parties()
        tier_global_model_logits = []
        logger.info('[FARAZ] Sending requests to parties to get their preferences')
        for i in range(self.current_tiers):
            tier_global_model_logits.append(self.get_model_logits(self.current_model_weights_per_tier[i]))
        party_preferences = self.ph.get_party_preferences(registered_parties, {'tier_model_logits': tier_global_model_logits})
        logger.info('[FARAZ] party_preferences: ' + str(party_preferences))
        
        return party_preferences
    
    def select_clients_per_tier(self):
        """
        Selects clients for each tier

        :return: selected clients
        :rtype: `list`
        """
        registered_parties = self.ph.get_registered_parties()
        
        selected_parties = {}
        
        if self.curr_round == self.pre_training_rounds:
            f1_scores = self.get_f1_scores_on_IID_data()
            self.select_clients_on_basis_of_f1_Scores(f1_scores)
            
            selected_parties = self.tier_client_idx
            
        elif self.curr_round > self.pre_training_rounds:
            
            party_preferences = self.get_party_preferences()
            tiers = party_preferences[0]
            parties = party_preferences[1]
            for i in range(0, len(tiers)):
                if selected_parties.get(int(tiers[i])) is not None:
                    selected_parties[int(tiers[i])].append(parties[i])
                else:
                    selected_parties[int(tiers[i])] = [parties[i]]
                
        else:
            selected_parties[0] = registered_parties
                       
        return selected_parties
    
    def get_selected_parties(self):
        """
        Returns selected clients

        :return: selected clients
        :rtype: `list`
        """
        selected_parties = self.select_clients_per_tier()
        return selected_parties
    
    def start_global_training_by_tier(self):
        """
        Starts an iterative global federated learning training process.
        """
        self.curr_round = 0
        while not self.reach_termination_criteria(self.curr_round):
            logger.info('[FARAZ] Starting round: ' + str(self.curr_round))
            # construct ModelUpdate
            if self.current_model_weights_per_tier:
                self.model_updates = []
                for current_model_weights in self.current_model_weights_per_tier:
                    self.model_updates.append(ModelUpdate(weights=current_model_weights))
            else:
                self.model_updates = None


            if self.model_updates:
                # log to Evidentia
                if self.evidencia:
                    self.evidencia.add_claim("sent_global_model",
                                            "{}, '\"{}\"'".format(self.curr_round + 1,
                                            hash_model_update_by_tier(self.model_updates)))

            lst_replies = {}
            
            selected_parties = self.get_selected_parties()
            #[FARAZ] query parties for each tier separately
            logger.info('[FARAZ] Initiating training requests')
            
            logger.info('[FARAZ] selected_parties: ' + str(selected_parties))
            for tier_id in selected_parties.keys():
                
                payload = {'hyperparams': {'local': self.params_local, 'tier': tier_id},
                        'model_updates': self.model_updates
                        }
                
                reply = self.query_parties(payload, selected_parties[tier_id], True)
                lst_replies[tier_id] = reply

            self.previous_model_updates = lst_replies
            # log to Evidentia
            if self.evidencia:
                updates_hashes = []
                for tier_id in lst_replies.keys():
                    for model_updates in lst_replies[tier_id]:
                        for update in model_updates:
                            updates_hashes.append(hash_model_update(update))
                            self.evidencia.add_claim("received_model_update_hashes",
                                                "{}, '{}'".format(self.curr_round + 1,
                                                str(updates_hashes).replace('\'', '"')))

            self.update_weights(lst_replies)
            # Update model if we are maintaining one
            for i in range(0, self.current_tiers):
                if self.fl_models[i] is not None:
                    self.fl_models[i].update_model(ModelUpdate(weights=self.current_model_weights_per_tier[i]))

            shapley_value_for_each_party = self.cal_shapley_value(lst_replies)
            logger.info(f'shapley_value_for_each_party: {shapley_value_for_each_party}')

            self.curr_round += 1
            self.save_current_state()
    
    def cal_aggregation_weight(self, lst_parties):
        '''
        get aggregation weight for each tier
        '''
        return 1 / len(lst_parties)

    def flatten_list_of_numpy(self, numpy_list):
        """
        flatten list of numpy to 1-dimensional numpy
        """
        temp_list = copy.deepcopy(numpy_list)
        res = []

        for item in temp_list:
            res.append(item.ravel())
        
        return np.concatenate(res)

    def get_gradient_on_test_data(self, aggregated_model_paramter):
        """
        Evaluate the local model based on the local test data.

        :param aggregated_model_paramter: parameter of aggregated_model. list of array
        :type aggregated_model_paramter: `list`
        :return: gradient of aggregated_model_paramter running on the aggregator test dataset. list of array
        :rtype: `list`
        """
        self.shapley_value_test_model.update_model(
            ModelUpdate(weights=aggregated_model_paramter)
        )

        (_), test_dataset = self.data_handler.get_data()
        gradients = self.shapley_value_test_model.get_gradient(
            train_data=test_dataset
        )
        return gradients
        

    def cal_shapley_value(self, lst_replies):
        """
        caculate shapley value for each party
        :param: lst_replies: info of parties update
        :type lst_replies: `dict[tier_id, tuple(lst_model_updates, lst_parties)]`
        :return: calculated shapley value for each party
        :rtype: `dict[tier_id, dict[party_id, shapley value]]`
        """
        # 2-layer dict 
        # For the first layer: key is tier_id, value is a dict
        # For the second layer: key is party_id, value is the shapley value
        shapley_value_for_each_party= {tier_id: {} for tier_id in range(self.current_tiers)}

        (_), test_dataset = self.data_handler.get_data()
        test_data_points_num = len(test_dataset[0])

        # calculate shapley value   
        for tier_id in range(self.current_tiers):
            # get aggregated model parameter for each tier
            if tier_id >= len(self.current_model_weights_per_tier):
                continue
            aggregated_model_paramter = self.current_model_weights_per_tier[tier_id]
            # get gradient of tier aggregated model on aggregator test dataset
            gradients = self.get_gradient_on_test_data(copy.deepcopy(aggregated_model_paramter))

            # calculate shapley value for each client
            # Note: using normalized aggregation since we delete 2 clients in the base case
            if tier_id not in lst_replies:
                continue
            lst_model_updates = lst_replies[tier_id][0]
            lst_parties = lst_replies[tier_id][1]

            for i in range(len(lst_model_updates)):
                party_parameter = lst_model_updates[i]
                party_parameter = self.fusion_collected_responses(modelUpdates=copy.deepcopy(party_parameter))
                party_id = lst_parties[i]
                normalized_parameter = party_parameter - aggregated_model_paramter

                aggregation_weight = self.cal_aggregation_weight(lst_parties=lst_parties)
                gradients_flatten = self.flatten_list_of_numpy(gradients)
                normalized_parameter_flatten = self.flatten_list_of_numpy(normalized_parameter)
                if len(gradients_flatten) != len(normalized_parameter_flatten):
                    raise ValueError('parameter size is not the same for calculating shapley value')

                shapley_value = - (1/test_data_points_num) * aggregation_weight * np.dot(gradients_flatten, normalized_parameter_flatten)
                shapley_value_for_each_party[tier_id][party_id] = shapley_value
        
        return shapley_value_for_each_party

    def fusion_collected_responses(self, modelUpdates, key='weights'):
        """
        Receives a model updates, where a model update is of the type
        `ModelUpdate`, using the values (indicating by the key)
        included in each model_update, it finds the mean.

        :param modelUpdates: A model updates of type `ModelUpdate` \
        to be averaged.
        :type modelUpdates:  `ModelUpdate`
        :param key: A key indicating what values the method will aggregate over.
        :type key: `str`
        :return: results after aggregation
        :rtype: `list`
        """        
        results = None
        try:
            results = np.array(modelUpdates.get(key))
        except Exception as ex:
            results = IterAvgFusionHandler.transform_update_to_np_array(modelUpdates.get(key))

        return results

    def update_weights(self, lst_model_updates):
        """
        Update the global model's weights with the list of collected
        model_updates from parties.
        In this method, it calls the self.fusion_collected_response to average
        the local model weights collected from parties and update the current
        global model weights by the results from self.fusion_collected_response.

        :param lst_model_updates: list of model updates of type `ModelUpdate` to be averaged.
        :type lst_model_updates: `list`
        :return: None
        """
        for tier_id in lst_model_updates.keys():
            if self.curr_round > self.pre_training_rounds:
                self.current_model_weights_per_tier[tier_id] = self.fusion_collected_responses_by_tier(lst_model_updates[tier_id])
            else:
                updated_weights = self.fusion_collected_responses_by_tier(lst_model_updates[tier_id])
                for tier_id in range(self.current_tiers):
                    self.current_model_weights_per_tier[tier_id] = updated_weights


    def fusion_collected_responses_by_tier(self, modelUpdates, key='weights'):
        """
        Receives a list of model updates, where a model update is of the type
        `ModelUpdate`, using the values (indicating by the key)
        included in each model_update, it finds the mean.

        :param lst_model_updates: List of model updates of type `ModelUpdate` \
        to be averaged.
        :type lst_model_updates:  `list`
        :param key: A key indicating what values the method will aggregate over.
        :type key: `str`
        :return: results after aggregation
        :rtype: `list`
        """
        v = []
        if type(modelUpdates) is tuple:
            modelUpdates = modelUpdates[0]
        for update in modelUpdates:
            try:
                update = np.array(update.get(key))
            except Exception as ex:
                update = IterAvgFusionHandler.transform_update_to_np_array(update.get(key))

            v.append(update)
        results = np.mean(np.array(v), axis=0)

        return results

    def reach_termination_criteria(self, curr_round):
        """
        Returns True when termination criteria has been reached, otherwise
        returns False.
        Termination criteria is reached when the number of rounds run reaches
        the one provided as global rounds hyperparameter.
        If a `DataHandler` has been provided and a targeted accuracy has been
        given in the list of hyperparameters, early termination is verified.

        :param curr_round: Number of global rounds that already run
        :type curr_round: `int`
        :return: boolean
        :rtype: `boolean`
        """
      
        if curr_round >= self.rounds:
            logger.info('Reached maximum global rounds. Finish training :) ')
            return True

        return self.terminate_with_metrics(curr_round)
    
    def send_global_models(self):
        """
        Send global models to all the parties
        """
        # Select data parties
        lst_parties = self.ph.get_available_parties()

        model_updates = self.get_global_models()
        payload = {'model_updates': model_updates
                   }

        logger.info('Sync Global Models' + str(model_updates))
        self.ph.sync_model_parties(lst_parties, payload)
        
    def get_global_model(self):
        """
        Returns last model_update

        :return: model_update
        :rtype: `ModelUpdate`
        """
        return ModelUpdate(weights=self.current_model_weights)
    
    def get_global_models(self):
        """
        Returns last model_updates

        :return: model_updates
        :rtype: `List of ModelUpdate`
        """
        global_models = []
        for i in range(0, self.current_tiers):
            global_models.append(ModelUpdate(weights=self.current_model_weights_per_tier[i]))
        return global_models

    def get_current_metrics(self):
        """Returns metrics pertaining to current state of fusion handler

        :return: metrics
        :rtype: `dict`
        """
        fh_metrics = {}
        fh_metrics['rounds'] = self.rounds
        fh_metrics['curr_round'] = self.curr_round
        fh_metrics['acc'] = self.global_accuracy
        #fh_metrics['model_update'] = self.model_update
        return fh_metrics

    @staticmethod
    def transform_update_to_np_array(update):
        """
        Transform a update of type list of numpy.ndarray to a numpy.ndarray 
        of numpy.ndarray.
        This method is a way to resolve the ValueError raised by numpy when 
        all the numpy.ndarray inside the provided list have the same 
        first dimension.

        A example of the possible case:
        a = [b, c], where a is of type list, b and c is of type numpy.ndarray.
        When b.shape[0] == c.shape[0] and b.shape[1] != c.shape[1], 
        the following line of code will cause numpy to raise a ValueError: 
        Could not broadcast input array from shape XXX(b.shape) into shape XX (c.shape).

        np.array(a)

        :param update: The input list of numpy.ndarray.
        :type update: `list`
        :return: the resulting update of type numpy.ndarray
        :rtype: `np.ndarray`
        """
        if update[0].shape[0]!= 2:
            update.append(np.zeros((2,)))
            update = np.array(update)
        else:
            update.append(np.zeros((3,)))
            update = np.array(update)
        return update[:-1]
