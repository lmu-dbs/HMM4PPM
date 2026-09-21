from __future__ import annotations

import numpy as np
import pandas as pd
import math
from multiprocessing.managers import DictProxy
from multiprocessing import shared_memory, Process
import time
from collections import Counter
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from enum import Enum
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from concurrent.futures import ProcessPoolExecutor
from ..data.sequencedata import SequenceData
from ..util.sequence_utils import _child_matches_with_sequence, _filter_start_end
import torch
import pyro
import pyro.distributions as dist
from torch.nn.functional import pad

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)

track_progress = False

class Task(Enum):
    NAP = 'nap'
    RTP = 'rtp'
    
class TreePredictor():
    """Prediction model using tree-based models
    """

    def __init__(self, include_phases: bool, predict_phase: bool, phase_as_feature: bool, pred_model, hmm_model, hmm_predictor, seq_len: int, **model_args):

        params = {'pred_model': pred_model,
                  }
        logger.info(f'Initializing prediction model - { {k:v for k,v in params.items()} }')

        self.pred_model = pred_model

        self.data_train = None
        self.data_test = None
        self.sequences = None
        self.sequence_trees = None
        self.train_model_preprocessors = dict()
        self.phase_pred_models = dict()
        
        self.include_phases = include_phases
        self.predict_phase = predict_phase
        self.phase_as_feature = phase_as_feature
        
        self.seq_len = seq_len
        
        self.hmm_model = hmm_model
        self.hmm_predictor = hmm_predictor
        
    def fit(self) -> None:
        """Fitting the model to X (training data). This involves the sequence extraction, generation of sequence trees
        and the ML model training for next activity prediction (/ remaining duration prediction)
        """
        training_sets = dict()
        phase_pred_models = dict()
        
        if self.phase_as_feature: # we train a global model
            phase = 0
            
            training_set_X, training_set_Y_disc, training_set_Y_cont, train_model_preprocessor_X, train_model_preprocessor_Y, train_dtypes_X, train_dtypes_Y = self.get_training_set_phase(per_phase=False)
            training_sets[phase] = (training_set_X, training_set_Y_disc, training_set_Y_cont)
            
            phase_pred_models = self.train_pred_model(training_set_X, training_set_Y_disc, training_set_Y_cont)
            
        else: # we train n_phases local models
            
            self.phases = np.unique(self.seq_data["train"].phases)
            
            training_set_X, training_set_Y_disc, training_set_Y_cont, train_model_preprocessor_X, train_model_preprocessor_Y, train_dtypes_X, train_dtypes_Y = self.get_training_set_phase(per_phase=True)
            
            for phase in tqdm(self.phases):
                training_sets[phase] = (training_set_X[phase], training_set_Y_disc[phase], training_set_Y_cont[phase])
                
                if training_set_X[phase].size != 0:
                    phase_pred_models[phase]= self.train_pred_model(training_set_X[phase], training_set_Y_disc[phase], training_set_Y_cont[phase])               

        self.phase_pred_models = phase_pred_models
        self.train_model_preprocessors = (train_model_preprocessor_X, train_model_preprocessor_Y)
        self.training_set_dtypes = (train_dtypes_X, train_dtypes_Y)
        
        logger.info('Phase model training completed!')


    def get_training_set_phase(self, per_phase: bool):

        # two general types
        # 
        # type 1: cluster-then-predict -> phase_as_feature = False
        #   we train n_phases models that are queried when some specific phase is active
        #   each model has a training set that has contains prefixes of the events from that phase up to a seq_len of [3, 5, 10, etc.]
        # 
        # type 2: extended feature set model -> phase_as_feature = True
        #   we train one model for the complete training dataset
        #   the model has a training set that has contains prefixes of the events from that phase up to a seq_len of [3, 5, 10, etc.]
        # 
        # both types can either forecast the phase themselves (multi-target model; predict_phase=True) or not (single-target model; predict_phase=False)
        # the latter need HMM filtering and prediction for future phases in the prediction loop
        
        # discrete
        activity_data = self.seq_data["train"].activities
        phase_data = self.seq_data["train"].phases
        resource_data = self.seq_data["train"].resources
        
        # continuous
        tsle_data = self.seq_data["train"].tsles
        tscs_data = self.seq_data["train"].tscss
        
        active_phases = self.seq_data["train"].phases[:,-1]
        
        train_activity_classes = [int(_) for _ in range(int(max(set(np.unique(activity_data)))) + 1)]
        train_phase_classes = [int(_) for _ in range(int(max(set(np.unique(phase_data)))) + 1)]
        train_resource_classes = [int(_) for _ in range(int(max(set(np.unique(resource_data)))) + 1)]
        
        next_activity_data = self.seq_data["train"].next_activity
        next_phase_data = self.seq_data["train"].next_phase
        next_resource_data = self.seq_data["train"].next_resource
        next_tsle_data = self.seq_data["train"].next_tsle
        next_tscs_data = self.seq_data["train"].next_tscs
        
        if self.phase_as_feature:
            
            disc_categories_X = [train_activity_classes]*self.seq_len + [train_phase_classes]*self.seq_len + [train_resource_classes]*self.seq_len
            preprocessor_X_disc = OneHotEncoder(categories=disc_categories_X, handle_unknown="ignore", sparse_output=False)
            preprocessor_X_cont = FunctionTransformer(func=None)
            
            X_train_disc = np.concatenate((activity_data, phase_data, resource_data), axis=1)
            X_train_cont = np.concatenate((tsle_data, tscs_data), axis=1)
            
            X_train_disc_enc = preprocessor_X_disc.fit_transform(X_train_disc)
            X_train_cont_enc = preprocessor_X_cont.fit_transform(X_train_cont)
            
            X_train_enc = np.concatenate((X_train_disc_enc, X_train_cont_enc), axis=1)
            
        else:
            
            disc_categories_X = [train_activity_classes]*self.seq_len + [train_resource_classes]*self.seq_len
            preprocessor_X_disc = OneHotEncoder(categories=disc_categories_X, handle_unknown="ignore", sparse_output=False)
            preprocessor_X_cont = FunctionTransformer(func=None)
            
            X_train_disc = np.concatenate((activity_data, resource_data), axis=1)
            X_train_cont = np.concatenate((tsle_data, tscs_data), axis=1)
            
            X_train_disc_enc = preprocessor_X_disc.fit_transform(X_train_disc)
            X_train_cont_enc = preprocessor_X_cont.fit_transform(X_train_cont)
            
            X_train_enc = np.concatenate((X_train_disc_enc, X_train_cont_enc), axis=1)
        
        if self.predict_phase:
            
            disc_categories_Y = [train_activity_classes]*1 + [train_phase_classes]*1 + [train_resource_classes]*1
            preprocessor_Y_disc = OneHotEncoder(categories=disc_categories_Y, handle_unknown="ignore", sparse_output=False)
            preprocessor_Y_cont = FunctionTransformer(func=None)
            
            Y_train_disc = np.stack((next_activity_data, next_phase_data, next_resource_data), axis=1)
            Y_train_cont = np.stack((next_tsle_data, next_tscs_data), axis=1)
            
            Y_train_disc_enc = preprocessor_Y_disc.fit_transform(Y_train_disc)
            Y_train_cont_enc = preprocessor_Y_cont.fit_transform(Y_train_cont)
            
            Y_train_enc = np.concatenate((Y_train_disc_enc, Y_train_cont_enc), axis=1)

        else:
            
            disc_categories_Y = [train_activity_classes]*1 + [train_resource_classes]*1
            preprocessor_Y_disc = OneHotEncoder(categories=disc_categories_Y, handle_unknown="ignore", sparse_output=False)
            preprocessor_Y_cont = FunctionTransformer(func=None)
            
            Y_train_disc = np.stack((next_activity_data, next_resource_data), axis=1)
            Y_train_cont = np.stack((next_tsle_data, next_tscs_data), axis=1)
            
            Y_train_disc_enc = preprocessor_Y_disc.fit_transform(Y_train_disc)
            Y_train_cont_enc = preprocessor_Y_cont.fit_transform(Y_train_cont)
            
            Y_train_enc = np.concatenate((Y_train_disc_enc, Y_train_cont_enc), axis=1)

        if per_phase:
            phase_training_datasets_X = {k: np.ndarray((0, X_train_enc.shape[1])) for k in train_phase_classes}
            phase_training_datasets_Y = {k: (np.ndarray((0, Y_train_disc_enc.shape[1])), (np.ndarray((0, Y_train_cont_enc.shape[1])))) for k in train_phase_classes}
            phase_training_datasets_Y_disc = {k: np.ndarray((0, Y_train_disc_enc.shape[1])) for k in train_phase_classes}
            phase_training_datasets_Y_cont = {k: np.ndarray((0, Y_train_cont_enc.shape[1])) for k in train_phase_classes}
            
            for obs_idx, phase in enumerate(active_phases):
                phase_training_datasets_X[phase] = np.concatenate((phase_training_datasets_X[phase], X_train_enc[obs_idx:obs_idx+1, :]))
                phase_training_datasets_Y_disc[phase] = np.concatenate((phase_training_datasets_Y_disc[phase], Y_train_disc_enc[obs_idx:obs_idx+1, :]))
                phase_training_datasets_Y_cont[phase] = np.concatenate((phase_training_datasets_Y_cont[phase], Y_train_cont_enc[obs_idx:obs_idx+1, :]))
            
            phase_training_data_X = phase_training_datasets_X
            phase_training_data_Y_disc = phase_training_datasets_Y_disc
            phase_training_data_Y_cont = phase_training_datasets_Y_cont
        else:
            phase_training_data_X = X_train_enc
            phase_training_data_Y_disc = Y_train_disc_enc
            phase_training_data_Y_cont = Y_train_cont_enc
        
        X_dtypes = None
        Y_dtypes = None
        
        return phase_training_data_X, phase_training_data_Y_disc, phase_training_data_Y_cont, (preprocessor_X_disc, preprocessor_X_cont), (preprocessor_Y_disc, preprocessor_Y_cont), X_dtypes, Y_dtypes

    def train_pred_model(self, training_set_X, training_set_Y_disc, training_set_Y_cont):
        
        model_disc, model_cont = PredModelFactory.create(self.pred_model)
        model_disc.fit(training_set_X, training_set_Y_disc)
        model_cont.fit(training_set_X, training_set_Y_cont)

        return (model_disc, model_cont)

    def predict(self, task: str, break_buffer: float, filter_tokens: bool, ncores: int, **pred_args) -> list[list[int]]:
        """Generates predictions for the chosen task (next activity prediction or remaining trace prediction)
        for a given set of sequences X.

        Args:
            task (str): The prediction task. Specify either 'nap' for next activity prediction or 'rtp' for
            remaining trace prediction
            break_buffer (float): factor by which a predicted sequence can overflow the length of the longest prefix
            seen in the training data

        Raises:
            NotImplementedError: If a different task other that 'nap' or 'rtp' is chosen

        Returns:
            np.array: The full predicted sequences (containing the given sequences with the appended predictions)
        """

        logger.info(f'Starting prediction - {task.upper()}')
        try:
            task = Task(task)
        except ValueError:
            raise ValueError(f'invalid task: {task} - only next activity prediction (nap) and \
                                      remaining trace prediction (rtp) are valid tasks')
        
        convert_start_time = time.perf_counter()
        prefix_array = _build_prefix_array(self.data_test.relevant_prefixes, self.seq_len)
        pred_start_time = time.perf_counter()
        
        include_phases = pred_args.get('include_phases')
        hmm_model = pred_args.get('hmm_model')
        hmm_predictor = pred_args.get('hmm_predictor')
        
        if include_phases and hmm_model is not None and hmm_predictor is not None: # and not self.predict_phase:
            assert prefix_array.shape[0] == pred_args['hmm_model'].model_args_test['sequences'].shape[0], "hmm sequences shape and prefix array shape do not match"
        
            sequence_indices = torch.arange(0, prefix_array.shape[0])
            filtered_sequences_tensor = get_filtered_tensor(sequence_indices, hmm_predictor=hmm_predictor)
            lengths = hmm_predictor.model.model_args_test['lengths'][sequence_indices]
            max_prefix_len = prefix_array.shape[2]
        
            padded_filtered_sequences = list()
            # pad all tensor sequences to max_prefix_len with length of prefix alpha values available
            for idx in range(filtered_sequences_tensor.shape[1]):
                seq_to_pad = filtered_sequences_tensor[:lengths[idx],idx,:]
                padded_seq = pad(seq_to_pad, (0, 0, max_prefix_len - seq_to_pad.shape[0], 0), value=np.nan)
                padded_filtered_sequences.append(padded_seq)
        
            padded_filtered_sequences = torch.stack(padded_filtered_sequences, dim=0).permute(0, 2, 1)
        
            assert prefix_array.shape[0] == padded_filtered_sequences.shape[0] and prefix_array.shape[2] == padded_filtered_sequences.shape[2], "shapes do not match after filtering"
        
            prefix_array = np.concatenate((prefix_array, padded_filtered_sequences.numpy()), axis=1)

        if task==Task.RTP:
            # do remaining trace prediction
            max_prefix_len = max([len(prefix['prefix']) for prefix in self.data_test.relevant_prefixes])
            max_seq_len = int(break_buffer*max_prefix_len)
            if ncores==1:
                predicted_traces = np.array([lst + [np.nan] * (max_seq_len - len(lst)) for lst in [self._predict_sequence(prefix=row,
                                                                                                                          break_after_seq_len=max_seq_len,
                                                                                                                          include_phases=include_phases,
                                                                                                                          hmm_predictor=hmm_predictor) for row in tqdm(prefix_array)]])
            else:
                raise NotImplementedError('multiprocessing not implemented for tree-based models - use ncores=1 and expect longer durations for prediction')

            pred_duration = time.perf_counter() - pred_start_time

            predictions = _reconvert_traces_from_shared_mem(predicted_traces)

            if filter_tokens:
                filtered_predictions = list()
                for pred in predictions:

                    try:
                        filtered_predictions.append(_filter_start_end(pred, self.start_activity, self.end_activity))
                    except TypeError:
                        filtered_predictions.append(None)

                predictions = filtered_predictions
            
            self.predicted_sequences = predictions
        
        elif task==Task.NAP:
            
            raise NotImplementedError('NAP not explicitly implemented for tree-based-models - next activities are inferred from first element of predicted suffixes')

        pred_plus_convert_duration = time.perf_counter() - convert_start_time

        return predictions, pred_duration, pred_plus_convert_duration

    def _predict_sequence(self, prefix: list[int], break_after_seq_len: int = 10e5, verbose: bool = False, include_phases: bool = False, hmm_predictor = None) -> list[int]:
        """Predicts the remaining activities for a given prefix

        Args:
            prefix (list[int]): A prefix to predict remaining activities for
        
        Returns:
           list[int]: the predicted sequence containing the prefix and the predicted remaining activities
        """

        nan_rows = np.all(np.isnan(prefix), axis=0)
        non_nan_prefix = prefix[:, ~nan_rows]
        act_prefix = [int(el) for el in non_nan_prefix[0, :]]
        phase_prefix = [int(el) for el in non_nan_prefix[1, :]]
        resource_prefix = [int(el) for el in non_nan_prefix[2, :]]
        tsle_prefix = [el for el in non_nan_prefix[3, :]]
        tscs_prefix = [el for el in non_nan_prefix[4, :]]
        
        non_nan_prefix_pred = np.stack((act_prefix, phase_prefix, resource_prefix, tsle_prefix, tscs_prefix)).T

        phase_prefix = [int(el) for el in non_nan_prefix[1, :]]
        alpha_prefix = non_nan_prefix[6:, :]
        
        predicted_sequence = [el for el in act_prefix if not np.isnan(el)]
        predicted_phase_sequence = [el for el in phase_prefix if not np.isnan(el)]
        initial_pred_seq_len = len(predicted_sequence)
        
        max_n_pred_steps = break_after_seq_len - initial_pred_seq_len
        if include_phases and not self.predict_phase:
            predicted_phases = update_phase_information_from_alpha_single_sequence(alpha=torch.tensor(alpha_prefix[:, -1]), n_pred_steps=max_n_pred_steps, hmm_predictor=hmm_predictor)
        else:
            predicted_phases = [0] * max_n_pred_steps
        
        complete_predicted_phase_sequence = predicted_phase_sequence + [int(p) if isinstance(predicted_phase_sequence[0], int) else float(p) for p in predicted_phases]
        
        predicted_sequence = act_prefix[:]
        initial_prefix_len = len(predicted_sequence)

        # start prediction loop
        prediction_idx = 0
        while predicted_sequence[-1] != self.end_activity and len(predicted_sequence) < break_after_seq_len:

            next_obs = self._predict_activity(prefix=non_nan_prefix_pred, break_after_seq_len=1, verbose=verbose, for_sfx=True)
            
            predicted_activity = int(next_obs[0, 0])
            
            predicted_sequence.append(predicted_activity)
            
            if predicted_sequence[-1] != self.end_activity:
                if self.predict_phase and self.phase_as_feature:
                    # shift the non_nan_prefix
                    non_nan_prefix_pred = np.concatenate((non_nan_prefix_pred, next_obs), axis=0)[-self.seq_len:]
                elif not self.predict_phase and self.phase_as_feature:
                    
                    phase_array_idx = 1 # do we need other indices when we do not include phases as feature? -> then we do not need phases at all...should be always at 1
                    next_phase = complete_predicted_phase_sequence[prediction_idx + initial_prefix_len]
                    next_obs = np.concatenate((next_obs[0, :phase_array_idx], np.array(float(next_phase)).reshape(1, ), next_obs[0, phase_array_idx:])).reshape(1, -1)
                    
                    non_nan_prefix_pred = np.concatenate((non_nan_prefix_pred, next_obs), axis=0)[-self.seq_len:]
                elif not self.predict_phase and not self.phase_as_feature:
                    phase_array_idx = 1 # do we need other indices when we do not include phases as feature? -> then we do not need phases at all...should be always at 1
                    next_phase = complete_predicted_phase_sequence[prediction_idx + initial_prefix_len]
                    next_obs = np.concatenate((next_obs[0, :phase_array_idx], np.array(float(next_phase)).reshape(1, ), next_obs[0, phase_array_idx:])).reshape(1, -1)
                    
                    non_nan_prefix_pred = np.concatenate((non_nan_prefix_pred, next_obs), axis=0)[-self.seq_len:]
                elif self.predict_phase and not self.phase_as_feature:
                    non_nan_prefix_pred = np.concatenate((non_nan_prefix_pred, next_obs), axis=0)[-self.seq_len:]
            
            prediction_idx += 1

        return predicted_sequence[initial_prefix_len:]
    
    def _batch_predict_sequence(self, prefixes, break_after_seq_len, worker_id, **kwargs):
        
        print_indices = [i for i in range(len(prefixes))][::max(1, int(len(prefixes) / 10))][1:] + [len(prefixes) - 1]

        predicted_sequences = np.zeros((len(prefixes), break_after_seq_len))
        predicted_sequences[:] = np.nan
        
        for prefix_idx, prefix in enumerate(prefixes):
            
            pred_sequence = self._predict_sequence(prefix=prefix,
                                                   break_after_seq_len=break_after_seq_len, 
                                                   **kwargs)
            
            predicted_sequences[prefix_idx, :len(pred_sequence)] = pred_sequence
            
            if prefix_idx in print_indices:
                logger.info(f"worker {worker_id}: {prefix_idx} prefixes completed ({100*(prefix_idx+1)/len(prefixes):.2f}%)")
        return predicted_sequences

    def _predict_activity(self, prefix: list[int], break_after_seq_len: int = 1, verbose: bool = False, for_sfx: bool = False) -> int:
        """Predicts the next activity for a given sequence

        Args:
            prefix (list[int]): A prefix to predict remaining activities for
        
        Returns:
            int: the predicted activity
        """

        if break_after_seq_len != 1:
            raise ValueError(f"break_after_seq_len should not be overridden - overridden with: {break_after_seq_len}")

        nan_rows = np.all(np.isnan(prefix), axis=1)
        non_nan_prefix = prefix[~nan_rows, :][-self.seq_len:, :]
        non_nan_prefix_len = non_nan_prefix.shape[0]
        act_prefix = [int(non_nan_prefix[0, 0])] * (self.seq_len - non_nan_prefix_len) + [int(el) for el in non_nan_prefix[:, 0]]
        phase_prefix = [int(non_nan_prefix[0, 1])] * (self.seq_len - non_nan_prefix_len) + [int(el) for el in non_nan_prefix[:, 1]]
        resource_prefix = [int(non_nan_prefix[0, 2])] * (self.seq_len - non_nan_prefix_len) + [int(el) for el in non_nan_prefix[:, 2]]
        tsle_prefix = [non_nan_prefix[0, 3]] * (self.seq_len - non_nan_prefix_len) + [el for el in non_nan_prefix[:, 3]]
        tscs_prefix = [non_nan_prefix[0, 4]] * (self.seq_len - non_nan_prefix_len) + [el for el in non_nan_prefix[:, 4]]
        
        predicted_sequence = act_prefix[:]

        # start prediction
        if predicted_sequence[-1] != self.end_activity:

            if self.predict_phase:
                
                if self.phase_as_feature: # we have one global model
                    input_data_disc = np.concatenate((act_prefix, phase_prefix, resource_prefix)).reshape(1, -1)
                    input_data_cont = np.concatenate((tsle_prefix, tscs_prefix)).reshape(1, -1)
                    
                    encoded_input_disc = self.train_model_preprocessors[0][0].transform(input_data_disc)
                    encoded_input_cont = self.train_model_preprocessors[0][1].transform(input_data_cont)
                    encoded_input = np.concatenate((encoded_input_disc, encoded_input_cont), axis=1)
                    
                    pred_target_disc = self.phase_pred_models[0].predict(encoded_input)
                    pred_target_cont = self.phase_pred_models[1].predict(encoded_input)
                else: # we have n phase models
                    active_phase = int(phase_prefix[-1])
                    if active_phase not in self.phase_pred_models.keys():
                        active_phase = int(np.random.choice([_ for _ in self.phase_pred_models.keys()]))
                    input_data_disc = np.concatenate((act_prefix, resource_prefix)).reshape(1, -1)
                    input_data_cont = np.concatenate((tsle_prefix, tscs_prefix)).reshape(1, -1)
                    
                    encoded_input_disc = self.train_model_preprocessors[0][0].transform(input_data_disc)
                    encoded_input_cont = self.train_model_preprocessors[0][1].transform(input_data_cont)
                    encoded_input = np.concatenate((encoded_input_disc, encoded_input_cont), axis=1)
                    
                    pred_target_disc = self.phase_pred_models[active_phase][0].predict(encoded_input)
                    pred_target_cont = self.phase_pred_models[active_phase][1].predict(encoded_input)
            
            else:
                if self.phase_as_feature: # we have one global model
                    input_data_disc = np.concatenate((act_prefix, phase_prefix, resource_prefix)).reshape(1, -1)
                    input_data_cont = np.concatenate((tsle_prefix, tscs_prefix)).reshape(1, -1)
                    
                    encoded_input_disc = self.train_model_preprocessors[0][0].transform(input_data_disc)
                    encoded_input_cont = self.train_model_preprocessors[0][1].transform(input_data_cont)
                    encoded_input = np.concatenate((encoded_input_disc, encoded_input_cont), axis=1)
                    
                    pred_target_disc = self.phase_pred_models[0].predict(encoded_input)
                    pred_target_cont = self.phase_pred_models[1].predict(encoded_input)
                else: # we have n phase models
                    active_phase = int(phase_prefix[-1])
                    if active_phase not in self.phase_pred_models.keys():
                        active_phase = int(np.random.choice([_ for _ in self.phase_pred_models.keys()]))
                    input_data_disc = np.concatenate((act_prefix, resource_prefix)).reshape(1, -1)
                    input_data_cont = np.concatenate((tsle_prefix, tscs_prefix)).reshape(1, -1)
                    
                    encoded_input_disc = self.train_model_preprocessors[0][0].transform(input_data_disc)
                    encoded_input_cont = self.train_model_preprocessors[0][1].transform(input_data_cont)
                    encoded_input = np.concatenate((encoded_input_disc, encoded_input_cont), axis=1)
                    
                    pred_target_disc = self.phase_pred_models[active_phase][0].predict(encoded_input)
                    pred_target_cont = self.phase_pred_models[active_phase][1].predict(encoded_input)
            
            disc_pred_dec = self.train_model_preprocessors[1][0].inverse_transform(pred_target_disc)
            cont_pred_dec = self.train_model_preprocessors[1][1].inverse_transform(pred_target_cont)
            
            for idx, disc_pred_val in enumerate(disc_pred_dec[0, :]):
                if disc_pred_val is None:
                    disc_pred_dec[0, idx] = float(0)
            
            next_obs = np.concatenate((disc_pred_dec, cont_pred_dec), axis=1).astype('float')

            if for_sfx:
                return next_obs
            else:
                predicted_activity = int(next_obs[0, 0])
                return predicted_activity

    def _batch_predict_activity(self, prefixes, break_after_seq_len, worker_id, **kwargs):

        print_indices = [i for i in range(len(prefixes))][::max(1, int(len(prefixes) / 10))][1:] + [len(prefixes) - 1]

        predicted_activities = np.zeros((len(prefixes), 1))
        predicted_activities[:] = np.nan
        
        for prefix_idx, prefix in enumerate(prefixes):
            
            pred_activity = self._predict_activity(prefix=prefix, 
                                                   break_after_seq_len=break_after_seq_len, 
                                                   **kwargs)
            
            predicted_activities[prefix_idx] = pred_activity
            
            if prefix_idx in print_indices:
                logger.info(f"worker {worker_id}: {prefix_idx} prefixes completed ({100*(prefix_idx+1)/len(prefixes):.2f}%)")
        return predicted_activities
        
    def load_data(self, train: SequenceData, val: SequenceData, test: SequenceData, phase_annotated_df: pd.DataFrame):
        self.data_train = train
        self.data_val = val
        self.data_test = test
    
        self.phase_annotated_df = phase_annotated_df

    def prepare_train(self, padding_size: int = 0):
        # TODO
        # include doc string

        logger.info('Preparing training data...')
        if self.data_train is None:
            raise ValueError('data not found - make sure to load train and test data with load_data()')
        
        train_case_ids = self.data_train.data[self.data_train.case_identifier].unique().tolist()
    
        # get phases from annotated dataset
        self.data_train.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_train.case_identifier].isin(train_case_ids)]['phase'].reset_index(drop=True)
        
        # generate next phase in data
        self.data_train.data['next_phase'] = self.data_train.data.groupby(self.data_train.case_identifier)['phase'].shift(-1)
        self.data_train.data['next_phase'] = self.data_train.data.groupby(self.data_train.case_identifier)['next_phase'].ffill()
        
        self.train_max_trace_len = self.data_train._get_max_trace_len()
        self.data_train.pad_columns(cols_to_pad=[self.data_train.activity_identifier], n_pad=padding_size)
        act_idx = self.data_train.data.groupby(self.data_train.case_identifier).apply(lambda x: pd.Series(range(-padding_size, 
                                                                                          len(x)-padding_size)))
        self.data_train.data['activity_idx'] = act_idx.reset_index(drop=True)

        # forward and backward fill timestamp column
        self.data_train.data[self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].ffill()
        self.data_train.data[self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].bfill()

        # self.data_train.encode_activities()
        # self.data_train.encode_attributes(attrs_to_encode=self.data_train.attributes)
        self.data_train.extract_traces(columns=[self.data_train.activity_identifier, 'phase'] + [self.data_train.resource_identifier, 'tsle', 'tscs'])
        self.start_activity = self.data_train.start_activity
        self.end_activity = self.data_train.end_activity

        self.data_train.generate_prefixes(attributes=[self.data_train.resource_identifier, 'tsle', 'tscs'], include_phases=self.include_phases)
        self.data_train.pick_relevant_prefixes()

        self.max_prefix_len = max([len(prefix['prefix']) for prefix in self.data_train.relevant_prefixes])

        logger.info('Training data prepared!')
        
    def prepare_val(self, act_encoder: LabelEncoder = None, attr_encoders: list[LabelEncoder] = None, filter_sequences: bool = True, padding_size: int = 0):
        
        logger.info('Preparing test data...')
        if self.data_val is None:
            raise ValueError('data not found - make sure to load train and test data with load_data()')
        
        val_case_ids = self.data_val.data[self.data_val.case_identifier].unique().tolist()
    
        # get phases from annotated dataset
        self.data_val.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_val.case_identifier].isin(val_case_ids)]['phase'].reset_index(drop=True)
    
        # generate next phase in data
        self.data_val.data['next_phase'] = self.data_val.data.groupby(self.data_val.case_identifier)['phase'].shift(-1)
        self.data_val.data['next_phase'] = self.data_val.data.groupby(self.data_val.case_identifier)['next_phase'].ffill()

        self.data_val.pad_columns(cols_to_pad=[self.data_val.activity_identifier], n_pad=padding_size)
        act_idx = self.data_val.data.groupby(self.data_val.case_identifier).apply(lambda x: pd.Series(range(-padding_size, 
                                                                                            len(x)-padding_size)))
        self.data_val.data['activity_idx'] = act_idx.reset_index(drop=True)

        # forward and backward fill timestamp column
        self.data_val.data[self.data_val.timestamp_identifier] = self.data_val.data.groupby(self.data_val.case_identifier)[self.data_val.timestamp_identifier].ffill()
        self.data_val.data[self.data_val.timestamp_identifier] = self.data_val.data.groupby(self.data_val.case_identifier)[self.data_val.timestamp_identifier].bfill()

        # self.data_test.encode_activities(act_encoder=act_encoder)
        # self.data_test.encode_attributes(attrs_to_encode=self.data_test.attributes, attr_encoders=attr_encoders)
        self.data_val.extract_traces(columns=[self.data_val.activity_identifier, 'phase'] + [self.data_train.resource_identifier, 'tsle', 'tscs'])

        self.data_val.generate_prefixes(attributes=[self.data_val.resource_identifier, 'tsle', 'tscs'], include_phases=self.include_phases)
        self.data_val.pick_relevant_prefixes()

        self.data_val.generate_full_sequences(filter_sequences=filter_sequences)
        self.data_val.generate_full_future_sequences(filter_sequences=filter_sequences)
        self.data_val.generate_next_activities()
        
        logger.info('Test data prepared!')
            
    def prepare_test(self, act_encoder: LabelEncoder = None, attr_encoders: list[LabelEncoder] = None, filter_sequences: bool = True, padding_size: int = 0):
        logger.info('Preparing test data...')
        if self.data_test is None:
            raise ValueError('data not found - make sure to load train and test data with load_data()')
        
        test_case_ids = self.data_test.data[self.data_test.case_identifier].unique().tolist()
    
        # get phases from annotated dataset
        self.data_test.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_test.case_identifier].isin(test_case_ids)]['phase'].reset_index(drop=True)
    
        # generate next phase in data
        self.data_test.data['next_phase'] = self.data_test.data.groupby(self.data_test.case_identifier)['phase'].shift(-1)
        self.data_test.data['next_phase'] = self.data_test.data.groupby(self.data_test.case_identifier)['next_phase'].ffill()
        
        self.data_test.pad_columns(cols_to_pad=[self.data_test.activity_identifier], n_pad=padding_size)
        act_idx = self.data_test.data.groupby(self.data_test.case_identifier).apply(lambda x: pd.Series(range(-padding_size, 
                                                                                          len(x)-padding_size)))
        self.data_test.data['activity_idx'] = act_idx.reset_index(drop=True)

        # forward and backward fill timestamp column
        self.data_test.data[self.data_test.timestamp_identifier] = self.data_test.data.groupby(self.data_test.case_identifier)[self.data_test.timestamp_identifier].ffill()
        self.data_test.data[self.data_test.timestamp_identifier] = self.data_test.data.groupby(self.data_test.case_identifier)[self.data_test.timestamp_identifier].bfill()

        # self.data_test.encode_activities(act_encoder=act_encoder)
        # self.data_test.encode_attributes(attrs_to_encode=self.data_test.attributes, attr_encoders=attr_encoders)
        self.data_test.extract_traces(columns=[self.data_test.activity_identifier, 'phase'] + [self.data_train.resource_identifier, 'tsle', 'tscs'])

        self.data_test.generate_prefixes(attributes=[self.data_test.resource_identifier, 'tsle', 'tscs'], include_phases=self.include_phases)
        self.data_test.pick_relevant_prefixes()

        self.data_test.generate_full_sequences(filter_sequences=filter_sequences)
        self.data_test.generate_full_future_sequences(filter_sequences=filter_sequences)
        self.data_test.generate_next_activities()
        
        logger.info('Test data prepared!')

    def generate_sequences(self, X: SequenceData):
        """Generates sequences of activities inside traces by process stage. 

        Args:
            X (SequenceData): The SequenceData object we generate the sequences for. This has to be prepared
            such that it has generated traces as class attribute (SequenceData.prepare_train())
        """
        logger.info(f'Generating sequences from trace data...')
        sequences = list()
        
        for trace in X.traces:
            sequence = trace[X.activity_identifier]
            sequences.append(sequence)

        self.sequences = sequences

        logger.info(f'Sequence generation completed!')
    
    def _batch_prefixes(self, nbatches: int):
        nprefixes = len(self.data_test.relevant_prefixes)
        batchsize = math.ceil(nprefixes/nbatches)
        for ndx in range(0, nprefixes, batchsize):
            yield self.data_test.relevant_prefixes[ndx:min(ndx + batchsize, nprefixes)]

    def _shared_worker(self, mode: str, 
                       worker_id: int, task: str, shm_name: str, shape: tuple, dtype: np.dtype,
                       start: int, end: int, max_seq_len: int, 
                       res_shm_name: str, res_shape: tuple, res_shm_dtype: np.dtype,
                       **kwargs):
        
        # attach to input data
        shm = shared_memory.SharedMemory(name=shm_name)
        prefix_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        # attach to results buffer
        res_shm = shared_memory.SharedMemory(name=res_shm_name)
        res = np.ndarray(res_shape, dtype=res_shm_dtype, buffer=res_shm.buf)

        try: 
            if task==Task.NAP:
                result = self._batch_predict_activity(mode=mode,
                                                      prefixes=prefix_array[start:end], 
                                                      break_after_seq_len=max_seq_len, 
                                                      worker_id=worker_id,
                                                      **kwargs)
            elif task==Task.RTP:
                result = self._batch_predict_sequence(mode=mode,
                                                      prefixes=prefix_array[start:end], 
                                                      break_after_seq_len=max_seq_len, 
                                                      worker_id=worker_id,
                                                      **kwargs)

            # write result to shared results buffer
            res[start:end] = result

        except Exception as e:
            logger.error(f"something went wrong in worker {worker_id}")
            raise

        finally:
            shm.close()
            res_shm.close()

class PredModelFactory:
    _pred_model = {
        "DecisionTree": (DecisionTreeClassifier, DecisionTreeRegressor),
        "RandomForest": (RandomForestClassifier, RandomForestRegressor),
        }

    @classmethod
    def create(cls, pred_model, **pred_model_params):
        if pred_model in cls._pred_model:
            pred_model_class_disc, pred_model_class_cont = (cls._pred_model[pred_model][0], cls._pred_model[pred_model][1])
            return pred_model_class_disc(**pred_model_params), pred_model_class_cont(**pred_model_params)
        else:
            raise ValueError(f"Unknown pred model: {pred_model}")


def update_phase_information(sequence_indices: torch.Tensor, sequence_lengths: torch.Tensor, n_pred_steps: int, hmm_predictor: HMMPredictor):
    
    num_samples = hmm_predictor.model.model_args_test['num_samples']
    num_sequences = sequence_indices.shape[0]
    probs_transition = hmm_predictor.model.trained_params['AutoDelta.probs_x']
    filtered_sequences = list()
    sequences = hmm_predictor.model.model_args_test['sequences'][sequence_indices,:,:]
    lengths = hmm_predictor.model.model_args_test['lengths'][sequence_indices]
    filtered_sequences_tensor, state_grid = hmm_predictor.filter_external(sequences=sequences, lengths=lengths, log=False)
    
    for seq_idx in range(filtered_sequences_tensor.shape[1]):
        # seq_obs_len = self.model.model_args_test['lengths'][seq_idx] + 1
        seq_obs_len = lengths[seq_idx] + 1
        filtered_sequences.append(filtered_sequences_tensor[:seq_obs_len, seq_idx, :])
        
    # we have the actual alpha values for the indexed prefixes
    # now extrapolate once with those and sample emissions
    # merge emissions with the PPM model predictions (if specified)
    # then filter again and report the next phase
    
    # we pick the last unpadded state of our sequences 
    if hmm_predictor.model.model_args_test['log_prob']:
        current_beliefs = filtered_sequences_tensor[lengths-1, torch.arange(num_sequences)].exp()
    else:
        current_beliefs = filtered_sequences_tensor[lengths-1, torch.arange(num_sequences)]
    
    with pyro.plate("sequences", num_sequences, dim=-1) as seq_idx:
        with pyro.plate("samples", num_samples, dim=-2) as sample_seq_idx:

            tuple_idx = pyro.sample(
                "x_tuple_0",
                dist.Categorical(current_beliefs[seq_idx]) # CHECK IF current_beliefs IS CORRECTLY HANDLED WITH NESTED PLATE STRUCTURE - DO WE NEED FURTHER INDEXING IN OTHER SAMPLE STATEMENTS?
            )
            state_tuple = state_grid[tuple_idx]  # (num_samples, num_sequences, markov_order)
            full_phases_tensor = state_tuple[:, :, probs_transition.dim():] # just empty but broadcastable tensor for all next phases

            for t in range(n_pred_steps):
                next_phases = pyro.sample(
                    f"x_future_{t}",
                    dist.Categorical(probs_transition[tuple(state_tuple.permute(2, 0, 1))]) # (num_samples, num_sequences, )
                )
                
                state_tuple = torch.cat([state_tuple[:, :, 1:], next_phases.unsqueeze(-1)], dim=2)
                full_phases_tensor = torch.cat([full_phases_tensor, next_phases.unsqueeze(-1)], dim=2)
    
    next_n_phases_mode, _ = full_phases_tensor.mode(dim=0)
    
    # # draw phases from filtered_sequences
    # phases = torch.stack([f_seq[-1,:].exp().argmax() for f_seq in filtered_sequences])
    
    return next_n_phases_mode.squeeze(0).tolist()

def get_filtered_tensor(sequence_indices: torch.Tensor, hmm_predictor: HMMPredictor):
    
    sequences = hmm_predictor.model.model_args_test['sequences'][sequence_indices,:,:]
    lengths = hmm_predictor.model.model_args_test['lengths'][sequence_indices]
    filtered_sequences_tensor, _ = hmm_predictor.filter_external(sequences=sequences, lengths=lengths, marginal_state_probs=False, log=False)
    
    return filtered_sequences_tensor

def update_phase_information_from_alpha(sequence_indices: torch.Tensor, alpha: torch.Tensor, n_pred_steps: int, hmm_predictor: HMMPredictor):
    
    num_samples = hmm_predictor.model.model_args_test['num_samples']
    num_sequences = sequence_indices.shape[0]
    probs_transition = hmm_predictor.model.trained_params['AutoDelta.probs_x']
    
    # we pick the last unpadded state of our sequences 
    if hmm_predictor.model.model_args_test['log_prob']:
        alpha = alpha.exp()
    else:
        alpha = alpha
    
    with pyro.plate("sequences", num_sequences, dim=-1) as seq_idx:
        with pyro.plate("samples", num_samples, dim=-2) as sample_seq_idx:

            tuple_idx = pyro.sample(
                "x_tuple_0",
                dist.Categorical(alpha[seq_idx]) # CHECK IF current_beliefs IS CORRECTLY HANDLED WITH NESTED PLATE STRUCTURE - DO WE NEED FURTHER INDEXING IN OTHER SAMPLE STATEMENTS?
            )
            state_tuple = hmm_predictor.state_grid[tuple_idx]  # (num_samples, num_sequences, markov_order)
            full_phases_tensor = state_tuple[:, :, probs_transition.dim():] # just empty but broadcastable tensor for all next phases

            for t in range(n_pred_steps):
                next_phases = pyro.sample(
                    f"x_future_{t}",
                    dist.Categorical(probs_transition[tuple(state_tuple.permute(2, 0, 1))]) # (num_samples, num_sequences, )
                )
                
                state_tuple = torch.cat([state_tuple[:, :, 1:], next_phases.unsqueeze(-1)], dim=2)
                full_phases_tensor = torch.cat([full_phases_tensor, next_phases.unsqueeze(-1)], dim=2)
    
    next_n_phases_mode, _ = full_phases_tensor.mode(dim=0)
    
    return next_n_phases_mode.squeeze(0).tolist()

def update_phase_information_from_alpha_single_sequence(alpha: torch.Tensor, n_pred_steps: int, hmm_predictor: HMMPredictor):
    
    num_samples = hmm_predictor.model.model_args_test['num_samples']
    probs_transition = hmm_predictor.model.trained_params['AutoDelta.probs_x']
    
    # we pick the last unpadded state of our sequences 
    if hmm_predictor.model.model_args_test['log_prob']:
        alpha = alpha.exp()
    else:
        alpha = alpha
    
    with pyro.plate("samples", num_samples, dim=-2):

        tuple_idx = pyro.sample(
            "x_tuple_0",
            dist.Categorical(alpha)
        )
        state_tuple = hmm_predictor.state_grid[tuple_idx]  # (num_samples, num_sequences, markov_order)
        full_phases_tensor = state_tuple[:, :, probs_transition.dim():] # just empty but broadcastable tensor for all next phases

        for t in range(n_pred_steps):
            next_phases = pyro.sample(
                f"x_future_{t}",
                dist.Categorical(probs_transition[tuple(state_tuple.permute(2, 0, 1))]) # (num_samples, num_sequences, )
            )
            
            state_tuple = torch.cat([state_tuple[:, :, 1:], next_phases.unsqueeze(-1)], dim=2)
            full_phases_tensor = torch.cat([full_phases_tensor, next_phases.unsqueeze(-1)], dim=2)
    
    next_n_phases_mode, _ = full_phases_tensor.mode(dim=0)
    
    return next_n_phases_mode.squeeze(0).tolist()