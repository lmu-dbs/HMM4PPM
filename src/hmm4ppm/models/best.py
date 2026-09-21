import numpy as np
import pandas as pd
import math
from multiprocessing import shared_memory, Process
from joblib import Parallel, delayed
import time
from collections import Counter
from tqdm import tqdm
from enum import Enum
from sklearn.preprocessing import LabelEncoder
from ..data.sequencedata import SequenceData
from ..util.sequence_utils import _get_pattern_center, _child_matches_with_sequence, _filter_start_end
from ..util.logging import init_logging

from ..pyro.prediction import HMMPredictor
import pyro
import pyro.distributions as dist

import torch
from torch.nn.functional import pad

logger = init_logging(__name__, "best.log")

class Task(Enum):
    NAP = 'nap'
    RTP = 'rtp'
    
class BESTPredictorHMMPhases():
    """Prediction model using the Hierarchical Central Activity Pattern prediction algorithm.
    BEST is capable of predicting next activities as well as remaining traces for sequences of activities
    """

    def __init__(self, max_pattern_size, process_stage_width_percentage, min_freq, prune_func, include_phases, choice_tracker_keys_nap: list[str] = [], choice_tracker_keys_rtp: list[str] = [], parallelization_lib = 'joblib', **model_args):

        params = {'max_pattern_size':max_pattern_size,
                        'process_stage_width_percentage':process_stage_width_percentage,
                        'min_freq':min_freq,
                        'prune_func':prune_func,
                        'include_phases':include_phases}
        logger.info(f'Initializing prediction model - { {k:v for k,v in params.items()} }')
        if max_pattern_size % 2 == 0 or max_pattern_size <= 1:
            raise ValueError('max_pattern_size must be an odd integer > 1')

        self.max_pattern_size = max_pattern_size
        self.process_stage_width_percentage = process_stage_width_percentage
        self.min_freq = min_freq
        self.prune_func = prune_func
        self.include_phases = include_phases
        
        self._pattern_sizes = [_ for _ in range(1, self.max_pattern_size+1, 2)]
        self._padding_size = int(max_pattern_size/2)+1
        
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.hca_patterns = None

        # length of chosen pattern in prediction tracker
        self.choice_tracker_nap = {key:[] for key in choice_tracker_keys_nap}
        self.choice_tracker_rtp = {key:[] for key in choice_tracker_keys_rtp}
        
        self.parallelization_lib = parallelization_lib
        
    def fit(self) -> None:
        """Fitting the model to X (training data). This involves the pattern generation as well as matching
        of the patterns with their respective children/parents to be able to construct a hierarchical tree of
        central activity patterns
        """
        self.generate_patterns(self.data_train)
        self.find_child_patterns()

        unpruned_nodes = dict()
        pruned_nodes = dict()
        stage_trees = dict()

        logger.info(f'Building pattern tree for {len(self._stages)} stages...')

        for stage in tqdm(self._stages):
            stage_matches = self._matches_per_stage[stage]
            stage_dict_matches = _get_matches_dict(pattern=(), all_matches=stage_matches, max_pattern_size=self.max_pattern_size, min_freq=self.min_freq)
            stage_trees[stage] = stage_dict_matches

            tree_nodes = extract_tree_with_pruning(stage_dict_matches, prune_func=self.prune_func)

            current_unpruned_nodes = {key:node for key, node in tree_nodes.items() if node['pruned'] is False}
            current_pruned_nodes = {key:node for key, node in tree_nodes.items() if node['pruned'] is True}
            pruned_nodes[stage] = current_pruned_nodes
            unpruned_nodes[stage] = current_unpruned_nodes
        
        logger.info('Pattern tree built!')

        self._pruned_nodes = pruned_nodes
        self._unpruned_nodes = unpruned_nodes
        self._stage_trees = stage_trees

    def predict(self, eval_pattern_size: int, task: str, selection_method: str, break_buffer: float, filter_tokens: bool, ncores: int, weight: float|None = None, **pred_args) -> list[list[int]]:
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
        prefix_array = _build_prefix_array(self.data_test.relevant_prefixes, eval_pattern_size)
        pred_start_time = time.perf_counter()

        include_phases = pred_args.get('include_phases')
        hmm_model = pred_args.get('hmm_model')
        hmm_predictor = pred_args.get('hmm_predictor')
        
        if include_phases and hmm_model is not None and hmm_predictor is not None:
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

        # if task==Task.RTP:
        if True:
            max_prefix_len = max([len(prefix['prefix']) for prefix in self.data_test.relevant_prefixes])
            
            max_seq_len = int(break_buffer*max_prefix_len)

            predicted_traces = list()
            if ncores == 1:
                predicted_traces = np.array([lst + [np.nan] * (max_seq_len - len(lst)) for lst in [self._predict_sequence(eval_pattern_size=eval_pattern_size,
                                                                                                                          prefix=row,
                                                                                                                          idx=row_idx,
                                                                                                                          selection_method=selection_method,
                                                                                                                          weight=weight,
                                                                                                                          break_after_seq_len=max_seq_len,
                                                                                                                          include_phases=include_phases,
                                                                                                                          hmm_model=hmm_model,
                                                                                                                          hmm_predictor=hmm_predictor) for row_idx, row in enumerate(tqdm(prefix_array))]])
            else:
                shm_input, shared_input, shm_result, shared_result = _setup_shared_memory(prefix_array, max_seq_len=max_seq_len)

                # generating index slices according to number of workers (ncores)
                chunksize = len(prefix_array) // ncores
                slices = [
                    (i * chunksize, (i + 1) * chunksize if i < ncores - 1 else len(prefix_array))
                    for i in range(ncores)
                ]

                if self.parallelization_lib == 'joblib':
                    t0 = time.perf_counter()
                    try:
                        Parallel(n_jobs=ncores)(
                            delayed(self._shared_worker)(eval_pattern_size, selection_method, weight, i, task, shm_input.name, prefix_array.shape, prefix_array.dtype, s, e, max_seq_len, 
                            shm_result.name, shared_result.shape, shared_result.dtype,)
                            for i, (s, e) in enumerate(slices)
                        )
                        elapsed = time.perf_counter() - t0
                        predicted_traces = shared_result.copy()
                    finally:
                        for s in [shm_input, shm_result]:
                            s.close()
                            s.unlink()
                            
                elif self.parallelization_lib == 'multiprocessing':
                    procs = [
                        Process(
                            target=self._shared_worker,
                            args=(eval_pattern_size, selection_method, weight, i, task, shm_input.name, prefix_array.shape, prefix_array.dtype, s, e, max_seq_len, 
                            shm_result.name, shared_result.shape, shared_result.dtype,
                                )
                        )
                        for i, (s, e) in enumerate(slices)
                    ]

                    t0 = time.perf_counter()
                    try:
                        for p in procs: p.start()
                        for p in procs: p.join()
                        for p in procs:
                            if p.exitcode != 0:
                                raise RuntimeError(f"Worker {p.pid} failed with exit code {p.exitcode}")
                        elapsed = time.perf_counter() - t0
                        predicted_traces = shared_result.copy()
                    finally:
                        for s in [shm_input, shm_result]:
                            s.close()
                            s.unlink()
                else:
                    raise ValueError('unknown parallelization method')
                
                logger.info(f"Workers done in {elapsed:.3f}s  |  {len(predicted_traces):,} prefixes processed")
                logger.info(f"Total time elapsed: {time.perf_counter() - pred_start_time:.3f}s")

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
        
        # elif task==Task.NAP:
        if True:
            max_seq_len = 1
            
            if ncores==1:
                predicted_activities = np.array([self._predict_activity(eval_pattern_size=eval_pattern_size,
                                                                        prefix=row,
                                                                        selection_method=selection_method,
                                                                        weight=weight) for row in tqdm(prefix_array)])

            else:
                shm_input, shared_input, shm_result, shared_result = _setup_shared_memory(prefix_array, max_seq_len=max_seq_len)

                # generating index slices according to number of workers (ncores)
                chunksize = len(prefix_array) // ncores
                slices = [
                    (i * chunksize, (i + 1) * chunksize if i < ncores - 1 else len(prefix_array))
                    for i in range(ncores)
                ]
                
                if self.parallelization_lib == 'joblib':
                    t0 = time.perf_counter()
                    try:
                        Parallel(n_jobs=ncores)(
                            delayed(self._shared_worker)(eval_pattern_size, selection_method, weight, i, task, shm_input.name, prefix_array.shape, prefix_array.dtype, s, e, max_seq_len, 
                            shm_result.name, shared_result.shape, shared_result.dtype,)
                            for i, (s, e) in enumerate(slices)
                        )
                        elapsed = time.perf_counter() - t0
                        predicted_activities = shared_result.copy()
                    finally:
                        for s in [shm_input, shm_result]:
                            s.close()
                            s.unlink()
                            
                elif self.parallelization_lib == 'multiprocessing':
                    procs = [
                        Process(
                            target=self._shared_worker,
                            args=(eval_pattern_size, selection_method, weight, i, task, shm_input.name, prefix_array.shape, prefix_array.dtype, s, e, max_seq_len, 
                                  shm_result.name, shared_result.shape, shared_result.dtype,
                                  )
                        )
                        for i, (s, e) in enumerate(slices)
                    ]

                    t0 = time.perf_counter()
                    try:
                        for p in procs: p.start()
                        for p in procs: p.join()
                        for p in procs:
                            if p.exitcode != 0:
                                raise RuntimeError(f"Worker {p.pid} failed with exit code {p.exitcode}")
                        elapsed = time.perf_counter() - t0
                        predicted_activities = shared_result.copy()
                    finally:
                        for s in [shm_input, shm_result]:
                            s.close()
                            s.unlink()
                else:
                    raise ValueError('unknown parallelization method')

                logger.info(f"Workers done in {elapsed:.3f}s  |  {len(predicted_activities):,} prefixes processed")
                logger.info(f"Total time elapsed: {time.perf_counter() - pred_start_time:.3f}s")

            pred_duration = time.perf_counter() - convert_start_time

            if ncores > 1:
                predictions = _reconvert_activities_from_shared_mem(predicted_activities)
            else:
                predictions = predicted_activities
            
            self.predicted_activities = predictions

        pred_plus_convert_duration = time.perf_counter() - convert_start_time

        return predictions, pred_duration, pred_plus_convert_duration

    def _predict_sequence(self, eval_pattern_size: int, prefix: list[int], idx: int, selection_method: str, weight: float|None = None, break_after_seq_len: int = 10e5, include_phases: bool = False, hmm_model = None, hmm_predictor = None, verbose: bool = False) -> list[int]:
        """Predicts the remaining activities for a given prefix

        Args:
            prefix (list[int]): A prefix to predict remaining activities for
        
        Returns:
           list[int]: the predicted sequence containing the prefix and the predicted remaining activities
        """
        act_prefix = prefix[0, :]
        initial_prefix_len = int(prefix[1, -1])
        phase_prefix = prefix[2, :]
        alpha_prefix = prefix[3:, :]
        
        act_phase_len_diff = ~np.isnan(act_prefix).sum() - ~np.isnan(phase_prefix).sum()
        
        predicted_sequence = [el for el in act_prefix if not np.isnan(el)]
        predicted_phase_sequence = [el for el in phase_prefix if not np.isnan(el)]
        initial_pred_seq_len = len(predicted_sequence)
        
        max_n_pred_steps = break_after_seq_len - initial_prefix_len
        if include_phases:
            predicted_phases = update_phase_information_from_alpha_single_sequence(alpha=torch.tensor(alpha_prefix[:, -1]), n_pred_steps=max_n_pred_steps, hmm_predictor=hmm_predictor)
        else:
            predicted_phases = [0] * max_n_pred_steps
        
        complete_predicted_phase_sequence = predicted_phase_sequence + [int(p) if isinstance(predicted_phase_sequence[0], int) else float(p) for p in predicted_phases]
        
        max_process_stage = max(self._unpruned_nodes.keys())
        min_process_stage = min(self._unpruned_nodes.keys())

        # start prediction loop
        prediction_idx = len(predicted_sequence) - act_phase_len_diff - 1
        # we could also go with prediction_idx = len(predicted_phase_sequence) - 1
        while predicted_sequence[-1] != self.end_activity and prediction_idx + initial_prefix_len < break_after_seq_len:
            # current_process_stage = min(int((len(predicted_sequence[last_start:]) - 1) / self._abs_process_stage_width), max_process_stage)
            current_process_stage = complete_predicted_phase_sequence[prediction_idx]
            
            current_prediction, pattern_attributes = self._pred_for_process_stage(stage=current_process_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
            increase_stage, decrease_stage = current_process_stage + 1, current_process_stage - 1

            while len(current_prediction)==0 and not (increase_stage>max_process_stage and decrease_stage<min_process_stage):
                # look for next higher stage
                increase_stage = min(increase_stage, max_process_stage)
                current_increase_prediction, increase_pattern_attributes = self._pred_for_process_stage(stage=increase_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
                increase_stage += 1

                # look for next lower stage
                decrease_stage = max(decrease_stage, min_process_stage)
                current_decrease_prediction, decrease_pattern_attributes = self._pred_for_process_stage(stage=decrease_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
                decrease_stage -= 1

                increase_prob = increase_pattern_attributes.get('prob_local')
                decrease_prob = decrease_pattern_attributes.get('prob_local')

                # if both find something take the one with the higher probability
                if decrease_prob is not None and increase_prob is not None:
                    if increase_prob > decrease_prob:
                        current_prediction = current_increase_prediction
                        pattern_attributes = increase_pattern_attributes
                    else:
                        current_prediction = current_decrease_prediction
                        pattern_attributes = decrease_pattern_attributes
                elif increase_prob is not None:
                    current_prediction = current_increase_prediction
                    pattern_attributes = increase_pattern_attributes
                elif decrease_prob is not None:
                    current_prediction = current_decrease_prediction
                    pattern_attributes = decrease_pattern_attributes
                else:
                    # nothing has been found in any of the other stages - repeat with different stages
                    pass
        
            if len(current_prediction) == 0:
                if len(predicted_sequence) == len(act_prefix):
                    logger.debug(f'We did not find any predicted activities for prefix {act_prefix}')
                else:
                    logger.debug(f'We did not find any further predicted activities for running prediction {predicted_sequence}')
                return predicted_sequence[len(act_prefix):]
            
            for choice_metric in self.choice_tracker_rtp.keys():
                try:
                    self.choice_tracker_rtp[choice_metric].append(pattern_attributes[choice_metric])
                except KeyError as e:
                    wrong_keys = [key for key in self.choice_tracker_rtp.keys() if key not in pattern_attributes.keys()]
                    raise KeyError(f"Pattern attributes do not match those of the choice trackers - wrong keys: {', '.join(wrong_keys)}") from e
            prediction_idx += len(current_prediction) - 1
            # predicted_phase_sequence.extend([int(p) if isinstance(predicted_phase_sequence[0], int) else  float(p) for p in predicted_phases]) # next phases are appended to the phase sequence
            predicted_sequence.extend(current_prediction[1:]) # whole pattern is appended to the prediction

        return predicted_sequence[initial_pred_seq_len:]
    
    def _batch_predict_sequence(self, eval_pattern_size, prefixes, selection_method, weight, break_after_seq_len, worker_id, **kwargs):
        
        print_indices = [i for i in range(len(prefixes))][::max(1, int(len(prefixes) / 10))][1:] + [len(prefixes) - 1]

        predicted_sequences = np.zeros((len(prefixes), break_after_seq_len))
        predicted_sequences[:] = np.nan

        for prefix_idx, prefix in enumerate(prefixes):
            pred_sequence = self._predict_sequence(eval_pattern_size=eval_pattern_size, 
                                                   prefix=prefix, 
                                                   selection_method=selection_method, 
                                                   weight=weight, 
                                                   break_after_seq_len=break_after_seq_len, 
                                                   **kwargs)
            
            predicted_sequences[prefix_idx, :len(pred_sequence)] = pred_sequence

            if prefix_idx in print_indices:
                logger.info(f"worker {worker_id}: {prefix_idx} prefixes completed ({100*(prefix_idx+1)/len(prefixes):.2f}%)")

        return predicted_sequences


    def _predict_activity(self, eval_pattern_size: int, prefix: list[int], selection_method: str, weight: float|None = None, break_after_seq_len: int = 1, verbose: bool = False) -> int:

        """Predicts the next activity for a given sequence

        Args:
            prefix (list[int]): A prefix to predict remaining activities for
        
        Returns:
            int: the predicted activity
        """

        if break_after_seq_len != 1:
            raise ValueError(f"break_after_seq_len should not be overridden - overridden with: {break_after_seq_len}")
        
        act_prefix = prefix[0,:]
        phase_prefix = prefix[1,:]
        
        act_phase_len_diff = ~np.isnan(act_prefix).sum() - ~np.isnan(phase_prefix).sum()
    
        predicted_sequence = [el for el in act_prefix if not np.isnan(el)]
        predicted_phase_sequence = [el for el in phase_prefix if not np.isnan(el)]

        max_process_stage = max(self._unpruned_nodes.keys())
        min_process_stage = min(self._unpruned_nodes.keys())

        # start prediction loop
        prediction_idx = len(predicted_sequence) - act_phase_len_diff - 1
        # we could also go with prediction_idx = len(predicted_phase_sequence) - 1
        if predicted_sequence[-1] != self.end_activity:
            # current_process_stage = min(int((len(predicted_sequence[last_start:]) - 1) / self._abs_process_stage_width), max_process_stage)
            current_process_stage = predicted_phase_sequence[prediction_idx]
            
            current_prediction, pattern_attributes = self._pred_for_process_stage(stage=current_process_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
            increase_stage, decrease_stage = current_process_stage + 1, current_process_stage - 1

            while len(current_prediction)==0 and not (increase_stage>max_process_stage and decrease_stage<min_process_stage):
                # look for next higher stage
                increase_stage = min(increase_stage, max_process_stage)
                current_increase_prediction, increase_pattern_attributes = self._pred_for_process_stage(stage=increase_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
                increase_stage += 1

                # look for next lower stage
                decrease_stage = max(decrease_stage, min_process_stage)
                current_decrease_prediction, decrease_pattern_attributes = self._pred_for_process_stage(stage=decrease_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
                decrease_stage -= 1

                increase_prob = increase_pattern_attributes.get('prob_local')
                decrease_prob = decrease_pattern_attributes.get('prob_local')

                # if both find something take the one with the higher probability
                if decrease_prob is not None and increase_prob is not None:
                    if increase_prob > decrease_prob:
                        current_prediction = current_increase_prediction
                        pattern_attributes = increase_pattern_attributes
                    else:
                        current_prediction = current_decrease_prediction
                        pattern_attributes = decrease_pattern_attributes
                elif increase_prob is not None:
                    current_prediction = current_increase_prediction
                    pattern_attributes = increase_pattern_attributes
                elif decrease_prob is not None:
                    current_prediction = current_decrease_prediction
                    pattern_attributes = decrease_pattern_attributes
                else:
                    # nothing has been found in any of the other stages - repeat with different stages
                    pass
        
            if len(current_prediction) == 0:
                if verbose:
                    print("WE DID NOT FIND ANY SUITABLE PREDICTION FOR THE CURRENT SEQUENCE ANYWHERE")
                return
            
            for choice_metric in self.choice_tracker_nap.keys():
                try:
                    self.choice_tracker_nap[choice_metric].append(pattern_attributes[choice_metric])
                except KeyError as e:
                    wrong_keys = [key for key in self.choice_tracker_nap.keys() if key not in pattern_attributes.keys()]
                    raise KeyError(f"Pattern attributes do not match those of the choice trackers - wrong keys: {', '.join(wrong_keys)}") from e
        return current_prediction[1]

    def _batch_predict_activity(self, eval_pattern_size, prefixes, selection_method, weight, break_after_seq_len, worker_id, **kwargs):

        print_indices = [i for i in range(len(prefixes))][::max(1, int(len(prefixes) / 10))][1:] + [len(prefixes) - 1]

        predicted_activities = np.zeros((len(prefixes), 1))
        predicted_activities[:] = np.nan

        for prefix_idx, prefix in enumerate(prefixes):
            pred_activity = self._predict_activity(eval_pattern_size=eval_pattern_size, 
                                                   prefix=prefix, 
                                                   selection_method=selection_method, 
                                                   weight=weight, 
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

    def prepare_train(self):
        
        logger.info('Preparing training data...')
        if self.data_train is None:
            raise ValueError('data not found - make sure to load train and test data with load_data()')
        
        train_case_ids = self.data_train.data[self.data_train.case_identifier].unique().tolist()
        
        # get phases from annotated dataset
        self.data_train.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_train.case_identifier].isin(train_case_ids)]['phase'].reset_index(drop=True)
        
        try:
            start_activity = self.data_train.start_activity
            end_activity = self.data_train.end_activity
        except:
            start_activity = 'START'
            end_activity = 'END'
        
        self.train_max_trace_len = self.data_train._get_max_trace_len()
        self.data_train.pad_columns(cols_to_pad=[self.data_train.activity_identifier],
                                    n_pad=self._padding_size - 1, # we already have a 1 pre/post padding from HMM
                                    forward_pad=end_activity,
                                    backward_pad=start_activity)
        
        # forward and backward fill timestamp column
        self.data_train.data[self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].ffill()
        self.data_train.data[self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].bfill()

        # self.data_train.encode_activities()
        self.data_train.extract_traces(columns=[self.data_train.activity_identifier, 'phase'])
        self.start_activity = self.data_train.start_activity
        self.end_activity = self.data_train.end_activity

        self.data_train.generate_prefixes(include_phases=self.include_phases)
        self.data_train.pick_relevant_prefixes()

        self.max_prefix_len = max([len(prefix['prefix']) for prefix in self.data_train.relevant_prefixes])

        logger.info('Training data prepared!')

    def prepare_val(self, act_encoder: LabelEncoder|None = None, filter_sequences: bool = True):
        
        logger.info('Preparing validation data...')
        if self.data_val is None:
            raise ValueError('data not found - make sure to load train, validation and test data with load_data()')
        
        val_case_ids = self.data_val.data[self.data_val.case_identifier].unique().tolist()
        
        # get phases from annotated dataset
        self.data_val.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_val.case_identifier].isin(val_case_ids)]['phase'].reset_index(drop=True)
        
        try:
            # we access START/END encoding from training here
            start_activity = self.start_activity
            end_activity = self.end_activity
        except:
            start_activity = 'START'
            end_activity = 'END'
        
        self.data_val.pad_columns(cols_to_pad=[self.data_val.activity_identifier],
                                  n_pad=self._padding_size - 1, # we already have a 1 pre/post padding from HMM
                                  forward_pad=end_activity,
                                  backward_pad=start_activity)
        # forward and backward fill timestamp column
        self.data_val.data[self.data_val.timestamp_identifier] = self.data_val.data.groupby(self.data_val.case_identifier)[self.data_val.timestamp_identifier].ffill()
        self.data_val.data[self.data_val.timestamp_identifier] = self.data_val.data.groupby(self.data_val.case_identifier)[self.data_val.timestamp_identifier].bfill()

        # self.data_test.encode_activities(act_encoder=act_encoder)
        self.data_val.extract_traces(columns=[self.data_val.activity_identifier, 'phase'])

        self.data_val.generate_prefixes(include_phases=self.include_phases)
        self.data_val.pick_relevant_prefixes()

        self.data_val.generate_full_sequences(filter_sequences=filter_sequences)
        self.data_val.generate_full_future_sequences(filter_sequences=filter_sequences)
        self.data_val.generate_next_activities()
        logger.info('Validation data prepared!')
    
    def prepare_test(self, act_encoder: LabelEncoder|None = None, filter_sequences: bool = True):
        
        logger.info('Preparing test data...')
        if self.data_test is None:
            raise ValueError('data not found - make sure to load train and test data with load_data()')
        
        test_case_ids = self.data_test.data[self.data_test.case_identifier].unique().tolist()
        
        # get phases from annotated dataset
        self.data_test.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_test.case_identifier].isin(test_case_ids)]['phase'].reset_index(drop=True)
        
        try:
            # we access START/END encoding from training here
            start_activity = self.start_activity
            end_activity = self.end_activity
        except:
            start_activity = 'START'
            end_activity = 'END'
        
        self.data_test.pad_columns(cols_to_pad=[self.data_test.activity_identifier],
                                   n_pad=self._padding_size - 1, # we already have a 1 pre/post padding from HMM
                                   forward_pad=end_activity,
                                   backward_pad=start_activity)
        # forward and backward fill timestamp column
        self.data_test.data[self.data_test.timestamp_identifier] = self.data_test.data.groupby(self.data_test.case_identifier)[self.data_test.timestamp_identifier].ffill()
        self.data_test.data[self.data_test.timestamp_identifier] = self.data_test.data.groupby(self.data_test.case_identifier)[self.data_test.timestamp_identifier].bfill()
        
        # self.data_test.encode_activities(act_encoder=act_encoder)
        self.data_test.extract_traces(columns=[self.data_test.activity_identifier, 'phase'])

        self.data_test.generate_prefixes(include_phases=self.include_phases)
        self.data_test.pick_relevant_prefixes()

        self.data_test.generate_full_sequences(filter_sequences=filter_sequences)
        self.data_test.generate_full_future_sequences(filter_sequences=filter_sequences)
        self.data_test.generate_next_activities()
        logger.info('Test data prepared!')

    def generate_patterns(self, X: SequenceData):
        """Generates sequential patterns of activities inside traces by process stage. The generated patterns are
        given as a dict with pattern sizes as keys. Each pattern is a tuple of two elements where the first element is
        the pattern as a list of ints and the second element is the respective process stage
        (with the absolute width being specified by self.process_stage_width_percentage)

        Args:
            X (SequenceData): The SequenceData object we generate the sequential patterns for. This has to be prepared
            such that it has generated traces as class attribute (SequenceData.prepare_train())
        """
        logger.info(f'Generating patterns from trace data...')
        patterns_by_size = dict()
        patterns = list()
        
        for pattern_size in self._pattern_sizes:
            current_size_patterns = list()
            for trace in X.traces:
                act_trace_to_process = trace[X.activity_identifier]
                phase_trace_to_process = trace['phase']
                trace_index_start = self._padding_size-int(pattern_size/2)-1
                trace_index_end = len(act_trace_to_process)-self._padding_size+int(pattern_size/2)+1
                act_trace_to_process = act_trace_to_process[trace_index_start:trace_index_end] # ensure no patterns of only END and START tokens
                if self.include_phases:
                    phase_trace_to_process = phase_trace_to_process[trace_index_start:trace_index_end]
                else:
                    phase_trace_to_process = [0] * len(act_trace_to_process)
                
                current_size_patterns.extend(make_pattern_stage_tuples((act_trace_to_process, phase_trace_to_process), pattern_size=pattern_size,
                                                   pad_short_seqs=True, include_phases=self.include_phases))
                
            patterns_by_size[pattern_size] = current_size_patterns
            patterns.extend(current_size_patterns)
        
        self._hca_patterns_by_size = patterns_by_size
        self.hca_patterns = patterns

        pattern_strings = [(','.join([str(el) for el in pattern[0]]), pattern[1]) for pattern in patterns]
        pattern_freqs_by_stage = calc_stage_wise_freqs(pattern_strings)
        self._hca_pattern_freqs_by_stage = pattern_freqs_by_stage

        logger.info(f'Pattern generation completed!')

    def find_child_patterns(self):
        
        logger.info(f'Searching for child patterns...')
        all_stage_values = [pattern[1] for pattern in self.hca_patterns]
        
        stages = set(all_stage_values)
        self._stages = stages

        original_sizes = [0] + self._pattern_sizes[:-1]
        extended_sizes = self._pattern_sizes[:]

        matches_per_stage = dict()

        for stage in stages:
            unique_ext_patterns = list()
            all_matches = dict()
            for orig_size, ext_size in zip(original_sizes, extended_sizes):

                if len(unique_ext_patterns) == 0:
                    if orig_size == 0:
                        orig_patterns = [[]]
                    else:
                        orig_patterns = [pattern[0] for pattern in self._hca_patterns_by_size[orig_size] 
                                        if pattern[1]==stage]
                    
                    orig_pattern_counts = dict(Counter(map(tuple, orig_patterns)))
                    unique_orig_patterns = {pattern:{'prob':1,
                                                     'global_prob':freq/sum(orig_pattern_counts.values()),
                                                     'freq':freq}
                                            for pattern, freq in orig_pattern_counts.items()}
                else:
                    unique_orig_patterns = unique_ext_patterns.copy()

                ext_patterns = [pattern[0] for pattern in self._hca_patterns_by_size[ext_size] 
                                if pattern[1]==stage]
                ext_pattern_counts = dict(Counter(map(tuple, ext_patterns)))
                unique_ext_patterns = {pattern:{'prob':float(),
                                                'global_prob':freq/sum(ext_pattern_counts.values()),
                                                'freq':freq}
                                       for pattern, freq in ext_pattern_counts.items()}
                
                # get matching extended patterns for each original pattern
                size_matches = dict()
                for orig, orig_attrs in unique_orig_patterns.items():
                    matches = dict()
                    # match_counts = list()
                    if orig == ():
                    # if orig == []:
                        pattern_string = ''
                    else:
                        pattern_string = ','.join([str(el) for el in orig])
                    
                    if pattern_string not in all_matches.keys():
                        for ext, ext_attrs in unique_ext_patterns.items():
                            if orig == ext[1:-1]:
                                matches[ext] = ext_attrs
                        # scale probabilities to unit sum (for the matching extended patterns)
                        probs_to_rescale = [val['global_prob'] for val in matches.values()]
                        rescaled_probs = _rescale_probs(probs_to_rescale)
                        for idx, match_attrs in enumerate(matches.values()):
                            match_attrs['prob'] = rescaled_probs[idx]
                        
                        for ext, ext_attrs in matches.items():
                            unique_ext_patterns[ext] = ext_attrs
                        size_matches[pattern_string] = {'prob':orig_attrs['prob'], 
                                                        'global_prob':orig_attrs['global_prob'], 
                                                        'freq':orig_attrs['freq'], 
                                                        'matches':matches}
                all_matches[orig_size] = size_matches
            matches_per_stage[stage] = all_matches
        
        self._matches_per_stage = matches_per_stage
        logger.info(f'Child pattern search completed!')

    def _pred_for_process_stage(self, eval_pattern_size: int, stage: int, sequence: list[int], selection_method: str, weight: float|None = None, verbose: bool = False) -> list[int]:
        
        if self.prune_func is not None:
        
            current_important_patterns = self._unpruned_nodes[stage]

            empty_key = [key for key, val in current_important_patterns.items() if val['name']=='']
            if empty_key:
                if verbose:
                    print('i deleted the empty key, sir :)')
                current_important_patterns.pop(empty_key[0])

            center_activity = sequence[-1]
            # check for children of center activity
            children = {key:val 
                        for key, val in current_important_patterns.items() 
                        if _get_pattern_center(val) == center_activity}

            # we exclude single activity children here (because we check for matches with the center activity (not with the left part))
            non_atomic_patterns = [key for key, child in children.items() if len(child['name'].split(',')) > 1]
            children = {child_key: children[child_key] for child_key in non_atomic_patterns}

            if verbose:
                print(children)

            applying_children = {key:val 
                                    for key, val in children.items() 
                                    if _child_matches_with_sequence(val, sequence)}
            
            if len(applying_children)==0:
                return [], 0
            
            probs = [val['prob'] for val in applying_children.values()]
            dists = [val['total_log_rpif_dist'] for val in applying_children.values()]
            lens_children = [len(val['name'].split(',')) for val in applying_children.values()]

            max_prob = max(probs)
            argmax_prob_indices = [idx for idx, p in enumerate(probs) if p==max_prob]
            argmax_children = [[int(act) for act in [val['name'] for val in applying_children.values()][pick].split(',')] for pick in argmax_prob_indices]
            argmax_children_dists = [dists[idx] for idx in argmax_prob_indices]

            min_dist = min(argmax_children_dists)
            argmin_dist_indices = [idx for idx, d in enumerate([dists[i] for i in argmax_prob_indices]) if d==min_dist]
            min_dist_pick = np.argmin(argmax_children_dists)        
        else:
            
            all_applying_children = self.extract_matching_patterns(stage, sequence)

            # kick single activity patterns from extracted matching patterns
            all_applying_children = all_applying_children[1:]
            
            lens_all_children = [len(p['name'].split(',')) for p in all_applying_children]
            applying_children = [c for c, c_len in zip(all_applying_children, lens_all_children) if c_len <= eval_pattern_size]

            if len(applying_children)==0:
                return [], {}

            # get probs, dists and lens
            probs = [p['prob'] for p in applying_children]
            global_probs = [p['global_prob'] for p in applying_children]
            dists = [p['total_log_rpif_dist'] for p in applying_children]
            lens_children = [len(p['name'].split(',')) for p in applying_children]

            # for weighted distance
            global_probs_parent = [p["global_prob_parent"] for p in applying_children]
            local_beds = [p["log_rpif_dist"] for p in applying_children]
            global_beds_parents = [p["total_log_rpif_dist"] - p["log_rpif_dist"] for p in applying_children]

            # DIFFERENT SELECTION METHODS:
            # PROB_LEN_DIST: 1. min(local BED) 2. max pattern length 3. min (global BED) - what we used in the BPM paper
            # PROB_DIST: 1. min(local BED) 2. min (global BED)
            # WEIGHTED_DIST: weighted distance metric between local BED and global BED (of the respective parent patterns)
            # WEIGHTED_DIST_LEN: 1. weighted distance metric from WEIGHTED_DIST 2. max pattern length
            # WEIGHTED_PROBS: weighted probabilty (sum) between conditional extension probability and global occurrence prob of the parent
            # WEIGHTED_PROBS_LEN: 1. weighted probabilty (sum) from WEIGHTED_PROBS 2. max pattern length
            # MAX_JOINT_PROB: joint probability (product) between conditional extension probability and global occurrence prob of the parent
            # MAX_JOINT_PROB_LEN: 1. joint probability (product) from MAX_JOINT_PROB 2. max pattern length 

            if selection_method == 'PROB_LEN_DIST':
                # 1 - filter for max prob (minimal local BED)
                max_prob = max(probs)
                argmax_prob_indices = [idx for idx, p in enumerate(probs) if p==max_prob]

                # 2 - filter for maximal match length (longest patterns)
                argmax_prob_children_lens = [lens_children[idx] for idx in argmax_prob_indices]
                max_len = max(argmax_prob_children_lens)
                argmax_prob_argmax_len_indices = [idx for idx, l in enumerate(lens_children) if l==max_len and idx in argmax_prob_indices]

                # 3 - filter for minimal global BED
                argmax_prob_argmax_len_dists = [dists[idx] for idx in argmax_prob_argmax_len_indices]
                min_dist_max_len = min(argmax_prob_argmax_len_dists)
                argmax_prob_argmax_len_argmin_dist_indices = [idx for idx, d in enumerate(dists) if d==min_dist_max_len and idx in argmax_prob_argmax_len_indices]
            
                candidate_children = [[p for p in applying_children][pick] for pick in argmax_prob_argmax_len_argmin_dist_indices] # with max prob - max len - min dist

                if len(argmax_prob_argmax_len_argmin_dist_indices) > 1:
                    # pattern_lens = [len(c) for c in candidate_children]
                    pattern_lens = [len([act for act in c['name'].split(',')]) for c in candidate_children]
                    logger.debug(f'we have {len(argmax_prob_argmax_len_argmin_dist_indices)} patterns with same cond prob of {max_prob}, len of {max_len} and min distance of {min_dist_max_len:.4f} - choosing randomly')
                    picked_child = candidate_children[np.random.choice(range(0, len(candidate_children)))]
                else:
                    picked_child = candidate_children[0]
            
            elif selection_method == 'PROB_DIST':
                # 1 - filter for max prob (minimal local BED)
                max_prob = max(probs)
                argmax_prob_indices = [idx for idx, p in enumerate(probs) if p==max_prob]

                # 2 - filter for minimal global BED
                argmax_prob_dists = [dists[idx] for idx in argmax_prob_indices]
                min_dist = min(argmax_prob_dists)
                argmax_prob_argmax_len_dists = [dists[idx] for idx in argmax_prob_argmax_len_indices]
                argmax_prob_argmin_dist_indices = [idx for idx, d in enumerate(dists) if d==min_dist and idx in argmax_prob_indices]

                candidate_children = [[p for p in applying_children][pick] for pick in argmax_prob_argmin_dist_indices]

                if len(argmax_prob_argmin_dist_indices) > 1:
                    # pattern_lens = [len(c) for c in candidate_children]
                    pattern_lens = [len([act for act in c['name'].split(',')]) for c in candidate_children]
                    logger.debug(f'we have {len(argmax_prob_argmin_dist_indices)} patterns with same cond prob of {max_prob} and min distance of {min_dist:.4f} - they have lens {pattern_lens} - choosing randomly')
                    picked_child = candidate_children[np.random.choice(range(0, len(candidate_children)))]
                else:
                    picked_child = candidate_children[0]

            elif selection_method == "WEIGHTED_DIST":
                
                if weight is None:
                    raise ValueError(f"Need to set a weight for selection method {selection_method}")

                # calculate weighted distance
                weighted_dists = [weight * lb + (1 - weight) * gbp for lb, gbp in zip(local_beds, global_beds_parents)]
                min_weighted_dist = min(weighted_dists)
                argmin_weighted_dist_indices = [idx for idx, wd in enumerate(weighted_dists) if wd == min_weighted_dist]
                
                candidate_children = [[p for p in applying_children][pick] for pick in argmin_weighted_dist_indices]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            elif selection_method == "WEIGHTED_PROBS":

                if weight is None:
                    raise ValueError(f"Need to set a weight for selection method {selection_method}")

                # calculate weighted probs
                weighted_probs = [weight * bed + (1 - weight) * gpp for bed, gpp in zip(probs, global_probs_parent)]
                max_weighted_probs = max(weighted_probs)
                argmax_weighted_probs = [idx for idx, wd in enumerate(weighted_probs) if wd == max_weighted_probs]
                candidate_children = [[p for p in applying_children][pick]for pick in argmax_weighted_probs]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            elif selection_method == "WEIGHTED_PROBS_LEN":
                
                if weight is None:
                    raise ValueError(f"Need to set a weight for selection method {selection_method}")

                # 1 - calculate weighted probs
                weighted_probs = [weight * bed + (1 - weight) * gpp for bed, gpp in zip(probs, global_probs_parent)]
                max_weighted_probs = max(weighted_probs)
                argmax_weighted_probs_indices = [idx for idx, wd in enumerate(weighted_probs) if wd == max_weighted_probs]

                # 2 - filter for maximal match length (longest patterns)
                argmax_weighted_probs_lens = [lens_children[idx] for idx in argmax_weighted_probs_indices]
                max_len = max(argmax_weighted_probs_lens)
                argmax_weighted_probs_argmax_len_indices = [idx for idx, l in enumerate(lens_children) if l==max_len and idx in argmax_weighted_probs_indices]
                
                candidate_children = [[p for p in applying_children][pick] for pick in argmax_weighted_probs_argmax_len_indices]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            elif selection_method == "WEIGHTED_DIST_LEN":
                
                if weight is None:
                    raise ValueError(f"Need to set a weight for selection method {selection_method}")

                # 1 - calculate weighted distance
                weighted_dists = [weight * lb + (1 - weight) * gbp for lb, gbp in zip(local_beds, global_beds_parents)]
                min_weighted_dist = min(weighted_dists)
                argmin_weighted_dist_indices = [idx for idx, wd in enumerate(weighted_dists) if wd == min_weighted_dist]

                # 2 - filter for maximal match length (longest patterns)
                argmin_weighted_dist_lens = [lens_children[idx] for idx in argmin_weighted_dist_indices]
                max_len = max(argmin_weighted_dist_lens)
                argmin_weighted_dist_argmax_len_indices = [idx for idx, l in enumerate(lens_children) if l==max_len and idx in argmin_weighted_dist_indices]
                
                candidate_children = [[p for p in applying_children][pick] for pick in argmin_weighted_dist_argmax_len_indices]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            elif selection_method == "MAX_JOINT_PROB":

                # calculate joint probs
                joint_probs = [cp*gpp for cp, gpp in zip(probs, global_probs_parent)]
                max_joint_probs = max(joint_probs)
                argmax_joint_probs = [idx for idx, jp in enumerate(joint_probs) if jp == max_joint_probs]

                candidate_children = [[p for p in applying_children][pick] for pick in argmax_joint_probs]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            elif selection_method == "MAX_JOINT_PROB_LEN":

                # 1 - calculate join probability
                joint_probs = [cp*gpp for cp, gpp in zip(probs, global_probs_parent)]
                max_joint_probs = max(joint_probs)
                argmax_joint_probs_indices = [idx for idx, jp in enumerate(joint_probs) if jp == max_joint_probs]

                # 2 - filter for maximal match length (longest patterns)
                argmax_joint_probs_lens = [lens_children[idx] for idx in argmax_joint_probs_indices]
                max_len = max(argmax_joint_probs_lens)
                argmax_joint_probs_argmax_len_indices = [idx for idx, l in enumerate(lens_children) if l==max_len and idx in argmax_joint_probs_indices]
                
                candidate_children = [[p for p in applying_children][pick] for pick in argmax_joint_probs_argmax_len_indices]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            else:
                raise NotImplementedError(f"Chosen selection method {selection_method} not implemented!")
        
        for prob, length in zip(probs, lens_children):

            if verbose:
                print(f"child prob: {prob}, child len: {length}")

        try:
            picked_pattern = [int(act) for act in picked_child['name'].split(',')]
        except ValueError:
            picked_pattern = [float(act) for act in picked_child['name'].split(',')]

        pred = picked_pattern[math.floor(len(picked_pattern)/2):]

        pattern_attributes = {'prob_local': picked_child['prob'], 
                              'prob_global': picked_child['global_prob_parent'],
                              'len': len(picked_pattern), 
                              'dist_local': picked_child['log_rpif_dist'],
                              'dist_global': picked_child['total_log_rpif_dist'],
                              'n_remaining_candidates': len(candidate_children)}

        return pred, pattern_attributes
    
    def _batch_prefixes(self, nbatches: int):
        nprefixes = len(self.data_test.relevant_prefixes)
        batchsize = math.ceil(nprefixes/nbatches)
        for ndx in range(0, nprefixes, batchsize):
            yield self.data_test.relevant_prefixes[ndx:min(ndx + batchsize, nprefixes)]
    
    def extract_matching_patterns(self, process_stage: int, sequence: list[int], current_size: int = 1, current_tree: dict = None, total_log_rpif_dist: float = 0, global_prob: float = 1) -> dict:
        
        # initial tree lookup
        if current_tree is None:
            try:
                current_tree = self._stage_trees[process_stage]
            except KeyError:
                return []
        
        # look in the tree for the first matching node
        # tree structure is a dictionary with pattern sizes - we only look on the level that equals our current size
        n_matching_elements = math.ceil(current_size/2)

        sequence_to_match = sequence[-n_matching_elements:]
        tree_match_string = ','.join([str(el) for el in sequence_to_match])
        matching_child_idxs = [idx for idx, child in enumerate(current_tree['children'])
                          if ','.join(child['name'].split(',')[:n_matching_elements])==tree_match_string]
        matching_nodes = [current_tree['children'][idx] for idx in matching_child_idxs]

        if any([len(node['name'].split(',')) != current_size for node in matching_nodes]):
            raise ValueError('sizes are not handled correctly here!')
        
        for mn in matching_nodes[:]: # loop over shallow copy of list of matching nodes
            mn['total_log_rpif_dist'] = total_log_rpif_dist + mn['log_rpif_dist']
            mn["global_prob_parent"] = global_prob

            if current_size < self.max_pattern_size:
                matching_nodes.extend(self.extract_matching_patterns(process_stage, sequence, current_size+2, mn, mn['total_log_rpif_dist'], mn["global_prob"]))

        return matching_nodes
    
    def _shared_worker(self, eval_pattern_size: int, selection_method: str, weight: float, 
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
                result = self._batch_predict_activity(eval_pattern_size=eval_pattern_size, 
                                                      prefixes=prefix_array[start:end], 
                                                      selection_method=selection_method, 
                                                      weight=weight, 
                                                      break_after_seq_len=max_seq_len, 
                                                      worker_id=worker_id,
                                                      **kwargs)
            elif task==Task.RTP:
                result = self._batch_predict_sequence(eval_pattern_size=eval_pattern_size, 
                                                      prefixes=prefix_array[start:end], 
                                                      selection_method=selection_method, 
                                                      weight=weight, 
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

class BESTPredictorVanillaPhases():
    """Prediction model using the Hierarchical Central Activity Pattern prediction algorithm.
    BEST is capable of predicting next activities as well as remaining traces for sequences of activities
    """

    def __init__(self, max_pattern_size, process_stage_width_percentage, min_freq, prune_func, include_phases, choice_tracker_keys_nap: list[str] = [], choice_tracker_keys_rtp: list[str] = [], parallelization_lib = 'joblib', **model_args):

        params = {'max_pattern_size':max_pattern_size,
                        'process_stage_width_percentage':process_stage_width_percentage,
                        'min_freq':min_freq,
                        'prune_func':prune_func,
                        'include_phases':include_phases}
        logger.info(f'Initializing prediction model - { {k:v for k,v in params.items()} }')
        if max_pattern_size % 2 == 0 or max_pattern_size <= 1:
            raise ValueError('max_pattern_size must be an odd integer > 1')

        self.max_pattern_size = max_pattern_size
        self.process_stage_width_percentage = process_stage_width_percentage
        self.min_freq = min_freq
        self.prune_func = prune_func
        self.include_phases = include_phases
        
        self._pattern_sizes = [_ for _ in range(1, self.max_pattern_size+1, 2)]
        self._padding_size = int(max_pattern_size/2)+1
        
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.hca_patterns = None

        # length of chosen pattern in prediction tracker
        self.choice_tracker_nap = {key:[] for key in choice_tracker_keys_nap}
        self.choice_tracker_rtp = {key:[] for key in choice_tracker_keys_rtp}
        
        self.parallelization_lib = parallelization_lib
        
    def fit(self) -> None:
        """Fitting the model to X (training data). This involves the pattern generation as well as matching
        of the patterns with their respective children/parents to be able to construct a hierarchical tree of
        central activity patterns
        """
        self.generate_patterns(self.data_train)
        self.find_child_patterns()

        unpruned_nodes = dict()
        pruned_nodes = dict()
        stage_trees = dict()

        logger.info(f'Building pattern tree for {len(self._stages)} stages...')

        for stage in tqdm(self._stages):
            stage_matches = self._matches_per_stage[stage]
            stage_dict_matches = _get_matches_dict(pattern=(), all_matches=stage_matches, max_pattern_size=self.max_pattern_size, min_freq=self.min_freq)
            stage_trees[stage] = stage_dict_matches

            tree_nodes = extract_tree_with_pruning(stage_dict_matches, prune_func=self.prune_func)

            current_unpruned_nodes = {key:node for key, node in tree_nodes.items() if node['pruned'] is False}
            current_pruned_nodes = {key:node for key, node in tree_nodes.items() if node['pruned'] is True}
            pruned_nodes[stage] = current_pruned_nodes
            unpruned_nodes[stage] = current_unpruned_nodes
        
        logger.info('Pattern tree built!')

        self._pruned_nodes = pruned_nodes
        self._unpruned_nodes = unpruned_nodes
        self._stage_trees = stage_trees

    def predict(self, eval_pattern_size: int, task: str, selection_method: str, break_buffer: float, filter_tokens: bool, ncores: int, weight: float|None = None, **pred_args) -> list[list[int]]:
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
        prefix_array = _build_prefix_array_vanilla(self.data_test.relevant_prefixes)
        pred_start_time = time.perf_counter()
        
        # if task==Task.RTP:
        if True:
            max_prefix_len = max([len(prefix['prefix']) for prefix in self.data_test.relevant_prefixes])
            
            max_seq_len = int(break_buffer*max_prefix_len)

            predicted_traces = list()
            if ncores == 1:
                predicted_traces = np.array([lst + [np.nan] * (max_seq_len - len(lst)) for lst in [self._predict_sequence(eval_pattern_size=eval_pattern_size,
                                                                                                                          prefix=row,
                                                                                                                          selection_method=selection_method,
                                                                                                                          weight=weight,
                                                                                                                          break_after_seq_len=max_seq_len) for row in tqdm(prefix_array)]])
            else:
                shm_input, shared_input, shm_result, shared_result = _setup_shared_memory(prefix_array, max_seq_len=max_seq_len)

                # generating index slices according to number of workers (ncores)
                chunksize = len(prefix_array) // ncores
                slices = [
                    (i * chunksize, (i + 1) * chunksize if i < ncores - 1 else len(prefix_array))
                    for i in range(ncores)
                ]

                if self.parallelization_lib == 'joblib':
                    t0 = time.perf_counter()
                    try:
                        Parallel(n_jobs=ncores)(
                            delayed(self._shared_worker)(eval_pattern_size, selection_method, weight, i, task, shm_input.name, prefix_array.shape, prefix_array.dtype, s, e, max_seq_len, 
                            shm_result.name, shared_result.shape, shared_result.dtype,)
                            for i, (s, e) in enumerate(slices)
                        )
                        elapsed = time.perf_counter() - t0
                        predicted_traces = shared_result.copy()
                    finally:
                        for s in [shm_input, shm_result]:
                            s.close()
                            s.unlink()
                            
                elif self.parallelization_lib == 'multiprocessing':
                    procs = [
                        Process(
                            target=self._shared_worker,
                            args=(eval_pattern_size, selection_method, weight, i, task, shm_input.name, prefix_array.shape, prefix_array.dtype, s, e, max_seq_len, 
                            shm_result.name, shared_result.shape, shared_result.dtype,
                                )
                        )
                        for i, (s, e) in enumerate(slices)
                    ]

                    t0 = time.perf_counter()
                    try:
                        for p in procs: p.start()
                        for p in procs: p.join()
                        for p in procs:
                            if p.exitcode != 0:
                                raise RuntimeError(f"Worker {p.pid} failed with exit code {p.exitcode}")
                        elapsed = time.perf_counter() - t0
                        predicted_traces = shared_result.copy()
                    finally:
                        for s in [shm_input, shm_result]:
                            s.close()
                            s.unlink()
                else:
                    raise ValueError('unknown parallelization method')
                
                logger.info(f"Workers done in {elapsed:.3f}s  |  {len(predicted_traces):,} prefixes processed")
                logger.info(f"Total time elapsed: {time.perf_counter() - pred_start_time:.3f}s")

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
        
        # elif task==Task.NAP:
        if True:
            max_seq_len = 1
            
            if ncores==1:
                predicted_activities = np.array([self._predict_activity(eval_pattern_size=eval_pattern_size,
                                                                        prefix=row,
                                                                        selection_method=selection_method,
                                                                        weight=weight) for row in tqdm(prefix_array)])

            else:
                shm_input, shared_input, shm_result, shared_result = _setup_shared_memory(prefix_array, max_seq_len=max_seq_len)

                # generating index slices according to number of workers (ncores)
                chunksize = len(prefix_array) // ncores
                slices = [
                    (i * chunksize, (i + 1) * chunksize if i < ncores - 1 else len(prefix_array))
                    for i in range(ncores)
                ]
                
                if self.parallelization_lib == 'joblib':
                    t0 = time.perf_counter()
                    try:
                        Parallel(n_jobs=ncores)(
                            delayed(self._shared_worker)(eval_pattern_size, selection_method, weight, i, task, shm_input.name, prefix_array.shape, prefix_array.dtype, s, e, max_seq_len, 
                            shm_result.name, shared_result.shape, shared_result.dtype,)
                            for i, (s, e) in enumerate(slices)
                        )
                        elapsed = time.perf_counter() - t0
                        predicted_activities = shared_result.copy()
                    finally:
                        for s in [shm_input, shm_result]:
                            s.close()
                            s.unlink()
                            
                elif self.parallelization_lib == 'multiprocessing':
                    procs = [
                        Process(
                            target=self._shared_worker,
                            args=(eval_pattern_size, selection_method, weight, i, task, shm_input.name, prefix_array.shape, prefix_array.dtype, s, e, max_seq_len, 
                                  shm_result.name, shared_result.shape, shared_result.dtype,
                                  )
                        )
                        for i, (s, e) in enumerate(slices)
                    ]

                    t0 = time.perf_counter()
                    try:
                        for p in procs: p.start()
                        for p in procs: p.join()
                        for p in procs:
                            if p.exitcode != 0:
                                raise RuntimeError(f"Worker {p.pid} failed with exit code {p.exitcode}")
                        elapsed = time.perf_counter() - t0
                        predicted_activities = shared_result.copy()
                    finally:
                        for s in [shm_input, shm_result]:
                            s.close()
                            s.unlink()
                else:
                    raise ValueError('unknown parallelization method')

                logger.info(f"Workers done in {elapsed:.3f}s  |  {len(predicted_activities):,} prefixes processed")
                logger.info(f"Total time elapsed: {time.perf_counter() - pred_start_time:.3f}s")

            pred_duration = time.perf_counter() - convert_start_time

            if ncores > 1:
                predictions = _reconvert_activities_from_shared_mem(predicted_activities)
            else:
                predictions = predicted_activities
            
            self.predicted_activities = predictions

        pred_plus_convert_duration = time.perf_counter() - convert_start_time

        return predictions, pred_duration, pred_plus_convert_duration

    def _predict_sequence(self, eval_pattern_size: int, prefix: list[int], selection_method: str, weight: float|None = None, break_after_seq_len: int = 10e5, verbose: bool = False) -> list[int]:
        """Predicts the remaining activities for a given prefix

        Args:
            prefix (list[int]): A prefix to predict remaining activities for
        
        Returns:
           list[int]: the predicted sequence containing the prefix and the predicted remaining activities
        """    
        predicted_sequence = [el for el in prefix if not np.isnan(el)]
        initial_prefix_len = len(predicted_sequence)

        try:
            last_start = len(predicted_sequence) - predicted_sequence[::-1].index(self.start_activity) - 1
        except ValueError as v_error:
            v_error.args = ('test_sequence contains no start_activities - predict only for non-left-truncated sequences' ,)
            raise

        max_process_stage = max(self._unpruned_nodes.keys())
        min_process_stage = min(self._unpruned_nodes.keys())

        # start prediction loop
        while predicted_sequence[-1] != self.end_activity and len(predicted_sequence) < break_after_seq_len:
            current_process_stage = min(int((len(predicted_sequence[last_start:]) - 1) / self._abs_process_stage_width), max_process_stage)
            
            current_prediction, pattern_attributes = self._pred_for_process_stage(stage=current_process_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
            increase_stage, decrease_stage = current_process_stage + 1, current_process_stage - 1

            while len(current_prediction)==0 and not (increase_stage>max_process_stage and decrease_stage<min_process_stage):
                # look for next higher stage
                increase_stage = min(increase_stage, max_process_stage)
                current_increase_prediction, increase_pattern_attributes = self._pred_for_process_stage(stage=increase_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
                increase_stage += 1

                # look for next lower stage
                decrease_stage = max(decrease_stage, min_process_stage)
                current_decrease_prediction, decrease_pattern_attributes = self._pred_for_process_stage(stage=decrease_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
                decrease_stage -= 1

                increase_prob = increase_pattern_attributes.get('prob_local')
                decrease_prob = decrease_pattern_attributes.get('prob_local')

                # if both find something take the one with the higher probability
                if decrease_prob is not None and increase_prob is not None:
                    if increase_prob > decrease_prob:
                        current_prediction = current_increase_prediction
                        pattern_attributes = increase_pattern_attributes
                    else:
                        current_prediction = current_decrease_prediction
                        pattern_attributes = decrease_pattern_attributes
                elif increase_prob is not None:
                    current_prediction = current_increase_prediction
                    pattern_attributes = increase_pattern_attributes
                elif decrease_prob is not None:
                    current_prediction = current_decrease_prediction
                    pattern_attributes = decrease_pattern_attributes
                else:
                    # nothing has been found in any of the other stages - repeat with different stages
                    pass
        
            if len(current_prediction) == 0:
                if len(predicted_sequence) == len(prefix):
                    logger.debug(f'We did not find any predicted activities for prefix {prefix}')
                else:
                    logger.debug(f'We did not find any further predicted activities for running prediction {predicted_sequence}')
                return predicted_sequence[len(prefix):]
            
            for choice_metric in self.choice_tracker_rtp.keys():
                try:
                    self.choice_tracker_rtp[choice_metric].append(pattern_attributes[choice_metric])
                except KeyError as e:
                    wrong_keys = [key for key in self.choice_tracker_rtp.keys() if key not in pattern_attributes.keys()]
                    raise KeyError(f"Pattern attributes do not match those of the choice trackers - wrong keys: {', '.join(wrong_keys)}") from e
            
            predicted_sequence.extend(current_prediction[1:]) # whole pattern is appended to the prediction

        return predicted_sequence[initial_prefix_len:]
    
    def _batch_predict_sequence(self, eval_pattern_size, prefixes, selection_method, weight, break_after_seq_len, worker_id, **kwargs):
        
        print_indices = [i for i in range(len(prefixes))][::max(1, int(len(prefixes) / 10))][1:] + [len(prefixes) - 1]

        predicted_sequences = np.zeros((len(prefixes), break_after_seq_len))
        predicted_sequences[:] = np.nan

        for prefix_idx, prefix in enumerate(prefixes):
            pred_sequence = self._predict_sequence(eval_pattern_size=eval_pattern_size, 
                                                   prefix=prefix, 
                                                   selection_method=selection_method, 
                                                   weight=weight, 
                                                   break_after_seq_len=break_after_seq_len, 
                                                   **kwargs)
            
            predicted_sequences[prefix_idx, :len(pred_sequence)] = pred_sequence

            if prefix_idx in print_indices:
                logger.info(f"worker {worker_id}: {prefix_idx} prefixes completed ({100*(prefix_idx+1)/len(prefixes):.2f}%)")

        return predicted_sequences


    def _predict_activity(self, eval_pattern_size: int, prefix: list[int], selection_method: str, weight: float|None = None, break_after_seq_len: int = 1, verbose: bool = False) -> int:

        """Predicts the next activity for a given sequence

        Args:
            prefix (list[int]): A prefix to predict remaining activities for
        
        Returns:
            int: the predicted activity
        """

        if break_after_seq_len != 1:
            raise ValueError(f"break_after_seq_len should not be overridden - overridden with: {break_after_seq_len}")
    
        predicted_sequence = [el for el in prefix if not np.isnan(el)]

        try:
            last_start = len(predicted_sequence) - predicted_sequence[::-1].index(self.start_activity) - 1
        except ValueError as v_error:
            v_error.args = ('test_sequence contains no start_activities - predict only for non-left-truncated sequences' ,)
            raise

        max_process_stage = max(self._unpruned_nodes.keys())
        min_process_stage = min(self._unpruned_nodes.keys())

        # start prediction loop
        if predicted_sequence[-1] != self.end_activity:
            current_process_stage = min(int((len(predicted_sequence[last_start:]) - 1) / self._abs_process_stage_width), max_process_stage)
            
            current_prediction, pattern_attributes = self._pred_for_process_stage(stage=current_process_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
            increase_stage, decrease_stage = current_process_stage + 1, current_process_stage - 1

            while len(current_prediction)==0 and not (increase_stage>max_process_stage and decrease_stage<min_process_stage):
                # look for next higher stage
                increase_stage = min(increase_stage, max_process_stage)
                current_increase_prediction, increase_pattern_attributes = self._pred_for_process_stage(stage=increase_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
                increase_stage += 1

                # look for next lower stage
                decrease_stage = max(decrease_stage, min_process_stage)
                current_decrease_prediction, decrease_pattern_attributes = self._pred_for_process_stage(stage=decrease_stage, eval_pattern_size=eval_pattern_size, sequence=predicted_sequence, selection_method=selection_method, weight=weight, verbose=verbose)
                decrease_stage -= 1

                increase_prob = increase_pattern_attributes.get('prob_local')
                decrease_prob = decrease_pattern_attributes.get('prob_local')

                # if both find something take the one with the higher probability
                if decrease_prob is not None and increase_prob is not None:
                    if increase_prob > decrease_prob:
                        current_prediction = current_increase_prediction
                        pattern_attributes = increase_pattern_attributes
                    else:
                        current_prediction = current_decrease_prediction
                        pattern_attributes = decrease_pattern_attributes
                elif increase_prob is not None:
                    current_prediction = current_increase_prediction
                    pattern_attributes = increase_pattern_attributes
                elif decrease_prob is not None:
                    current_prediction = current_decrease_prediction
                    pattern_attributes = decrease_pattern_attributes
                else:
                    # nothing has been found in any of the other stages - repeat with different stages
                    pass
        
            if len(current_prediction) == 0:
                if verbose:
                    print("WE DID NOT FIND ANY SUITABLE PREDICTION FOR THE CURRENT SEQUENCE ANYWHERE")
                return
            
            for choice_metric in self.choice_tracker_nap.keys():
                try:
                    self.choice_tracker_nap[choice_metric].append(pattern_attributes[choice_metric])
                except KeyError as e:
                    wrong_keys = [key for key in self.choice_tracker_nap.keys() if key not in pattern_attributes.keys()]
                    raise KeyError(f"Pattern attributes do not match those of the choice trackers - wrong keys: {', '.join(wrong_keys)}") from e
        return current_prediction[1]

    def _batch_predict_activity(self, eval_pattern_size, prefixes, selection_method, weight, break_after_seq_len, worker_id, **kwargs):

        print_indices = [i for i in range(len(prefixes))][::max(1, int(len(prefixes) / 10))][1:] + [len(prefixes) - 1]

        predicted_activities = np.zeros((len(prefixes), 1))
        predicted_activities[:] = np.nan

        for prefix_idx, prefix in enumerate(prefixes):
            pred_activity = self._predict_activity(eval_pattern_size=eval_pattern_size, 
                                                   prefix=prefix, 
                                                   selection_method=selection_method, 
                                                   weight=weight, 
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

    def prepare_train(self):
        
        logger.info('Preparing training data...')
        if self.data_train is None:
            raise ValueError('data not found - make sure to load train and test data with load_data()')
        
        train_case_ids = self.data_train.data[self.data_train.case_identifier].unique().tolist()
        
        # get phases from annotated dataset
        self.data_train.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_train.case_identifier].isin(train_case_ids)]['phase'].reset_index(drop=True)
        
        try:
            start_activity = self.data_train.start_activity
            end_activity = self.data_train.end_activity
        except:
            start_activity = 'START'
            end_activity = 'END'
        
        self.train_max_trace_len = self.data_train._get_max_trace_len()
        self.data_train.pad_columns(cols_to_pad=[self.data_train.activity_identifier],
                                    n_pad=self._padding_size - 1, # we already have a 1 pre/post padding from HMM
                                    forward_pad=end_activity,
                                    backward_pad=start_activity)
        # forward and backward fill timestamp column
        self.data_train.data[self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].ffill()
        self.data_train.data[self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].bfill()

        # self.data_train.encode_activities()
        self.data_train.extract_traces(columns=[self.data_train.activity_identifier, 'phase'])
        self.start_activity = self.data_train.start_activity
        self.end_activity = self.data_train.end_activity

        self.data_train.generate_prefixes(include_phases=self.include_phases)
        self.data_train.pick_relevant_prefixes()

        self.max_prefix_len = max([len(prefix['prefix']) for prefix in self.data_train.relevant_prefixes])

        logger.info('Training data prepared!')

    def prepare_val(self, act_encoder: LabelEncoder|None = None, filter_sequences: bool = True):
        
        logger.info('Preparing validation data...')
        if self.data_val is None:
            raise ValueError('data not found - make sure to load train, validation and test data with load_data()')
        
        val_case_ids = self.data_val.data[self.data_val.case_identifier].unique().tolist()
        
        # get phases from annotated dataset
        self.data_val.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_val.case_identifier].isin(val_case_ids)]['phase'].reset_index(drop=True)
        
        try:
            # we access START/END encoding from training here
            start_activity = self.start_activity
            end_activity = self.end_activity
        except:
            start_activity = 'START'
            end_activity = 'END'
        
        self.data_val.pad_columns(cols_to_pad=[self.data_val.activity_identifier],
                                  n_pad=self._padding_size - 1, # we already have a 1 pre/post padding from HMM
                                  forward_pad=end_activity,
                                  backward_pad=start_activity)
        # forward and backward fill timestamp column
        self.data_val.data[self.data_val.timestamp_identifier] = self.data_val.data.groupby(self.data_val.case_identifier)[self.data_val.timestamp_identifier].ffill()
        self.data_val.data[self.data_val.timestamp_identifier] = self.data_val.data.groupby(self.data_val.case_identifier)[self.data_val.timestamp_identifier].bfill()

        # self.data_test.encode_activities(act_encoder=act_encoder)
        self.data_val.extract_traces(columns=[self.data_val.activity_identifier, 'phase'])

        self.data_val.generate_prefixes(include_phases=self.include_phases)
        self.data_val.pick_relevant_prefixes()

        self.data_val.generate_full_sequences(filter_sequences=filter_sequences)
        self.data_val.generate_full_future_sequences(filter_sequences=filter_sequences)
        self.data_val.generate_next_activities()
        logger.info('Validation data prepared!')
    
    def prepare_test(self, act_encoder: LabelEncoder|None = None, filter_sequences: bool = True):
        
        logger.info('Preparing test data...')
        if self.data_test is None:
            raise ValueError('data not found - make sure to load train and test data with load_data()')
        
        test_case_ids = self.data_test.data[self.data_test.case_identifier].unique().tolist()
        
        # get phases from annotated dataset
        self.data_test.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_test.case_identifier].isin(test_case_ids)]['phase'].reset_index(drop=True)
        
        try:
            # we access START/END encoding from training here
            start_activity = self.start_activity
            end_activity = self.end_activity
        except:
            start_activity = 'START'
            end_activity = 'END'
        
        self.data_test.pad_columns(cols_to_pad=[self.data_test.activity_identifier],
                                   n_pad=self._padding_size - 1, # we already have a 1 pre/post padding from HMM
                                   forward_pad=end_activity,
                                   backward_pad=start_activity)
        # forward and backward fill timestamp column
        self.data_test.data[self.data_test.timestamp_identifier] = self.data_test.data.groupby(self.data_test.case_identifier)[self.data_test.timestamp_identifier].ffill()
        self.data_test.data[self.data_test.timestamp_identifier] = self.data_test.data.groupby(self.data_test.case_identifier)[self.data_test.timestamp_identifier].bfill()
        
        # self.data_test.encode_activities(act_encoder=act_encoder)
        self.data_test.extract_traces(columns=[self.data_test.activity_identifier, 'phase'])

        self.data_test.generate_prefixes(include_phases=self.include_phases)
        self.data_test.pick_relevant_prefixes()

        self.data_test.generate_full_sequences(filter_sequences=filter_sequences)
        self.data_test.generate_full_future_sequences(filter_sequences=filter_sequences)
        self.data_test.generate_next_activities()
        logger.info('Test data prepared!')

    def generate_patterns(self, X: SequenceData):
        """Generates sequential patterns of activities inside traces by process stage. The generated patterns are
        given as a dict with pattern sizes as keys. Each pattern is a tuple of two elements where the first element is
        the pattern as a list of ints and the second element is the respective process stage
        (with the absolute width being specified by self.process_stage_width_percentage)

        Args:
            X (SequenceData): The SequenceData object we generate the sequential patterns for. This has to be prepared
            such that it has generated traces as class attribute (SequenceData.prepare_train())
        """
        logger.info(f'Generating patterns from trace data...')
        patterns_by_size = dict()
        patterns = list()
        
        abs_process_stage_width = int((self.train_max_trace_len + 2)*self.process_stage_width_percentage) + 1
        self._abs_process_stage_width = abs_process_stage_width

        for pattern_size in self._pattern_sizes:
            current_size_patterns = list()
            for trace in X.traces:
                act_trace_to_process = trace[X.activity_identifier]
                phase_trace_to_process = trace['phase']
                trace_index_start = self._padding_size-int(pattern_size/2)-1
                trace_index_end = len(act_trace_to_process)-self._padding_size+int(pattern_size/2)+1
                act_trace_to_process = act_trace_to_process[trace_index_start:trace_index_end] # ensure no patterns of only END and START tokens
                if self.include_phases:
                    phase_trace_to_process = phase_trace_to_process[trace_index_start:trace_index_end]
                else:
                    phase_trace_to_process = [0] * len(act_trace_to_process)
                
                current_size_patterns.extend(make_pattern_stage_tuples_vanilla(act_trace_to_process, pattern_size=pattern_size,
                                                   pad_short_seqs=True, include_phases=self.include_phases, process_stage_width=self._abs_process_stage_width))
                
            patterns_by_size[pattern_size] = current_size_patterns
            patterns.extend(current_size_patterns)
        
        self._hca_patterns_by_size = patterns_by_size
        self.hca_patterns = patterns

        pattern_strings = [(','.join([str(el) for el in pattern[0]]), pattern[1]) for pattern in patterns]
        pattern_freqs_by_stage = calc_stage_wise_freqs_vanilla(pattern_strings)
        self._hca_pattern_freqs_by_stage = pattern_freqs_by_stage

        logger.info(f'Pattern generation completed!')

    def find_child_patterns(self):
        
        logger.info(f'Searching for child patterns...')
        all_stage_values = [pattern[1] for pattern in self.hca_patterns]
        
        stages = range(min(all_stage_values), max(all_stage_values)+1)
        self._stages = stages

        original_sizes = [0] + self._pattern_sizes[:-1]
        extended_sizes = self._pattern_sizes[:]

        matches_per_stage = dict()


        for stage in stages:
            unique_ext_patterns = list()
            all_matches = dict()
            for orig_size, ext_size in zip(original_sizes, extended_sizes):

                if len(unique_ext_patterns) == 0:
                    if orig_size == 0:
                        orig_patterns = [[]]
                    else:
                        orig_patterns = [pattern[0] for pattern in self._hca_patterns_by_size[orig_size] 
                                        if pattern[1]==stage]
                    
                    orig_pattern_counts = dict(Counter(map(tuple, orig_patterns)))
                    unique_orig_patterns = {pattern:{'prob':1,
                                                     'global_prob':freq/sum(orig_pattern_counts.values()),
                                                     'freq':freq}
                                            for pattern, freq in orig_pattern_counts.items()}
                else:
                    unique_orig_patterns = unique_ext_patterns.copy()

                ext_patterns = [pattern[0] for pattern in self._hca_patterns_by_size[ext_size] 
                                if pattern[1]==stage]
                ext_pattern_counts = dict(Counter(map(tuple, ext_patterns)))
                unique_ext_patterns = {pattern:{'prob':float(),
                                                'global_prob':freq/sum(ext_pattern_counts.values()),
                                                'freq':freq}
                                       for pattern, freq in ext_pattern_counts.items()}
                
                # get matching extended patterns for each original pattern
                size_matches = dict()
                for orig, orig_attrs in unique_orig_patterns.items():
                    matches = dict()
                    # match_counts = list()
                    if orig == ():
                    # if orig == []:
                        pattern_string = ''
                    else:
                        pattern_string = ','.join([str(el) for el in orig])
                    
                    if pattern_string not in all_matches.keys():
                        for ext, ext_attrs in unique_ext_patterns.items():
                            if orig == ext[1:-1]:
                                matches[ext] = ext_attrs
                        # scale probabilities to unit sum (for the matching extended patterns)
                        probs_to_rescale = [val['global_prob'] for val in matches.values()]
                        rescaled_probs = _rescale_probs(probs_to_rescale)
                        for idx, match_attrs in enumerate(matches.values()):
                            match_attrs['prob'] = rescaled_probs[idx]
                        
                        for ext, ext_attrs in matches.items():
                            unique_ext_patterns[ext] = ext_attrs
                        size_matches[pattern_string] = {'prob':orig_attrs['prob'], 
                                                        'global_prob':orig_attrs['global_prob'], 
                                                        'freq':orig_attrs['freq'], 
                                                        'matches':matches}
                all_matches[orig_size] = size_matches
            matches_per_stage[stage] = all_matches
        
        self._matches_per_stage = matches_per_stage
        logger.info(f'Child pattern search completed!')

    def _pred_for_process_stage(self, eval_pattern_size: int, stage: int, sequence: list[int], selection_method: str, weight: float|None = None, verbose: bool = False) -> list[int]:
        
        if self.prune_func is not None:
        
            current_important_patterns = self._unpruned_nodes[stage]

            empty_key = [key for key, val in current_important_patterns.items() if val['name']=='']
            if empty_key:
                if verbose:
                    print('i deleted the empty key, sir :)')
                current_important_patterns.pop(empty_key[0])

            center_activity = sequence[-1]
            # check for children of center activity
            children = {key:val 
                        for key, val in current_important_patterns.items() 
                        if _get_pattern_center(val) == center_activity}

            # we exclude single activity children here (because we check for matches with the center activity (not with the left part))
            non_atomic_patterns = [key for key, child in children.items() if len(child['name'].split(',')) > 1]
            children = {child_key: children[child_key] for child_key in non_atomic_patterns}

            if verbose:
                print(children)

            applying_children = {key:val 
                                    for key, val in children.items() 
                                    if _child_matches_with_sequence(val, sequence)}
            
            if len(applying_children)==0:
                return [], 0
            
            probs = [val['prob'] for val in applying_children.values()]
            dists = [val['total_log_rpif_dist'] for val in applying_children.values()]
            lens_children = [len(val['name'].split(',')) for val in applying_children.values()]

            max_prob = max(probs)
            argmax_prob_indices = [idx for idx, p in enumerate(probs) if p==max_prob]
            argmax_children = [[int(act) for act in [val['name'] for val in applying_children.values()][pick].split(',')] for pick in argmax_prob_indices]
            argmax_children_dists = [dists[idx] for idx in argmax_prob_indices]

            min_dist = min(argmax_children_dists)
            argmin_dist_indices = [idx for idx, d in enumerate([dists[i] for i in argmax_prob_indices]) if d==min_dist]
            min_dist_pick = np.argmin(argmax_children_dists)        
        else:
            
            all_applying_children = self.extract_matching_patterns(stage, sequence)

            # kick single activity patterns from extracted matching patterns
            all_applying_children = all_applying_children[1:]
            
            lens_all_children = [len(p['name'].split(',')) for p in all_applying_children]
            applying_children = [c for c, c_len in zip(all_applying_children, lens_all_children) if c_len <= eval_pattern_size]

            if len(applying_children)==0:
                return [], {}

            # get probs, dists and lens
            probs = [p['prob'] for p in applying_children]
            global_probs = [p['global_prob'] for p in applying_children]
            dists = [p['total_log_rpif_dist'] for p in applying_children]
            lens_children = [len(p['name'].split(',')) for p in applying_children]

            # for weighted distance
            global_probs_parent = [p["global_prob_parent"] for p in applying_children]
            local_beds = [p["log_rpif_dist"] for p in applying_children]
            global_beds_parents = [p["total_log_rpif_dist"] - p["log_rpif_dist"] for p in applying_children]

            # DIFFERENT SELECTION METHODS:
            # PROB_LEN_DIST: 1. min(local BED) 2. max pattern length 3. min (global BED) - what we used in the BPM paper
            # PROB_DIST: 1. min(local BED) 2. min (global BED)
            # WEIGHTED_DIST: weighted distance metric between local BED and global BED (of the respective parent patterns)
            # WEIGHTED_DIST_LEN: 1. weighted distance metric from WEIGHTED_DIST 2. max pattern length
            # WEIGHTED_PROBS: weighted probabilty (sum) between conditional extension probability and global occurrence prob of the parent
            # WEIGHTED_PROBS_LEN: 1. weighted probabilty (sum) from WEIGHTED_PROBS 2. max pattern length
            # MAX_JOINT_PROB: joint probability (product) between conditional extension probability and global occurrence prob of the parent
            # MAX_JOINT_PROB_LEN: 1. joint probability (product) from MAX_JOINT_PROB 2. max pattern length 

            if selection_method == 'PROB_LEN_DIST':
                # 1 - filter for max prob (minimal local BED)
                max_prob = max(probs)
                argmax_prob_indices = [idx for idx, p in enumerate(probs) if p==max_prob]

                # 2 - filter for maximal match length (longest patterns)
                argmax_prob_children_lens = [lens_children[idx] for idx in argmax_prob_indices]
                max_len = max(argmax_prob_children_lens)
                argmax_prob_argmax_len_indices = [idx for idx, l in enumerate(lens_children) if l==max_len and idx in argmax_prob_indices]

                # 3 - filter for minimal global BED
                argmax_prob_argmax_len_dists = [dists[idx] for idx in argmax_prob_argmax_len_indices]
                min_dist_max_len = min(argmax_prob_argmax_len_dists)
                argmax_prob_argmax_len_argmin_dist_indices = [idx for idx, d in enumerate(dists) if d==min_dist_max_len and idx in argmax_prob_argmax_len_indices]
            
                candidate_children = [[p for p in applying_children][pick] for pick in argmax_prob_argmax_len_argmin_dist_indices] # with max prob - max len - min dist

                if len(argmax_prob_argmax_len_argmin_dist_indices) > 1:
                    # pattern_lens = [len(c) for c in candidate_children]
                    pattern_lens = [len([act for act in c['name'].split(',')]) for c in candidate_children]
                    logger.debug(f'we have {len(argmax_prob_argmax_len_argmin_dist_indices)} patterns with same cond prob of {max_prob}, len of {max_len} and min distance of {min_dist_max_len:.4f} - choosing randomly')
                    picked_child = candidate_children[np.random.choice(range(0, len(candidate_children)))]
                else:
                    picked_child = candidate_children[0]
            
            elif selection_method == 'PROB_DIST':
                # 1 - filter for max prob (minimal local BED)
                max_prob = max(probs)
                argmax_prob_indices = [idx for idx, p in enumerate(probs) if p==max_prob]

                # 2 - filter for minimal global BED
                argmax_prob_dists = [dists[idx] for idx in argmax_prob_indices]
                min_dist = min(argmax_prob_dists)
                argmax_prob_argmax_len_dists = [dists[idx] for idx in argmax_prob_argmax_len_indices]
                argmax_prob_argmin_dist_indices = [idx for idx, d in enumerate(dists) if d==min_dist and idx in argmax_prob_indices]

                candidate_children = [[p for p in applying_children][pick] for pick in argmax_prob_argmin_dist_indices]

                if len(argmax_prob_argmin_dist_indices) > 1:
                    # pattern_lens = [len(c) for c in candidate_children]
                    pattern_lens = [len([act for act in c['name'].split(',')]) for c in candidate_children]
                    logger.debug(f'we have {len(argmax_prob_argmin_dist_indices)} patterns with same cond prob of {max_prob} and min distance of {min_dist:.4f} - they have lens {pattern_lens} - choosing randomly')
                    picked_child = candidate_children[np.random.choice(range(0, len(candidate_children)))]
                else:
                    picked_child = candidate_children[0]

            elif selection_method == "WEIGHTED_DIST":
                
                if weight is None:
                    raise ValueError(f"Need to set a weight for selection method {selection_method}")

                # calculate weighted distance
                weighted_dists = [weight * lb + (1 - weight) * gbp for lb, gbp in zip(local_beds, global_beds_parents)]
                min_weighted_dist = min(weighted_dists)
                argmin_weighted_dist_indices = [idx for idx, wd in enumerate(weighted_dists) if wd == min_weighted_dist]
                
                candidate_children = [[p for p in applying_children][pick] for pick in argmin_weighted_dist_indices]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            elif selection_method == "WEIGHTED_PROBS":

                if weight is None:
                    raise ValueError(f"Need to set a weight for selection method {selection_method}")

                # calculate weighted probs
                weighted_probs = [weight * bed + (1 - weight) * gpp for bed, gpp in zip(probs, global_probs_parent)]
                max_weighted_probs = max(weighted_probs)
                argmax_weighted_probs = [idx for idx, wd in enumerate(weighted_probs) if wd == max_weighted_probs]
                candidate_children = [[p for p in applying_children][pick]for pick in argmax_weighted_probs]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            elif selection_method == "WEIGHTED_PROBS_LEN":
                
                if weight is None:
                    raise ValueError(f"Need to set a weight for selection method {selection_method}")

                # 1 - calculate weighted probs
                weighted_probs = [weight * bed + (1 - weight) * gpp for bed, gpp in zip(probs, global_probs_parent)]
                max_weighted_probs = max(weighted_probs)
                argmax_weighted_probs_indices = [idx for idx, wd in enumerate(weighted_probs) if wd == max_weighted_probs]

                # 2 - filter for maximal match length (longest patterns)
                argmax_weighted_probs_lens = [lens_children[idx] for idx in argmax_weighted_probs_indices]
                max_len = max(argmax_weighted_probs_lens)
                argmax_weighted_probs_argmax_len_indices = [idx for idx, l in enumerate(lens_children) if l==max_len and idx in argmax_weighted_probs_indices]
                
                candidate_children = [[p for p in applying_children][pick] for pick in argmax_weighted_probs_argmax_len_indices]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            elif selection_method == "WEIGHTED_DIST_LEN":
                
                if weight is None:
                    raise ValueError(f"Need to set a weight for selection method {selection_method}")

                # 1 - calculate weighted distance
                weighted_dists = [weight * lb + (1 - weight) * gbp for lb, gbp in zip(local_beds, global_beds_parents)]
                min_weighted_dist = min(weighted_dists)
                argmin_weighted_dist_indices = [idx for idx, wd in enumerate(weighted_dists) if wd == min_weighted_dist]

                # 2 - filter for maximal match length (longest patterns)
                argmin_weighted_dist_lens = [lens_children[idx] for idx in argmin_weighted_dist_indices]
                max_len = max(argmin_weighted_dist_lens)
                argmin_weighted_dist_argmax_len_indices = [idx for idx, l in enumerate(lens_children) if l==max_len and idx in argmin_weighted_dist_indices]
                
                candidate_children = [[p for p in applying_children][pick] for pick in argmin_weighted_dist_argmax_len_indices]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            elif selection_method == "MAX_JOINT_PROB":

                # calculate joint probs
                joint_probs = [cp*gpp for cp, gpp in zip(probs, global_probs_parent)]
                max_joint_probs = max(joint_probs)
                argmax_joint_probs = [idx for idx, jp in enumerate(joint_probs) if jp == max_joint_probs]

                candidate_children = [[p for p in applying_children][pick] for pick in argmax_joint_probs]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            elif selection_method == "MAX_JOINT_PROB_LEN":

                # 1 - calculate join probability
                joint_probs = [cp*gpp for cp, gpp in zip(probs, global_probs_parent)]
                max_joint_probs = max(joint_probs)
                argmax_joint_probs_indices = [idx for idx, jp in enumerate(joint_probs) if jp == max_joint_probs]

                # 2 - filter for maximal match length (longest patterns)
                argmax_joint_probs_lens = [lens_children[idx] for idx in argmax_joint_probs_indices]
                max_len = max(argmax_joint_probs_lens)
                argmax_joint_probs_argmax_len_indices = [idx for idx, l in enumerate(lens_children) if l==max_len and idx in argmax_joint_probs_indices]
                
                candidate_children = [[p for p in applying_children][pick] for pick in argmax_joint_probs_argmax_len_indices]

                if len(candidate_children) > 1:
                    picked_child = candidate_children[
                        np.random.choice(range(0, len(candidate_children)))
                    ]
                else:
                    picked_child = candidate_children[0]
            else:
                raise NotImplementedError(f"Chosen selection method {selection_method} not implemented!")
        
        for prob, length in zip(probs, lens_children):

            if verbose:
                print(f"child prob: {prob}, child len: {length}")

        try:
            picked_pattern = [int(act) for act in picked_child['name'].split(',')]
        except ValueError:
            picked_pattern = [float(act) for act in picked_child['name'].split(',')]

        pred = picked_pattern[math.floor(len(picked_pattern)/2):]

        pattern_attributes = {'prob_local': picked_child['prob'], 
                              'prob_global': picked_child['global_prob_parent'],
                              'len': len(picked_pattern), 
                              'dist_local': picked_child['log_rpif_dist'],
                              'dist_global': picked_child['total_log_rpif_dist'],
                              'n_remaining_candidates': len(candidate_children)}

        return pred, pattern_attributes
    
    def _batch_prefixes(self, nbatches: int):
        nprefixes = len(self.data_test.relevant_prefixes)
        batchsize = math.ceil(nprefixes/nbatches)
        for ndx in range(0, nprefixes, batchsize):
            yield self.data_test.relevant_prefixes[ndx:min(ndx + batchsize, nprefixes)]
    
    def extract_matching_patterns(self, process_stage: int, sequence: list[int], current_size: int = 1, current_tree: dict = None, total_log_rpif_dist: float = 0, global_prob: float = 1) -> dict:
        
        # initial tree lookup
        if current_tree is None:
            current_tree = self._stage_trees[process_stage]
        
        # look in the tree for the first matching node
        # tree structure is a dictionary with pattern sizes - we only look on the level that equals our current size
        n_matching_elements = math.ceil(current_size/2)

        sequence_to_match = sequence[-n_matching_elements:]
        tree_match_string = ','.join([str(el) for el in sequence_to_match])
        matching_child_idxs = [idx for idx, child in enumerate(current_tree['children'])
                          if ','.join(child['name'].split(',')[:n_matching_elements])==tree_match_string]
        matching_nodes = [current_tree['children'][idx] for idx in matching_child_idxs]

        if any([len(node['name'].split(',')) != current_size for node in matching_nodes]):
            raise ValueError('sizes are not handled correctly here!')
        
        for mn in matching_nodes[:]: # loop over shallow copy of list of matching nodes
            mn['total_log_rpif_dist'] = total_log_rpif_dist + mn['log_rpif_dist']
            mn["global_prob_parent"] = global_prob

            if current_size < self.max_pattern_size:
                matching_nodes.extend(self.extract_matching_patterns(process_stage, sequence, current_size+2, mn, mn['total_log_rpif_dist'], mn["global_prob"]))

        return matching_nodes
    
    def _shared_worker(self, eval_pattern_size: int, selection_method: str, weight: float, 
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
                result = self._batch_predict_activity(eval_pattern_size=eval_pattern_size, 
                                                      prefixes=prefix_array[start:end], 
                                                      selection_method=selection_method, 
                                                      weight=weight, 
                                                      break_after_seq_len=max_seq_len, 
                                                      worker_id=worker_id,
                                                      **kwargs)
            elif task==Task.RTP:
                result = self._batch_predict_sequence(eval_pattern_size=eval_pattern_size, 
                                                      prefixes=prefix_array[start:end], 
                                                      selection_method=selection_method, 
                                                      weight=weight, 
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

def _get_matches_dict(pattern: tuple[int], all_matches: dict, max_pattern_size: int, min_k: int = None, max_k: int = None, min_freq: float = None) -> dict:
    """Recursive search of pattern matches given a starting pattern. For each match, we calculate the conditional probability
    (occurrence probability given the parent pattern). In each call we identify the matching patterns and execute the search
    for each of the matches until we find no matches in the next bigger pattern size.

    Args:
        pattern (tuple[int]): the base pattern we mine the recursive match dictionary for.
        all_matches (dict): dictionary of patterns that holds patterns for different pattern sizes.
        The pattern sizes are the first level of the dictionary, actual patterns are on the second level, respectively, 
        with their corresponding matches as values (list of matching patterns that are in format list[int]).
        max_pattern_size (int): model parameter - maximum pattern size mined in the model
        min_k (int, optional): model parameter - minimum patterns to keep per match. Defaults to None.
        max_k (int, optional): model parameter - maximum patterns to keep per match. Cannot be combined with min_freq. Defaults to None.
        min_freq (float, optional): model parameter - minimum frequence for patterns to keep per match.
        Cannot be combined with max_k. Can be combined with min_k. Defaults to None.

    Returns:
        dict: recursive dict of matches starting at the given pattern. matches found are from next bigger pattern size.
    """
    assert min_freq is None or (min_freq > 0 and min_freq <= 1), 'pick min_freq between 0 and 1'
    assert not (min_freq is not None and max_k is not None), 'specify either max_k for top-k selection or min_freq for minimal relative frequency selection'

    matches = dict()

    current_pattern_size = len(pattern)

    if current_pattern_size > 1:
        parent_pattern = pattern[1:-1]
    else:
        parent_pattern = None
    
    # identify matches
    if current_pattern_size <= max_pattern_size - 2:
        pattern_string = ','.join([str(el) for el in pattern])
        current_matches = all_matches[current_pattern_size][pattern_string]['matches']
        cond_prob = all_matches[current_pattern_size][pattern_string]['prob']
        global_prob = all_matches[current_pattern_size][pattern_string]['global_prob']
        freq = all_matches[current_pattern_size][pattern_string]['freq']
    else:
        parent_pattern_string = ','.join([str(el) for el in parent_pattern])
        cond_prob = all_matches[current_pattern_size-2][parent_pattern_string]['matches'][pattern]['prob']
        global_prob = all_matches[current_pattern_size-2][parent_pattern_string]['matches'][pattern]['global_prob']
        freq = all_matches[current_pattern_size-2][parent_pattern_string]['matches'][pattern]['freq']
        current_matches = dict()

    if max_k:
        current_matches = {key:val for idx, (key, val) in enumerate(current_matches.items()) if idx < max_k}

    if min_freq:
        if min_k:
            current_matches = {key:val for idx, (key, val) in enumerate(current_matches.items()) if val['prob'] >= min_freq or idx < min_k}
        else:
            current_matches = {key:val for key, val in current_matches.items() if val['prob'] >= min_freq}
        
    matches['name'] = ','.join([str(item) for item in pattern])
    matches['prob'] = cond_prob
    matches['global_prob'] = global_prob
    matches['freq'] = freq
    if parent_pattern:
        len_diff = len(pattern) - len(parent_pattern)
        matches['len_diff'] = len_diff
        matches['rpif_dist'] = math.exp(0.5 * len_diff) / cond_prob
        matches['log_rpif_dist'] = math.log(matches['rpif_dist'])
    else:
        matches['len_diff'] = None
        matches['rpif_dist'] = 1 / cond_prob
        matches['log_rpif_dist'] = math.log(matches['rpif_dist'])
    matches['children'] = list()

    for match in current_matches.keys():
        children = _get_matches_dict(match, all_matches, max_pattern_size, min_k, max_k, min_freq)
        if len(children) > 0:
            matches['children'].append(children)

    return matches

def _get_matches_from_dict(match_dict: dict) -> int:
    matches = list()
    children = match_dict['children']
    if len(children) > 0:
        for child in children:
            matches.append(child['name'])
            matches.extend(_get_matches_from_dict(child))

    return matches

def calc_stage_wise_freqs(pattern_stage_tuples: tuple[str,int]) -> pd.DataFrame:
    all_stage_values = [nst[1] for nst in pattern_stage_tuples]
    stages = set(all_stage_values)

    stage_wise_freqs = dict()

    for stage in stages:
        stage_patterns = [pst for pst in pattern_stage_tuples if pst[1]==stage]
        stage_pattern_freqs = pd.DataFrame({'pattern':[sn[0] for sn in stage_patterns], 
                                          'counter':[1]*len(stage_patterns)}).groupby('pattern').apply(lambda x: x['counter'].sum()).reset_index(name='freq')
        stage_wise_freqs[stage] = stage_pattern_freqs

    return stage_wise_freqs

def calc_stage_wise_freqs_vanilla(pattern_stage_tuples: tuple[str,int]) -> pd.DataFrame:
    all_stage_values = [nst[1] for nst in pattern_stage_tuples]
    stages = range(min(all_stage_values), max(all_stage_values)+1)

    stage_wise_freqs = dict()

    for stage in stages:
        stage_patterns = [pst for pst in pattern_stage_tuples if pst[1]==stage]
        stage_pattern_freqs = pd.DataFrame({'pattern':[sn[0] for sn in stage_patterns], 
                                          'counter':[1]*len(stage_patterns)}).groupby('pattern').apply(lambda x: x['counter'].sum()).reset_index(name='freq')
        stage_wise_freqs[stage] = stage_pattern_freqs

    return stage_wise_freqs

def make_pattern_stage_tuples(vector: tuple[list[int|float], list[int|float]], pattern_size: int, pad_short_seqs: bool, include_phases: bool) -> tuple[list[int], int]:
    # TODO
    # include doc string

    assert pattern_size % 2 != 0, 'pick an uneven pattern size'

    act_vector = vector[0]
    phase_vector = vector[1]

    if pad_short_seqs and pattern_size > len(act_vector):
        # this transformation makes sense if we pad the sequences with END tokens
        # if no padding is being performed beforehand this introduces non-existent 
        # self-loops of the last element in the shorter sequences
        act_vector = act_vector + [act_vector[-1]] * (pattern_size - len(act_vector))
        phase_vector = phase_vector + [phase_vector[-1]] * (pattern_size - len(phase_vector))
    
    assert pattern_size <= len(act_vector), 'pattern size cannot be bigger than the vector - for bypass set pad_short_seqs=True'
    act_patterns = [act_vector[idx:idx+pattern_size] for idx in range(0, len(act_vector)-(pattern_size-1))]
    process_stages = [phase_vector[idx] for idx in range(int(pattern_size/2), len(act_vector)-(int(pattern_size/2)))]

    if include_phases:
        pattern_stage_tuples = [(p, stage) for p, stage in zip(act_patterns, process_stages)]
    else:
        pattern_stage_tuples = [(p, 0) for p in act_patterns]

    return pattern_stage_tuples

def make_pattern_stage_tuples_vanilla(vector: list[int], pattern_size: int, pad_short_seqs: bool, process_stage_width: int, include_phases: bool) -> tuple[list[int], int]:
    
    assert pattern_size % 2 != 0, 'pick an uneven pattern size'

    if pad_short_seqs and pattern_size > len(vector):
        # this transformation makes sense if we pad the sequences with END tokens
        # if no padding is being performed beforehand this introduces non-existent 
        # self-loops of the last element in the shorter sequences
        vector = vector + [vector[-1]] * (pattern_size - len(vector))
    
    assert pattern_size <= len(vector), 'pattern size cannot be bigger than the vector - for bypass set pad_short_seqs=True'
    patterns = [vector[idx:idx+pattern_size] for idx in range(0, len(vector)-(pattern_size-1))]

    process_stages = [int(act_idx/process_stage_width) for act_idx in range(0, len(vector)-(pattern_size-1))]
    pattern_stage_tuples = [(p, stage) for p, stage in zip(patterns, process_stages)]

    return pattern_stage_tuples

def extract_tree_with_pruning(match_dict: dict, root_coords: list[float, float] = None, level: int = 0, node_idx: int = 1, legend: dict = None, prune_func = None, node_pruned = False) -> dict:
    if legend is None:
        legend = dict()
    
    x_range = len(_get_matches_from_dict(match_dict))
    children = match_dict['children']
    node_pruned = False # this leads to pruning some patterns but possibly not pruning their descendants - if we not execute this line - all descendants of a pruned pattern are also automatically pruned
    
    if not root_coords:
        root_coords = [x_range/2, 0]
        x_coords = np.linspace(0, x_range, len(children))
    else:
        x_coords = np.linspace(-x_range/2, x_range/2, len(children)) + root_coords[0]

    # early return if minimum prob is not met
    if prune_func:
        prob = match_dict['prob']
        if level > 1:
            cutoff = prune_func(level-1)
            if prob < cutoff:
                node_pruned = True
    
    legend[node_idx] = {'name':match_dict['name'],
                        'total_log_rpif_dist':root_coords[1],
                        'local_log_rpif_dist':match_dict['log_rpif_dist'],
                        'node_level':level,
                        'n_children':len(children),
                        'prob':match_dict['prob'],
                        # 'freq':match_dict['freq'],
                        'pruned':node_pruned}
    
    for child, x_coord in zip(children, x_coords):              # for each child plot the next level
        extract_tree_with_pruning(child, [x_coord, root_coords[1] + child['log_rpif_dist']], level+1, node_idx+1, legend=legend, prune_func=prune_func, node_pruned = node_pruned)
        node_idx += len(_get_matches_from_dict(child))+1
    
    return legend

def _rescale_probs(probs):
    probs = np.array([_ for _ in probs])
    probs = probs/sum(probs)
    return probs

def _build_prefix_array(prefix_dict: dict, eval_pattern_size: int):
    """Builds array of prefixes of different length from list of prefixes. Pads shorter prefixes with np.nan from the left

    Args:
        prefix_dict (dict): list of prefixes
    """

    raw_prefixes = [prefix['prefix'] for prefix in prefix_dict]
    raw_phase_prefixes = [prefix['phase_prefix'] for prefix in prefix_dict]
    prefix_lens = [len(prefix) for prefix in raw_prefixes]
    
    array_len = math.ceil(eval_pattern_size/2)

    prefix_array = np.empty((len(prefix_dict), 3, array_len)) # we only need the last pattern_size/2 + 1 sequence elements for pattern matching
    prefix_array[:] = np.nan
    
    for prefix_idx, (prefix, phase_prefix, prefix_len) in enumerate(zip(raw_prefixes, raw_phase_prefixes, prefix_lens)):
        prefix_array[prefix_idx, 0, array_len-len(prefix[-array_len:]):] = prefix[-array_len:]
        prefix_array[prefix_idx, 1, -1] = prefix_len
        prefix_array[prefix_idx, 2, array_len-len(phase_prefix[-array_len:]):] = phase_prefix[-array_len:]
        
    return prefix_array

def _build_prefix_array_vanilla(prefix_dict: dict):
    """Builds array of prefixes of different length from list of prefixes. Pads shorter prefixes with np.nan from the left

    Args:
        prefix_dict (dict): list of prefixes
    """

    raw_prefixes = [prefix['prefix'] for prefix in prefix_dict]
    prefix_lens = [len(prefix) for prefix in raw_prefixes]
    max_prefix_len = max(prefix_lens)

    prefix_array = np.empty((len(prefix_dict), max_prefix_len))
    prefix_array[:] = np.nan

    for prefix_idx, (prefix, prefix_len) in enumerate(zip(raw_prefixes, prefix_lens)):
        prefix_array[prefix_idx, max_prefix_len-prefix_len:] = prefix

    return prefix_array

def _setup_shared_memory(data: np.ndarray, max_seq_len: int = None):
    """Sets up shared memory objects (multiprocessing.shared_memory.SharedMemory) for input data and corresponding results
    for multiple workers to work on

    Args:
        data (np.ndarray): prefix information in form of a left-padded np.ndarray
    """
    n_rows = len(data)

    # generate shared memory for input data and fill it with the input data
    shm_input = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_input = np.ndarray(data.shape, dtype=data.dtype, buffer=shm_input.buf)
    shared_input[:] = data

    # generate shared memory for results - one float per row
    # shape depends on the task
    #   NAP - single activities -> 1 float per row
    #   RTP - full remaining traces -> n floats per row with max_seq_len as shape indicator
    result_buffer = np.zeros((n_rows, max_seq_len), dtype=np.float64)
    shm_result = shared_memory.SharedMemory(create=True, size=result_buffer.nbytes)
    shared_result = np.ndarray(result_buffer.shape, dtype=result_buffer.dtype, buffer=shm_result.buf)
    
    return shm_input, shared_input, shm_result, shared_result

def _reconvert_activities_from_shared_mem(predicted_activities: np.ndarray):

    predicted_activities_list = list()
    for pa in predicted_activities:
        try:
            converted_pa = int(pa)
        except ValueError:
            converted_pa = None
        predicted_activities_list.append(converted_pa)

    return predicted_activities_list

def _reconvert_traces_from_shared_mem(predicted_traces: np.ndarray):

    predicted_traces_list = list()
    for pt in predicted_traces:
        try:
            relevant_trace = [pa for pa in pt if not np.isnan(pa)]
            converted_pt = [int(pa) for pa in relevant_trace]                
        except ValueError:
            converted_pt = None
        predicted_traces_list.append(converted_pt)

    return predicted_traces_list


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