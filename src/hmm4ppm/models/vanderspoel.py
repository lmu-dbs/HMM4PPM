from __future__ import annotations

import numpy as np
import pandas as pd
import math
from multiprocessing.managers import DictProxy
from multiprocessing import shared_memory, Process
import time
from collections import Counter
from tqdm import tqdm
from enum import Enum
from sklearn.preprocessing import LabelEncoder
from ..data.sequencedata import SequenceData
from ..util.sequence_utils import _child_matches_with_sequence, _filter_start_end

import heapq

import logging
logger = logging.getLogger(__name__)

track_progress = False

class Task(Enum):
    NAP = 'nap'
    RTP = 'rtp'
    
class VDSPredictorHMMPhases():
    """Prediction model using the approach by van der Spoel et al. (2012) with HMM phase integration
    """

    def __init__(self, include_phases: bool, hmm_model, hmm_predictor, **model_args):

        logger.info('Initializing prediction model')
        
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.graphs = None
        
        self.include_phases = include_phases
        
        self.hmm_model = hmm_model
        self.hmm_predictor = hmm_predictor
        
    def fit(self) -> None:
        """Fitting the model to X (training data). This involves the extraction of possible paths to end activities for
        suffix prediction
        """
        self.generate_sequences(self.data_train)
        
        if self.include_phases:
            self.transition_matrices = self.mine_directly_follows_graph_phase(horizon=1)
        else:    
            self.transition_matrix = self.mine_directly_follows_graph(horizon=1)
        
        logger.info('Graph mining completed!')

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
        prefix_array = _build_prefix_array(self.data_test.relevant_prefixes)
        pred_start_time = time.perf_counter()

        if True:
        # if task==Task.RTP:
            # do remaining trace prediction
            max_prefix_len = max([len(prefix['prefix']) for prefix in self.data_test.relevant_prefixes])
            max_seq_len = int(break_buffer*max_prefix_len)
            if ncores==1:
                predicted_traces = np.array([lst + [np.nan] * (max_seq_len - len(lst)) for lst in [self._predict_sequence(prefix=row,
                                                                                                                          include_phases=self.include_phases,
                                                                                                                          break_after_seq_len=max_seq_len) for row in tqdm(prefix_array)]])
            else:
                shm_input, shared_input, shm_result, shared_result = _setup_shared_memory(prefix_array, max_seq_len=max_seq_len)

                # generating index slices according to number of workers (ncores)
                chunksize = len(prefix_array) // ncores
                slices = [
                    (i * chunksize, (i + 1) * chunksize if i < ncores - 1 else len(prefix_array))
                    for i in range(ncores)
                ]

                procs = [
                    Process(
                        target=self._shared_worker,
                        args=(i, task, shm_input.name, prefix_array.shape, prefix_array.dtype, s, e, max_seq_len, 
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
        
        if True:
        # elif task==Task.NAP:

            max_prefix_len = max([len(prefix['prefix']) for prefix in self.data_test.relevant_prefixes])
            max_seq_len = int(break_buffer*max_prefix_len)
            
            if ncores==1:
                predicted_activities = np.array([self._predict_activity(prefix=row,
                                                                        include_phases=self.include_phases,
                                                                        ) for row in tqdm(prefix_array)])
                
            else:
                shm_input, shared_input, shm_result, shared_result = _setup_shared_memory(prefix_array, max_seq_len=1)

                # generating index slices according to number of workers (ncores)
                chunksize = len(prefix_array) // ncores
                slices = [
                    (i * chunksize, (i + 1) * chunksize if i < ncores - 1 else len(prefix_array))
                    for i in range(ncores)
                ]

                procs = [
                    Process(
                        target=self._shared_worker,
                        args=(i, task, shm_input.name, prefix_array.shape, prefix_array.dtype, s, e, max_seq_len, 
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
                logger.info(f"Workers done in {elapsed:.3f}s  |  {len(predicted_activities):,} prefixes processed")
                logger.info(f"Total time elapsed: {time.perf_counter() - pred_start_time:.3f}s")

            pred_duration = time.perf_counter() - convert_start_time

            predictions = _reconvert_activities_from_shared_mem(predicted_activities)
            
            self.predicted_activities = predictions

        pred_plus_convert_duration = time.perf_counter() - convert_start_time

        return predictions, pred_duration, pred_plus_convert_duration

    def _predict_sequence(self, prefix: list[int], include_phases: bool, break_after_seq_len: int = 10e5, verbose: bool = False) -> list[int]:
        """Predicts the remaining activities for a given prefix

        Args:
            prefix (list[int]): A prefix to predict remaining activities for
        
        Returns:
           list[int]: the predicted sequence containing the prefix and the predicted remaining activities
        """
        
        nan_rows = np.all(np.isnan(prefix), axis=0)
        non_nan_prefix = prefix[:, ~nan_rows]
        act_prefix = [el for el in non_nan_prefix[0, :]]
        
        prefix = act_prefix[:]
        
        if include_phases:
            markov_order = self.hmm_model.model_args_train['markov_order']
            n_phases = self.hmm_model.trained_params['AutoDelta.probs_initial'].shape[0]
            phase_prefix = [el for el in non_nan_prefix[1, :]]
            phase_hist = phase_prefix[-markov_order:]
            phase_hist_extension = list(np.random.choice([_ for _ in range(n_phases)], 
                                                    size=1, 
                                                    p=self.hmm_model.trained_params['AutoDelta.probs_initial'].numpy())) * (markov_order - len(phase_hist)) 
            phase_hist = [int(p) if isinstance(phase_hist[0], int) else float(p) for p in phase_hist_extension] + phase_hist

        try:
            last_start = len(prefix) - prefix[::-1].index(self.start_activity) - 1
        except ValueError as v_error:
            v_error.args = ('test_sequence contains no start_activities - predict only for non-left-truncated sequences' ,)
            raise

        # in case of unseen activities - take last known one
        start_act = prefix[-1]
        if start_act == -1:
            start_act = [act for act in prefix if act != -1][-1]
        
        try:    
            if include_phases:
                predicted_sequence, prob = self._most_likely_path_phases(start=start_act, end=self.data_train.end_activity, max_len=break_after_seq_len, phase_hist=phase_hist)
            else:
                predicted_sequence, prob = self._most_likely_path(start=start_act, end=self.data_train.end_activity, max_len=break_after_seq_len)
        except ValueError: # we cannot find the activity in the set of encoded activities (unseen data)
            predicted_sequence = None
        
        if predicted_sequence is None:
            return []
        else:
            return predicted_sequence[1:]
    
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

    def _predict_activity(self, prefix: list[int], include_phases: bool, break_after_seq_len: int = 10e5, verbose: bool = False) -> int:
        """Predicts the next activity for a given sequence

        Args:
            prefix (list[int]): A prefix to predict remaining activities for
        
        Returns:
            int: the predicted activity
        """

        nan_rows = np.all(np.isnan(prefix), axis=0)
        non_nan_prefix = prefix[:, ~nan_rows]
        act_prefix = [el for el in non_nan_prefix[0, :]]
        
        prefix = act_prefix[:]
        
        if include_phases:
            markov_order = self.hmm_model.model_args_train['markov_order']
            n_phases = self.hmm_model.trained_params['AutoDelta.probs_initial'].shape[0]
            phase_prefix = [el for el in non_nan_prefix[1, :]]
            phase_hist = phase_prefix[-markov_order:]
            phase_hist_extension = list(np.random.choice([_ for _ in range(n_phases)], 
                                                    size=1, 
                                                    p=self.hmm_model.trained_params['AutoDelta.probs_initial'].numpy())) * (markov_order - len(phase_hist)) 
            phase_hist = [int(p) if isinstance(phase_hist[0], int) else float(p) for p in phase_hist_extension] + phase_hist
        
        try:
            last_start = len(prefix) - prefix[::-1].index(self.start_activity) - 1
        except ValueError as v_error:
            v_error.args = ('test_sequence contains no start_activities - predict only for non-left-truncated sequences' ,)
            raise
        
        # in case of unseen activities - take last known one
        start_act = prefix[-1]
        if start_act == -1:
            start_act = [act for act in prefix if act != -1][-1]
        
        try:
            if include_phases:
                predicted_sequence, prob = self._most_likely_path_phases(start=start_act, end=self.data_train.end_activity, max_len=break_after_seq_len, phase_hist=phase_hist)
            else:
                predicted_sequence, prob = self._most_likely_path(start=start_act, end=self.data_train.end_activity, max_len=break_after_seq_len)
        except ValueError: # we cannot find the activity in the set of encoded activities (unseen data)
            predicted_sequence = None
        
        if predicted_sequence is None:
            return None
        else:
            return predicted_sequence[1]

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
            raise ValueError('data not found - make sure to load train, validation and test data with load_data()')
        
        train_case_ids = self.data_train.data[self.data_train.case_identifier].unique().tolist()
        
        # get phases from annotated dataset
        self.data_train.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_train.case_identifier].isin(train_case_ids)]['phase'].reset_index(drop=True)
        
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
        self.data_train.extract_traces(columns=[self.data_train.activity_identifier, 'phase'])
        self.start_activity = self.data_train.start_activity
        self.end_activity = self.data_train.end_activity

        self.data_train.generate_prefixes(include_phases=self.include_phases)
        self.data_train.pick_relevant_prefixes()

        self.max_prefix_len = max([len(prefix['prefix']) for prefix in self.data_train.relevant_prefixes])

        logger.info('Training data prepared!')

    def prepare_val(self, act_encoder: LabelEncoder = None, filter_sequences: bool = True, padding_size: int = 0):
        # TODO
        # include doc string
        logger.info('Preparing validation data...')
        if self.data_val is None:
            raise ValueError('data not found - make sure to load train, validation and test data with load_data()')
    
        val_case_ids = self.data_val.data[self.data_val.case_identifier].unique().tolist()
        
        # get phases from annotated dataset
        self.data_val.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_val.case_identifier].isin(val_case_ids)]['phase'].reset_index(drop=True)
        
        self.data_val.pad_columns(cols_to_pad=[self.data_val.activity_identifier], n_pad=padding_size)
        act_idx = self.data_val.data.groupby(self.data_val.case_identifier).apply(lambda x: pd.Series(range(-padding_size, 
                                                                                          len(x)-padding_size)))
        self.data_val.data['activity_idx'] = act_idx.reset_index(drop=True)
    
        # forward and backward fill timestamp column
        self.data_val.data[self.data_val.timestamp_identifier] = self.data_val.data.groupby(self.data_val.case_identifier)[self.data_val.timestamp_identifier].ffill()
        self.data_val.data[self.data_val.timestamp_identifier] = self.data_val.data.groupby(self.data_val.case_identifier)[self.data_val.timestamp_identifier].bfill()
    
        # self.data_test.encode_activities(act_encoder=act_encoder)
        # self.data_test.encode_attributes(attrs_to_encode=self.data_test.attributes, attr_encoders=attr_encoders)
        self.data_val.extract_traces(columns=[self.data_test.activity_identifier, 'phase'])
    
        self.data_val.generate_prefixes(include_phases=self.include_phases)
        self.data_val.pick_relevant_prefixes()
    
        self.data_val.generate_full_sequences(filter_sequences=filter_sequences)
        self.data_val.generate_full_future_sequences(filter_sequences=filter_sequences)
        self.data_val.generate_next_activities()
        logger.info('Test data prepared!')
    
    def prepare_test(self, act_encoder: LabelEncoder = None, filter_sequences: bool = True, padding_size: int = 0):
        # TODO
        # include doc string
        logger.info('Preparing test data...')
        if self.data_test is None:
            raise ValueError('data not found - make sure to load train, validation and test data with load_data()')
    
        test_case_ids = self.data_test.data[self.data_test.case_identifier].unique().tolist()
        
        # get phases from annotated dataset
        self.data_test.data['phase'] = self.phase_annotated_df[self.phase_annotated_df[self.data_test.case_identifier].isin(test_case_ids)]['phase'].reset_index(drop=True)
        
        self.data_test.pad_columns(cols_to_pad=[self.data_test.activity_identifier], n_pad=padding_size)
        act_idx = self.data_test.data.groupby(self.data_test.case_identifier).apply(lambda x: pd.Series(range(-padding_size, 
                                                                                          len(x)-padding_size)))
        self.data_test.data['activity_idx'] = act_idx.reset_index(drop=True)
    
        # forward and backward fill timestamp column
        self.data_test.data[self.data_test.timestamp_identifier] = self.data_test.data.groupby(self.data_test.case_identifier)[self.data_test.timestamp_identifier].ffill()
        self.data_test.data[self.data_test.timestamp_identifier] = self.data_test.data.groupby(self.data_test.case_identifier)[self.data_test.timestamp_identifier].bfill()
    
        # self.data_test.encode_activities(act_encoder=act_encoder)
        # self.data_test.encode_attributes(attrs_to_encode=self.data_test.attributes, attr_encoders=attr_encoders)
        self.data_test.extract_traces(columns=[self.data_test.activity_identifier, 'phase'])
    
        self.data_test.generate_prefixes(include_phases=self.include_phases)
        self.data_test.pick_relevant_prefixes()
    
        self.data_test.generate_full_sequences(filter_sequences=filter_sequences)
        self.data_test.generate_full_future_sequences(filter_sequences=filter_sequences)
        self.data_test.generate_next_activities()
        logger.info('Test data prepared!')
    
    def _most_likely_path(self, start, end, max_len):

        transition_matrix = self.transition_matrix
        
        n = len(transition_matrix)
        activities = list(transition_matrix.columns)
        matrix = np.array(transition_matrix, dtype=float)

        start_idx = activities.index(start)
        end_idx = activities.index(end)

        INF = float('inf')
        best_cost  = [INF] * n
        best_steps = [INF] * n
        best_cost[start_idx]  = 0.0
        best_steps[start_idx] = 0
        prev = [-1] * n

        heap = [(0.0, 0, start_idx)]

        while heap:
            cost, steps, u = heapq.heappop(heap)

            if cost > best_cost[u]:
                continue  # stale entry

            if u == end_idx:
                break

            if max_len is not None and steps > max_len:
                continue
                
            for v in range(n):
                p = matrix[u][v]
                if p <= 0:
                    continue

                new_cost  = cost + (-math.log(p))
                new_steps = steps + 1

                if new_cost < best_cost[v]:
                    best_cost[v]  = new_cost
                    best_steps[v] = new_steps
                    prev[v] = u
                    heapq.heappush(heap, (new_cost, new_steps, v))

        # Reconstruct path
        if best_cost[end_idx] == INF:
            return None, 0.0  # no path found within max_steps

        path_idx = []
        node = end_idx
        while node != -1:
            path_idx.append(node)
            node = prev[node]
        path_idx.reverse()

        path = [activities[i] for i in path_idx]
        probability = np.exp(-best_cost[end_idx])

        return path, probability
    
    def _most_likely_path_phases(self, start, end, max_len, phase_hist):
    
        transition_matrix = self.transition_matrices[f'phase_{int(phase_hist[-1])}']        

        n = len(transition_matrix)
        s = self.hmm_model.trained_params['AutoDelta.probs_initial'].shape[0]
        activities = list(transition_matrix.columns)
        # matrix = np.array(transition_matrix, dtype=float)

        start_idx = activities.index(start)
        end_idx = activities.index(end)

        INF = float('inf')
        best_cost  = [INF] * n
        best_steps = [INF] * n
        best_cost[start_idx]  = 0.0
        best_steps[start_idx] = 0
        prev = [-1] * n
        prev_phase = [-1] * s

        heap = [(0.0, 0, start_idx, phase_hist)]

        while heap:
            cost, steps, u, p_hist = heapq.heappop(heap)

            if cost > best_cost[u]:
                continue  # stale entry

            if u == end_idx:
                break

            if max_len is not None and steps > max_len:
                continue

            matrix = np.array(self.transition_matrices[f'phase_{int(p_hist[-1])}'], dtype=float)
            phase_transition_probs = self.hmm_model.trained_params['AutoDelta.probs_x'][tuple([int(p) for p in p_hist])]

            for next_phase in range(s):
                phase_p = phase_transition_probs[next_phase]

                if phase_p <= 0:
                    continue

                cost_phase = cost + (-math.log(phase_p))

                for v in range(n):
                    p = matrix[u][v]
                    if p <= 0:
                        continue

                    new_cost  = cost_phase + (-math.log(p))
                    new_steps = steps + 1

                    if new_cost < best_cost[v]:
                        best_cost[v]  = new_cost
                        best_steps[v] = new_steps
                        prev[v] = u
                        prev_phase[next_phase] = p_hist
                        heapq.heappush(heap, (new_cost, new_steps, v, p_hist[:-1] + [next_phase]))

        # Reconstruct path
        if best_cost[end_idx] == INF:
            return None, 0.0  # no path found within max_steps

        path_idx = []
        node = end_idx
        while node != -1:
            path_idx.append(node)
            node = prev[node]
        path_idx.reverse()

        path = [activities[i] for i in path_idx]
        probability = np.exp(-best_cost[end_idx])

        return path, probability
    
    def generate_sequences(self, X: SequenceData):
        """Generates sequences of activities inside traces by process stage. 

        Args:
            X (SequenceData): The SequenceData object we generate the sequences for. This has to be prepared
            such that it has generated traces as class attribute (SequenceData.prepare_train())
        """
        logger.info(f'Generating sequences from trace data...')
        sequences = list()
        phase_sequences = list()
        
        for trace in X.traces:
            sequence = trace[X.activity_identifier]
            sequences.append(sequence)
            phase_sequence = trace['phase']
            phase_sequences.append(phase_sequence)
            
        self.sequences = sequences
        self.phase_sequences = phase_sequences

        logger.info(f'Sequence generation completed!')

    def _shared_worker(self, 
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
                result = self._batch_predict_activity(prefixes=prefix_array[start:end], 
                                                      break_after_seq_len=max_seq_len, 
                                                      worker_id=worker_id,
                                                      **kwargs)
            elif task==Task.RTP:
                result = self._batch_predict_sequence(prefixes=prefix_array[start:end], 
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

    def mine_directly_follows_graph(self, horizon: int = 1, prob_threshold_rel: float = 0, prob_threshold_abs: float = 0,) -> pd.DataFrame:
        """
        Function for extracting a follows matrix from a list of sequences

                Parameters:
                        horizon (int): the horizon the transitions are extracted for. horizon=1 specifies a directly follows graph (DFG).
                                    horizon=-1 extracts an eventually follows graph (EFG)

                        prob_threshold_rel (float): A relative probability threshold to filter the follows matrix for per station.
                                                    All entries in the corresponding station row of the follows matrix that show a
                                                    probability of less than the threshold are rounded to zero.
                                                    Reasonable value is e.g. <0.01.

                        prob_threshold_abs (float): An absolute probability threshold to filter complete station rows/columns of the
                                                    follows matrix. All stations that have a lower probability of entries in the
                                                    follows matrix (across rows and columns) are excluded from the matrix.
                                                    Reasonable value is e.g. <0.005.
                Returns:
                        follows_matrix: a (directly) follows matrix from the event log specifying the transitions in the event log
        """

        all_activities = list()

        for seq in self.sequences:
            all_activities.extend(seq)
        
        activities = list(set(all_activities))

        if horizon == -1:
            horizon = max([len(seq) for seq in self.sequences])

        follows_matrix = np.zeros(
            shape=(len(activities), len(activities))
        ).astype(int)

        for seq in self.sequences:

            for i in range(0, len(seq) - 1):
                current_activity = activities.index(seq[i])
                current_remaining_trace_acts = seq[i + 1 :][
                    0 : min(horizon, len(seq) - (i + 1))
                ]
                current_remaining_trace_indices = [activities.index(a) for a in current_remaining_trace_acts]
                follows_matrix[current_activity][current_remaining_trace_indices] += 1

        # filtering with probability thresholds

        # relative
        for row in range(0, follows_matrix.shape[0]):
            for col in range(0, follows_matrix.shape[1]):
                if (
                    follows_matrix[row, col]
                    < prob_threshold_rel * follows_matrix[row, :].sum()
                ):
                    follows_matrix[row, col] = 0

                if (
                    follows_matrix[row, col]
                    < prob_threshold_rel * follows_matrix[:, col].sum()
                ):
                    follows_matrix[row, col] = 0

        # absolute
        acts_to_filter = list()
        for row_col in range(0, follows_matrix.shape[0]):  # matrix shape is quadratic
            act_occurrences = (
                follows_matrix[row_col, :].sum()
                + follows_matrix[:, row_col].sum()
                - follows_matrix[row_col, row_col]
            )
            if act_occurrences < prob_threshold_abs * follows_matrix.sum():
                acts_to_filter.append(row_col)

        follows_matrix[acts_to_filter, :] = 0
        follows_matrix[:, acts_to_filter] = 0

        # follows_matrix = np.apply_along_axis(lambda x: x/sum(x) if sum(x) > 0 else 1/follows_matrix.shape[0], 1, follows_matrix)
        follows_matrix = np.apply_along_axis(lambda x: x/sum(x), 1, follows_matrix)
        for row in range(follows_matrix.shape[0]):
            if all(np.isnan(follows_matrix[row, :])):
                follows_matrix[row, :] = 1/follows_matrix.shape[0]

        follows_matrix = pd.DataFrame(follows_matrix)
        follows_matrix.index, follows_matrix.columns = activities, activities

        return follows_matrix
    
    def mine_directly_follows_graph_phase(self, horizon: int = 1, prob_threshold_rel: float = 0, prob_threshold_abs: float = 0,) -> pd.DataFrame:
        """
        Function for extracting a follows matrix from a list of sequences for each of a process event logs phases.

                Parameters:
                        horizon (int): the horizon the transitions are extracted for. horizon=1 specifies a directly follows graph (DFG).
                                    horizon=-1 extracts an eventually follows graph (EFG)

                        prob_threshold_rel (float): A relative probability threshold to filter the follows matrix for per station.
                                                    All entries in the corresponding station row of the follows matrix that show a
                                                    probability of less than the threshold are rounded to zero.
                                                    Reasonable value is e.g. <0.01.

                        prob_threshold_abs (float): An absolute probability threshold to filter complete station rows/columns of the
                                                    follows matrix. All stations that have a lower probability of entries in the
                                                    follows matrix (across rows and columns) are excluded from the matrix.
                                                    Reasonable value is e.g. <0.005.
                Returns:
                        follows_matrices: a dictionary of (directly) follows matrices for each phase from the event log specifying the 
                                          phase-specific transitions in the event log
        """

        all_activities = list()

        for seq in self.sequences:
            all_activities.extend(seq)

        activities = list(set(all_activities))
        phases = [_ for _ in range(self.hmm_model.model_args_train['hidden_dim'])]

        if horizon == -1:
            horizon = max([len(seq) for seq in self.sequences])
            
        follows_matrices = {f'phase_{p}': np.zeros(shape=(len(activities), len(activities))).astype(int) for p in phases}

        # follows_matrix = np.zeros(
        #     shape=(len(activities), len(activities))
        # ).astype(int)

        for seq, p_seq in zip(self.sequences, self.phase_sequences):

            for i in range(0, len(seq) - 1):
                current_activity = activities.index(seq[i])
                current_phase = p_seq[i]
                current_remaining_trace_acts = seq[i + 1 :][
                    0 : min(horizon, len(seq) - (i + 1))
                ]
                current_remaining_trace_indices = [activities.index(a) for a in current_remaining_trace_acts]
                follows_matrices[f'phase_{current_phase}'][current_activity][current_remaining_trace_indices] += 1

        # filtering with probability thresholds

        for phase_key, follows_matrix in follows_matrices.items():

            # relative
            for row in range(0, follows_matrix.shape[0]):
                for col in range(0, follows_matrix.shape[1]):
                    if (
                        follows_matrix[row, col]
                        < prob_threshold_rel * follows_matrix[row, :].sum()
                    ):
                        follows_matrix[row, col] = 0

                    if (
                        follows_matrix[row, col]
                        < prob_threshold_rel * follows_matrix[:, col].sum()
                    ):
                        follows_matrix[row, col] = 0

            # absolute
            acts_to_filter = list()
            for row_col in range(0, follows_matrix.shape[0]):  # matrix shape is quadratic
                act_occurrences = (
                    follows_matrix[row_col, :].sum()
                    + follows_matrix[:, row_col].sum()
                    - follows_matrix[row_col, row_col]
                )
                if act_occurrences < prob_threshold_abs * follows_matrix.sum():
                    acts_to_filter.append(row_col)

            follows_matrix[acts_to_filter, :] = 0
            follows_matrix[:, acts_to_filter] = 0

            # follows_matrix = np.apply_along_axis(lambda x: x/sum(x) if sum(x) > 0 else 1/follows_matrix.shape[0], 1, follows_matrix)
            follows_matrix = np.apply_along_axis(lambda x: x/sum(x), 1, follows_matrix)
            for row in range(follows_matrix.shape[0]):
                if all(np.isnan(follows_matrix[row, :])):
                    follows_matrix[row, :] = 1/follows_matrix.shape[0]

            follows_matrix = pd.DataFrame(follows_matrix)
            follows_matrix.index, follows_matrix.columns = activities, activities
            
            follows_matrices[phase_key] = follows_matrix

        return follows_matrices

def _build_prefix_array(prefix_dict: dict):
    """Builds array of prefixes of different length from list of prefixes. Pads shorter prefixes with np.nan from the left

    Args:
        prefix_dict (dict): list of prefixes
    """

    raw_prefixes = [prefix['prefix'] for prefix in prefix_dict]
    raw_phase_prefixes = [prefix['phase_prefix'] for prefix in prefix_dict]
    prefix_lens = [len(prefix) for prefix in raw_prefixes]
    max_prefix_len = max(prefix_lens)

    prefix_array = np.empty((len(prefix_dict), 2, max_prefix_len))
    prefix_array[:] = np.nan
    
    for prefix_idx, (prefix, phase_prefix, prefix_len) in enumerate(zip(raw_prefixes, raw_phase_prefixes, prefix_lens)):
        prefix_array[prefix_idx, 0, max_prefix_len-prefix_len:] = prefix
        prefix_array[prefix_idx, 1, max_prefix_len-prefix_len:] = phase_prefix
        
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
        except (ValueError, TypeError):
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