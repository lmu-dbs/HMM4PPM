from __future__ import annotations

import numpy as np
import pandas as pd
import copy
import os
import math
import itertools
import datetime as dt

from sklearn.preprocessing import OrdinalEncoder
from ..encoding.util import EncodingFactory, TransformFactory

from ..util.sequence_utils import _filter_start_end

from ..util.logging import init_logging
logger = init_logging(__name__, "SequenceData.log")

class SequenceData:
    """Data class for preprocessing of .csv files into sequences that can be processed by the models implemented
    """
        
    def __init__(self, data, case_identifier, activity_identifier, resource_identifier, timestamp_identifier, encoding_params = None, **kwargs):
        
        self.logger = logger

        self.logger.info('Initializing sequence data...')

        self.data = data if data is not None else {}

        self.case_identifier = case_identifier
        self.activity_identifier = activity_identifier
        self.resource_identifier = resource_identifier
        self.timestamp_identifier = timestamp_identifier

        self.traces = None
        self.all_prefixes = None
        self.relevant_prefixes = None

        self.full_sequences_unfiltered = None
        self.full_future_sequences_unfiltered = None

        self.logger.info('Recoding sequence data types...')
        self.recode_types(encoding_params)

        self.logger.info('Data initialization completed!')

    @classmethod
    def from_csv(cls, load_path: os.PathLike, **kwargs) -> SequenceData:
        """Initialize SequenceData from a .csv file

        Args:
            load_path (os.PathLike): the path to the .csv file

        Returns:
            SequenceData: the new SequenceData object
        """
        data = pd.read_csv(load_path)
        return cls(data=data, **kwargs)

    @classmethod
    def from_obj(cls, obj: SequenceData, **kwargs) -> SequenceData:
        """Initialize SequenceData from an existing SequenceData object and update parameters with given kwargs

        Args:
            obj (SequenceData): the SequenceData we want to initialize from

        Raises:
            TypeError: if an object from a different class is given as obj

        Returns:
            SequenceData: the new SequenceData object
        """

        if not isinstance(obj, cls):
            raise TypeError("object must be an instance of SequenceData")
        new_init_params = copy.deepcopy(vars(obj))
        for param, value in kwargs.items():
            new_init_params[param] = value

        return cls(**new_init_params)
    
    def setup_encoders(self, encoding_params):

        self.encoders = dict()
        for type, cols in encoding_params.items():
            
            encoder = type.split('_')[-1]
            if encoder == 'OneHotEncoder':
                encoder_obj = EncodingFactory.create(encoder, sparse_output=False, handle_unknown='ignore') # TODO - should we also use encoded value here?
            elif encoder == 'OrdinalEncoder':
                encoder_obj = EncodingFactory.create("StrictlyNonNegativeOrdinalEncoder")
                # encoder_obj = EncodingFactory.create(encoder, handle_unknown="use_encoded_value", unknown_value=-1)
            else:
                encoder_obj = EncodingFactory.create(encoder)

            self.encoders.update({encoder: (encoder_obj, cols)})

    def encode_attributes(self, already_fitted: bool = False):
        
        # all_transform_cols = list()
        # for (_, cols) in self.transformers.values():
        #     all_transform_cols.extend(cols)
        # if len(all_transform_cols) > 0:
        #     data_to_encode = self.data_transformed
        # else:
        #     data_to_encode = self.data

        data_to_encode = self.data

        transforms = list()
        encode_cols = list()
        # if train_encoders is None:  # training encoding
        if not already_fitted:  # training encoding
            for _, (encoder, cols) in self.encoders.items():
                if len(cols) > 0:
                    encode_cols.extend(cols)
                    encoder.fit(data_to_encode[cols])
                    transformed_cols = encoder.transform(data_to_encode[cols])
                    transforms.append(transformed_cols)
        else:                       # test encoding with train encoders
            for _, (encoder, cols) in self.encoders.items():
                if len(cols) > 0:
                    encode_cols.extend(cols)
                    transformed_cols = encoder.transform(data_to_encode[cols])
                    transforms.append(transformed_cols)
        
        if len(transforms) != 0:
            encoded_data = np.hstack(transforms) # we stack transformed cols in order of encoder cols in the data_config
        else:
            encoded_data = transforms
        self.data[encode_cols] = encoded_data
        self.attribute_identifiers = encode_cols

    def setup_transformers(self, transform_params):

        self.transformers = dict()
        for type, cols in transform_params.items():
            
            transformer = type.split('_')[-1]
            if transformer == 'PowerTransformer':
                # we do not standardize - we encode afterwards
                transformer_obj = TransformFactory.create(transformer, method='yeo-johnson', standardize=False)
            else:
                transformer_obj = TransformFactory.create(transformer)

            self.transformers.update({transformer: (transformer_obj, cols)})

    def transform(self, already_fitted: bool = False):
        
        transforms = list()
        transform_cols = list()
        if not already_fitted:  # training transformations
        # if train_transformers is None:  # training transformations
            for _, (transformer, cols) in self.transformers.items():
                if len(cols) > 0:
                    transform_cols.extend(cols)
                    transformer.fit(self.data[cols])
                    transformed_cols = transformer.transform(self.data[cols])
                    transforms.append(transformed_cols)
        else:                       # test encoding with train transformers
            for _, (transformer, cols) in self.transformers.items():
                if len(cols) > 0:
                    transformed_cols = transformer.transform(self.data[cols])
                    transforms.append(transformed_cols)
        
        if len(transforms) == 0:
            self.data_transformed = self.data.copy()
        else:
            transformed_data = np.hstack(transforms) # we stack transformed cols in order of encoder cols in the data_config

            # retransform to pd.DataFrame so that we can follow up with encoding
            all_encode_cols = list()
            for (_, cols) in self.encoders.values():
                all_encode_cols.extend(cols)
                
            if len(all_encode_cols) > 0:
                data_to_transform = self.data.copy()
                all_transform_cols = list()
                for (_, cols) in self.transformers.values():
                    all_transform_cols.extend(cols)
                col_diff = set(all_transform_cols).difference(set(all_encode_cols))
                if len(col_diff):
                    raise ValueError(f"cols transformed that are not needed for encoding - {' ,'.join(col_diff)}")
                for idx, col in enumerate(all_transform_cols):
                    data_to_transform[col] = transformed_data[:, idx]
                self.data_transformed = data_to_transform
            else:
                self.data_transformed = transformed_data
            # self.attribute_identifiers = encode_cols

    def recode_types(self, encoding_params) -> pd.DataFrame:
        """Recodes the types of the timestamp (-> datetime) and activity (str) columns

        Returns:
            pd.DataFrame: the pd.DataFrame with recoded column dtypes
        """
        recoded_times = pd.to_datetime(self.data[self.timestamp_identifier], format='mixed')

        self.data.loc[:,self.timestamp_identifier] = recoded_times
        self.data.loc[:,self.activity_identifier] = self.data.loc[:,self.activity_identifier].astype(object)
        self.data.loc[:,self.activity_identifier] = self.data.loc[:,self.activity_identifier].astype(str)
        
        if encoding_params is not None:
            for encoder, cols in encoding_params.items():
                encoder = encoder.split('_')[-1]
                existing_cols = [col for col in cols if col in self.data.columns]
                if len(existing_cols) > 0:
                    
                    if encoder == 'OrdinalEncoder':
                        self.data[cols] = self.data[cols].astype(object)
                        self.data[cols] = self.data[cols].astype(str)
                    else:
                        self.data[cols] = self.data[cols].astype(float)


    def pad_columns(self, cols_to_pad: list[str], n_pad: int = 1,  forward_pad: str = 'END',  backward_pad: str = 'START') -> pd.DataFrame:
        """Pads the existing sequence data with specified START and/or END tokens

        Args:
            cols_to_pad (list[str]): the columns to pad
            n_pad (int, optional): the length of the padding. Defaults to 1.
            forward_pad (str, optional): the pad token string for the END pad. 
            None results in no padding. Defaults to 'END'.
            backward_pad (str, optional): the pad token string for the START pad. 
            None results in no padding. Defaults to 'START'.

        Returns:
            pd.DataFrame: the padded pd.DataFrame
        """
        logger.info('Padding columns...')
        self.start_token = backward_pad
        self.end_token = forward_pad

        padded_data = self.data.copy()
        previous_dtypes = padded_data.dtypes

        padded_data = padded_data.groupby(self.case_identifier).apply(lambda x: _inner_pad(x,
                                                                    case_identifier=self.case_identifier,
                                                                    cols_to_pad=cols_to_pad,
                                                                    n_pad=n_pad,
                                                                    forward_pad=forward_pad,
                                                                    backward_pad=backward_pad))
            
        padded_data = padded_data.reset_index(drop=True)
        for col, dtype in enumerate(previous_dtypes):
            try:
                padded_data[padded_data.columns[col]] = padded_data.iloc(axis=1)[col].astype(dtype)
            except pd.errors.IntCastingNaNError:
                pass

        self.data = padded_data
        logger.info('Column padding completed!')
    
    def extract_traces(self, columns: list[str]) -> list[dict]:
        # TODO
        # include doc string
        logger.info('Generating trace data...')
        
        traces = list()

        grouped = self.data.groupby(self.case_identifier)

        for name, group in grouped:
            sequences = _extract_trace(df=group, columns=columns)
            trace = {self.case_identifier:[name]*len(group), **sequences}
            traces.append(trace)

        self.traces = traces
        logger.info('Traces generated!')
    
    def encode_activities(self, act_encoder: OrdinalEncoder = None):
        # TODO
        # include doc string
        
        if act_encoder is None:
            act_encoder = EncodingFactory.create("StrictlyNonNegativeOrdinalEncoder")
            act_encoder.fit(pd.DataFrame(self.data[self.activity_identifier]))
            self.act_encoder = act_encoder

        self.act_mapping = dict(zip(act_encoder.categories_[0], act_encoder.transform(act_encoder.categories_[0].reshape(-1, 1)).squeeze(1).tolist()))
        encoded_activities = act_encoder.transform(pd.DataFrame(self.data[self.activity_identifier]))
        self.data[self.activity_identifier] = encoded_activities

        self.start_activity = self.act_mapping[self.start_token]
        self.end_activity = self.act_mapping[self.end_token]
    
    def generate_time_features(self, features: list[str] = ["tsmn", "tscs", "tsle"]):

        features = [feature.lower() for feature in features]
        feature_frame = pd.DataFrame()

        for feature in features:
            try:
                feature_func = globals()[f"_calculate_{feature}"]
                self.logger.info(
                    f"Generating time feature information: {feature.upper()}"
                )
                feature_series = self.data.groupby(self.case_identifier).apply(
                    lambda x: feature_func(
                        x,
                        timestamp_identifier=self.timestamp_identifier,
                    ),
                    include_groups=False,
                )
                feature_frame = pd.concat(
                    [feature_frame, feature_series], axis=1
                )
            except KeyError as e:
                e.add_note(f"no function for feature '{feature}' implemented")
                raise
        feature_frame.index = [idx[1] for idx in feature_frame.index]

        self.data = pd.concat([self.data, feature_frame], axis=1)
    
    def generate_prefixes(self, attributes: list[str] = None, include_phases: bool = False):
        prefixes = list()

        # TODO
        # we need to resolve if we need to pass attributes here or if we should always generate a multi-dimensional prefix
        # separation makes sense if we want to evaluate predicted activites separately from the attributes, vice versa
        # but we could also separate it later on. providing multi-dimensional prefixes is necessary for prediction inside
        # the HMM

        for trace in self.traces:
            case_id = trace[self.case_identifier][0]
            activity_sequence = trace[self.activity_identifier]
            if include_phases:
                phase_sequence = trace['phase']
            else:
                phase_sequence = [0] * len(activity_sequence)
            if attributes:
                full_attribute_sequences = dict()
                for attribute in attributes:
                    full_attribute_sequence = trace[attribute]
                    full_attribute_sequences[attribute] = full_attribute_sequence

            for prefix_size in range(1, len(activity_sequence)):
                if attributes:
                    full_future_attribute_sequences = {attr:seq[prefix_size:] for attr, seq in full_attribute_sequences.items()}
                    attribute_prefixes = {f'{attribute}_prefix': trace[attribute][:prefix_size] for attribute in attributes}
                    current_prefix = {'case_id':case_id,
                                        'prefix':activity_sequence[:prefix_size], 
                                        'phase_prefix':phase_sequence[:prefix_size], 
                                        'full_sequence':activity_sequence,
                                        'full_future_sequence':activity_sequence[prefix_size:],
                                        'full_attribute_sequences':full_attribute_sequences,
                                        'full_future_attribute_sequences':full_future_attribute_sequences,}
                    
                    current_prefix.update(attribute_prefixes)
                else:
                    current_prefix = {'case_id':case_id,
                                    'prefix':activity_sequence[:prefix_size], 
                                    'phase_prefix':phase_sequence[:prefix_size], 
                                    'full_sequence':activity_sequence,
                                    'full_future_sequence':activity_sequence[prefix_size:],}
                prefixes.append(current_prefix)

        self.all_prefixes = prefixes
    
    def pick_relevant_prefixes(self):
        relevant_prefixes = list()
        
        # TODO
        # this adds empty prefixes (only START tokens / START activities) to the relevant_prefixes list
        # HMM is taking prefixes with the first activity already known
        # we need to also start at first activity already known here
        for prefix_dict in self.all_prefixes:
            prefix = prefix_dict['prefix']
            full_sequence = prefix_dict['full_sequence']
            full_n_start_acts = sum([True if act==self.start_activity else False for act in full_sequence])

            prefix_n_start_acts = sum([True if act==self.start_activity else False for act in prefix])
            prefix_n_end_acts = sum([True if act==self.end_activity else False for act in prefix])

            # if prefix_n_start_acts == full_n_start_acts and prefix_n_end_acts == 0:
            #     relevant_prefixes.append(prefix_dict)
            if len(prefix) > full_n_start_acts and prefix_n_end_acts == 0:
                relevant_prefixes.append(prefix_dict)

        self.relevant_prefixes = relevant_prefixes
    
    def generate_full_sequences(self, filter_sequences: bool = True):
        if self.relevant_prefixes is None:
            raise ValueError('relevant prefixes were not generated')
        
        full_sequences = [prefix['full_sequence'] for prefix in self.relevant_prefixes]
        self.full_sequences_unfiltered = full_sequences

        if filter_sequences:
            filtered_full_sequences = [_filter_start_end(full_seq, self.start_activity, self.end_activity) for full_seq in full_sequences]
            self.full_sequences = filtered_full_sequences
        else:
            self.full_sequences = full_sequences
    
    def generate_full_future_sequences(self, filter_sequences: bool = True):
        if self.relevant_prefixes is None:
            raise ValueError('relevant prefixes were not generated')
        
        full_future_sequences = [prefix['full_future_sequence'] for prefix in self.relevant_prefixes]
        self.full_future_sequences_unfiltered = full_future_sequences

        if filter_sequences:
            filtered_full_future_sequences = [_filter_start_end(full_seq, self.start_activity, self.end_activity) for full_seq in full_future_sequences]
            self.full_future_sequences = filtered_full_future_sequences
        else:
            self.full_future_sequences = full_future_sequences

    def generate_next_activities(self):
        if self.relevant_prefixes is None:
            raise ValueError('relevant prefixes were not generated')
        
        next_activities = [prefix['full_sequence'][len(prefix['prefix'])] for prefix in self.relevant_prefixes]
        self.next_activities = next_activities
    
    def generate_full_attribute_sequences(self, attributes: list[str], filter_sequences: bool = True):
        if self.relevant_prefixes is None:
            raise ValueError('relevant prefixes were not generated')
        
        all_full_attribute_sequences = dict()

        for attribute in attributes:
            full_attribute_sequences = [prefix['full_attribute_sequences'][attribute] for prefix in self.relevant_prefixes]
            all_full_attribute_sequences[attribute] = full_attribute_sequences

        # TODO/FIXME filtering does not work - fix it! we look for encoded self.start_activity, self.end_activity in the sequence - we do not have them in there - either look for nan or perform zip with activity sequence for filtering
        # is this fixed already?
        if filter_sequences:
            if self.full_sequences is None:
                raise ValueError("Full sequences (full_sequences) not generated yet but needed for filtering sequences. Make sure to generate them with generate_full_sequences first")
            all_filtered_full_attribute_sequences = dict()
            for attribute in attributes:
                indices_to_filter = [_filter_start_end(full_seq, self.start_activity, self.end_activity, only_idx=True) for full_seq in self.full_sequences_unfiltered]
                filtered_full_attribute_sequences = [[full_attribute_seq[filter_idx] for filter_idx in filter_idxs] for full_attribute_seq, filter_idxs in zip(all_full_attribute_sequences[attribute], indices_to_filter)]
                all_filtered_full_attribute_sequences[attribute] = filtered_full_attribute_sequences 
            self.full_attribute_sequences = all_filtered_full_attribute_sequences
        else:
            self.full_attribute_sequences = all_full_attribute_sequences
    
    def generate_full_future_attribute_sequences(self, attributes: list[str], filter_sequences: bool = True):
        if self.relevant_prefixes is None:
            raise ValueError('relevant prefixes were not generated')
        
        all_full_future_attribute_sequences = dict()

        for attribute in attributes:
            full_future_attribute_sequences = [prefix['full_future_attribute_sequences'][attribute] for prefix in self.relevant_prefixes]
            all_full_future_attribute_sequences[attribute] = full_future_attribute_sequences

        if filter_sequences:
            if self.full_future_sequences is None:
                raise ValueError("Full future sequences (full_future_sequences) not generated yet but needed for filtering sequences. Make sure to generate them with generate_full_sequences first")
            
            all_filtered_full_future_attribute_sequences = dict()
            for attribute in attributes:
                indices_to_filter = [_filter_start_end(full_seq, self.start_activity, self.end_activity, only_idx=True) for full_seq in self.full_future_sequences_unfiltered]
                filtered_full_future_attribute_sequences = [[full_future_attribute_seq[filter_idx] for filter_idx in filter_idxs] for full_future_attribute_seq, filter_idxs in zip(all_full_future_attribute_sequences[attribute], indices_to_filter)]
                all_filtered_full_future_attribute_sequences[attribute] = filtered_full_future_attribute_sequences
            self.full_future_attribute_sequences = all_filtered_full_future_attribute_sequences
        else:
            self.full_future_attribute_sequences = all_full_future_attribute_sequences

    def generate_next_attributes(self, attributes: list[str]):
        if self.relevant_prefixes is None:
            raise ValueError('relevant prefixes were not generated')
        
        all_next_attributes = dict()
        for attribute in attributes:
            next_attributes = [prefix['full_attribute_sequences'][attribute][len(prefix['prefix'])] for prefix in self.relevant_prefixes]
            all_next_attributes[attribute] = next_attributes
        self.next_attributes = all_next_attributes

    def train_test_split(self, train_pct: float, val_pct: float, cv: int = 1) -> tuple[SequenceData, SequenceData] | list[tuple[SequenceData, SequenceData]]:
        """Function for splitting an existing SequenceData object into two distinct SequenceData objects (train and test).

        Args:
            train_pct (float): Percentage share of the data that should be transferred into the training set object.
            The final test set consists of the remaining sequences.
            val_pct (float): Percentage share of the training data that should be used for validation.
            cv (int, optional): If k-fold cross-validation should be performed where cv is the number of folds. 
            Defaults to False.

        Returns:
            tuple[SequenceData]: training and test instances of the corresponding sequences as SequenceData objects
        """
        if cv > 1: # generate k-fold cross validation datasets
            logger.info(f'Splitting train and test data ({1/cv*100:.0f}-{(1-(1/cv))*100:.0f} split with {cv} folds)')
            
            if not train_pct == val_pct*cv:
                val_pct = 1/cv
                logger.warning(f"desired fold number of {cv} does not conform with val_pct={val_pct} - overriding val_pct to {1/cv}")
            
            all_ids = self.data[self.case_identifier].unique()
            np.random.shuffle(all_ids)
            
            outer_train_ids = np.random.choice(a=all_ids, size=int(len(all_ids)*train_pct), replace=False)
            outer_train_set = set(outer_train_ids)
            outer_test_ids = [id for id in all_ids if id not in outer_train_set]
            
            outer_train_data = self.data[self.data[self.case_identifier].isin(outer_train_ids)]
            outer_test_data = self.data[self.data[self.case_identifier].isin(outer_test_ids)]
            
            outer_train_data_obj = SequenceData.from_obj(self, data=outer_train_data)
            outer_test_data_obj = SequenceData.from_obj(self, data=outer_test_data)
            
            # split the data into cv folds
            id_folds = [id_fold for id_fold in _batch_samples(outer_train_ids, cv)]
            folds = list()
            for idf_idx in range(0, len(id_folds)):
                test_ids = id_folds[idf_idx].tolist()
                train_ids = list(itertools.chain(*[id_folds[i] for i in range(0, len(id_folds)) if i != idf_idx]))
                train_data = self.data[self.data[self.case_identifier].isin(train_ids)]
                test_data = self.data[self.data[self.case_identifier].isin(test_ids)]

                train_data_obj = SequenceData.from_obj(self, data=train_data)
                test_data_obj = SequenceData.from_obj(self, data=test_data)
                folds.append(tuple([train_data_obj, test_data_obj]))

            return folds, outer_train_data_obj, outer_test_data_obj
        else:
            logger.info(f'Splitting train, validation and test data ({(train_pct*(1-val_pct))*100:.0f}-{(train_pct*val_pct)*100:.0f}-{(1-train_pct)*100:.0f} split)')
            all_ids = self.data[self.case_identifier].unique()
            train_ids = np.random.choice(a=all_ids, size=int(len(all_ids)*train_pct), replace=False)
            train_id_set = set(train_ids)
            
            test_ids = [id for id in all_ids if id not in train_id_set]
            
            val_ids = np.random.choice(a=train_ids, size=int(len(train_ids)*val_pct), replace=False)
            val_set = set(val_ids)
            
            final_train_ids = [id for id in train_ids if id not in val_set]
            
            train_data = self.data[self.data[self.case_identifier].isin(final_train_ids)]
            val_data = self.data[self.data[self.case_identifier].isin(val_ids)]
            test_data = self.data[self.data[self.case_identifier].isin(test_ids)]

            train_data_obj = SequenceData.from_obj(self, data=train_data)
            val_data_obj = SequenceData.from_obj(self, data=val_data)
            test_data_obj = SequenceData.from_obj(self, data=test_data)

            return train_data_obj, val_data_obj, test_data_obj
    
    def get_characteristics(self) -> dict:
        
        log_characteristics = dict()

        log_characteristics['n_cases'] = self._get_n_cases()
        log_characteristics['n_events'] = self._get_n_events()
        log_characteristics['n_activities'] = self._get_n_activities()
        log_characteristics['mean_trace_len'] = float(self._get_mean_trace_len())
        log_characteristics['median_trace_len'] = float(self._get_median_trace_len())
        log_characteristics['max_trace_len'] = self._get_max_trace_len()
        log_characteristics['n_variants'] = self._get_n_variants()
        
        return log_characteristics
    
    def _get_n_cases(self) -> int:
        n_cases = self.data[self.case_identifier].nunique()
        return n_cases

    def _get_n_events(self) -> int:
        n_events = len(self.data)
        return n_events

    def _get_n_activities(self) -> int:
        n_activities = self.data[self.activity_identifier].nunique()
        return n_activities

    def _get_median_trace_len(self) -> float:
        trace_lens = self._get_trace_lens()
        trace_lens.sort()
        center_idx = int(len(trace_lens)/2)
        if len(trace_lens) % 2 == 0:
            median_trace_len = sum(trace_lens[center_idx-1:center_idx+1])/2
        else:
            median_trace_len = trace_lens[center_idx]
        return median_trace_len

    def _get_mean_trace_len(self) -> float:
        mean_trace_len = self._get_trace_lens().mean()
        return mean_trace_len

    def _get_max_trace_len(self) -> float:
        max_trace_len = int(self._get_trace_lens().max())
        return max_trace_len

    def _get_trace_lens(self) -> np.array[int]:
        trace_lens = np.array(self.data[self.case_identifier].value_counts())
        return trace_lens
    
    def _get_n_variants(self) -> int:
        variant_strings = self.data.groupby(self.case_identifier)[self.activity_identifier].apply(lambda x: ','.join(x))
        unique_variants = variant_strings.unique()
        return len(unique_variants)

def _extract_trace(df: pd.DataFrame, columns: list[str]) -> dict:
    trace = dict()
    for col in columns:
        trace[col] = df[col].tolist()

    return trace

def _inner_pad(df: pd.DataFrame,
               case_identifier: str,
               cols_to_pad: list[str],
               n_pad: int = 1, 
               forward_pad: str | None = 'END', 
               backward_pad: str | None = 'START') -> pd.DataFrame:

    if backward_pad:
        df = df.reset_index(drop=True).reindex(range(-n_pad, len(df))).reset_index(drop=True)
        
        # TODO
        # include functionality to pad with last known value for case-level attributes (if needed)
        for col in cols_to_pad:
            dtype = df.loc(axis=1)[col].dtype
            if dtype in ['float', 'int'] and not isinstance(backward_pad, (float, int)):
                df.loc[0 : n_pad - 1, col] = 0
            else:
                df.loc[0 : n_pad - 1, col] = backward_pad
                
        df[case_identifier] = df[case_identifier].bfill()
    if forward_pad:
        df = df.reset_index(drop=True).reindex(range(0, len(df)+n_pad)).reset_index(drop=True)
        
        # TODO
        # include functionality to pad with first known value for case-level attributes
        for col in cols_to_pad:
            dtype = df.loc(axis=1)[col].dtype
            if dtype in ['float', 'int'] and not isinstance(forward_pad, (float, int)):
                df.loc[len(df) - n_pad :, col] = 0
            else:
                df.loc[len(df) - n_pad :, col] = forward_pad
        
        df[case_identifier] = df[case_identifier].ffill()
    
    return df

def _batch_samples(samples, nbatches: int):
    nsamples = len(samples)
    batchsize = math.ceil(nsamples/nbatches)
    for idx in range(0, nsamples, batchsize):
        yield samples[idx:min(idx + batchsize, nsamples)]

def _calculate_tsmn(df: pd.DataFrame, timestamp_identifier: str):
    tsmn = pd.Series(
        [
            ts.hour * 60 * 60 + ts.minute * 60 + ts.second
            for ts in df[timestamp_identifier]
        ], index=df.index
    )
    tsmn.name = "tsmn"
    return tsmn


def _calculate_tscs(df: pd.DataFrame, timestamp_identifier: str):
    start_time = min(df[timestamp_identifier])
    difference = pd.to_datetime(df[timestamp_identifier]) - start_time
    tscs = difference.apply(dt.timedelta.total_seconds).astype(int)
    tscs.name = "tscs"
    return tscs


def _calculate_tsle(df: pd.DataFrame, timestamp_identifier: str):
    if len(df) > 1:
        time_lagged = pd.to_datetime(df[timestamp_identifier]).shift(1).bfill()
        difference = pd.to_datetime(df[timestamp_identifier]) - time_lagged
        tsle = difference.apply(dt.timedelta.total_seconds).astype(int)
    else:
        tsle = (pd.to_datetime(df[timestamp_identifier]) - pd.to_datetime(df[timestamp_identifier])).apply(dt.timedelta.total_seconds).astype(int)
    tsle.name = "tsle"
    return tsle