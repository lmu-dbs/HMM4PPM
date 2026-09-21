import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from .sequencedata import SequenceData

# we pad with first seen value in pre-padding and with
# last known value in post-padding (TSLE is pre/post-padded with 0)
PAD_IDX = 0  # reserved padding / "no event yet" index for activities & roles

class TreeEventLogDatasetPhase(Dataset):
    def __init__(self, activities, resources, phases, tsles, tscss, next_activity, next_resource, next_phase, next_tsle, next_tscs):
        self.activities = activities
        self.resources = resources
        self.phases = phases
        self.tsles = tsles
        self.tscss = tscss
        self.next_activity = next_activity
        self.next_phase = next_phase
        self.next_resource = next_resource
        self.next_tsle = next_tsle
        self.next_tscs = next_tscs

    def __len__(self):
        return len(self.activities)

    def __getitem__(self, idx):
        return (
            self.activities[idx],
            self.resources[idx],
            self.phases[idx],
            self.tsles[idx],
            self.tscss[idx],
            self.next_activity[idx],
            self.next_resource[idx],
            self.next_phase[idx],
            self.next_tsle[idx],
            self.next_tscs[idx],
        )

class EventLogDatasetPhaseSuffix(Dataset):
    def __init__(self, activities, roles, phases, times, activity_suffix, role_suffix, phase_suffix, time_suffix, indices, lengths):
        self.activities = activities
        self.roles = roles
        self.phases = phases
        self.times = times
        self.activity_suffix = activity_suffix
        self.role_suffix = role_suffix
        self.phase_suffix = phase_suffix
        self.time_suffix = time_suffix
        self.indices = indices
        self.lengths = lengths

    def __len__(self):
        return len(self.activities)

    def __getitem__(self, idx):
        return (
            self.activities[idx],
            self.roles[idx],
            self.phases[idx],
            self.times[idx],
            self.activity_suffix[idx],
            self.role_suffix[idx],
            self.phase_suffix[idx],
            self.time_suffix[idx],
            self.indices[idx],
            self.lengths[idx],
        )

class TreeEventLogParserPhase:
    def __init__(self, case_col: str, activity_col: str, timestamp_col: str, resource_col: str, phase_col: str, hidden_dim_hmm: int, seq_len: int = 5, seed: int = 42, timestamp_format: str = None):
        self.case_col = case_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.resource_col = resource_col
        self.phase_col = phase_col
        self.seq_len = seq_len
        self.seed = seed
        self.timestamp_format = timestamp_format

        self.activity_vocab = None
        self.role_vocab = None
        self.role_vocab_col_idx = None
        self.hidden_dim_hmm = hidden_dim_hmm

    def load_and_prepare_data(self, data_train: SequenceData, data_val: SequenceData, data_test: SequenceData, phase_annotated_dataset: pd.DataFrame, min_prefix_len: int, include_phases: bool):
        
        df_train = data_train.data.reset_index(drop=True)
        df_val = data_val.data.reset_index(drop=True)
        df_test = data_test.data.reset_index(drop=True)
        
        self.activity_vocab = data_train.act_encoder

        try:
            resource_encoder_key = [enc_key for enc_key, (enc, cols) in data_train.encoders.items() if self.resource_col in cols][0]
            resource_col_enc_index = data_train.encoders[resource_encoder_key][1].index(self.resource_col)
        except KeyError as e:
            e.args("No resource column given in data preparation")
            raise
        
        self.role_vocab = data_train.encoders[resource_encoder_key]
        self.role_vocab_col_idx = resource_col_enc_index
        
        if not include_phases:
            df_train["phase"] = 0
            df_val["phase"] = 0
            df_test["phase"] = 0
            self.hidden_dim_hmm = 1
        
        splits = {'train': self._build_dataset(df_train, min_prefix_len),
                  'val': self._build_dataset(df_val, min_prefix_len),
                  'test': self._build_dataset(df_test, min_prefix_len),
                  }
        
        return splits

    @property
    def num_activities(self):
        return len(self.activity_vocab.categories_[0]) + 2 # + 2 because of StrictlyNonNegativeOrdinalEncoder

    @property
    def num_roles(self):
        return len(self.role_vocab[0].categories_[self.role_vocab_col_idx]) + 2 # + 2 because of StrictlyNonNegativeOrdinalEncoder

    @property
    def num_phases(self):
        return self.hidden_dim_hmm
    
    @property
    def num_time_features(self):
        return 2

    def _build_dataset(self, split_df: pd.DataFrame, min_prefix_len: int) -> TreeEventLogDatasetPhase:
        activities_list, resources_list, phases_list, tsle_list, tscs_list = [], [], [], [], []
        next_activity_list, next_resource_list, next_phase_list, next_tsle_list, next_tscs_list = [], [], [], [], []

        for _, case_df in split_df.groupby(self.case_col):
            acts = case_df[self.activity_col].tolist()
            resources = case_df[self.resource_col].tolist()
            phases = case_df['phase'].tolist()
            tsles = case_df['tsle'].tolist()
            tscss = case_df['tscs'].tolist()
    
            if len(acts) < min_prefix_len:
                pass

            for t in range(min_prefix_len, len(acts)):
                hist_acts = acts[:t][-self.seq_len:]
                hist_resources = resources[:t][-self.seq_len:]
                hist_phases = phases[:t][-self.seq_len:]
                hist_tsles = tsles[:t][-self.seq_len:]
                hist_tscss = tscss[:t][-self.seq_len:]

                pad_len = self.seq_len - len(hist_acts)
                hist_acts = [hist_acts[0]] * pad_len + hist_acts
                hist_resources = [hist_resources[0]] * pad_len + hist_resources
                hist_phases = [hist_phases[0]] * pad_len + hist_phases
                hist_tsles = [0.0] * pad_len + hist_tsles
                hist_tscss = [0.0] * pad_len + hist_tscss

                activities_list.append(hist_acts)
                resources_list.append(hist_resources)
                phases_list.append(hist_phases)
                tsle_list.append(hist_tsles)
                tscs_list.append(hist_tscss)

                next_activity_list.append(acts[t])
                next_resource_list.append(resources[t])
                next_phase_list.append(phases[t])
                next_tsle_list.append(tsles[t])
                next_tscs_list.append(tscss[t])

        activities = np.array(activities_list)
        resources = np.array(resources_list)
        phases = np.array(phases_list)
        tsles = np.array(tsle_list)  # (N, seq_len, 1)
        tscss = np.array(tscs_list)  # (N, seq_len, 1)
        
        next_activity = np.array(next_activity_list)
        next_resource = np.array(next_resource_list)
        next_phase = np.array(next_phase_list)
        next_tsle = np.array(next_tsle_list)  # (N, 1)
        next_tscs = np.array(next_tscs_list)  # (N, 1)
        
        return TreeEventLogDatasetPhase(activities, resources, phases, tsles, tscss, next_activity, next_resource, next_phase, next_tsle, next_tscs)