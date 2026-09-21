import pandas as pd
import torch
from ..encoding.util import StrictlyNonNegativeOrdinalEncoder

from torch.utils.data import Dataset
from .sequencedata import SequenceData

# we pad with first seen value in pre-padding and with
# last known value in post-padding (TSLE is pre/post-padded with 0)
PAD_IDX = 0  # reserved padding / "no event yet" index for activities & roles

class EventLogDatasetPhase(Dataset):
    
    def __init__(self, activities, roles, phases, times, next_activity, next_role, next_phase, next_time):
        self.activities = activities
        self.roles = roles
        self.phases = phases
        self.times = times
        self.next_activity = next_activity
        self.next_phase = next_phase
        self.next_role = next_role
        self.next_time = next_time

    def __len__(self):
        return len(self.activities)

    def __getitem__(self, idx):
        return (
            self.activities[idx],
            self.roles[idx],
            self.phases[idx],
            self.times[idx],
            self.next_activity[idx],
            self.next_role[idx],
            self.next_phase[idx],
            self.next_time[idx],
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

class EventLogParserPhase:
    def __init__(self, case_col: str, activity_col: str, timestamp_col: str, resource_col: str, phase_col: str, hidden_dim_hmm: int,  seq_len: int = 5, seed: int = 42, timestamp_format: str = None):
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
        
        self.time_scaler = None

    def load_and_prepare_data(self, data_train: SequenceData, data_val: SequenceData, data_test: SequenceData, phase_annotated_dataset: pd.DataFrame, min_prefix_len: int, include_phases: bool, embedding_training: bool):
        
        df_train = data_train.data.reset_index(drop=True)
        df_val = data_val.data.reset_index(drop=True)
        df_test = data_test.data.reset_index(drop=True)
        
        
        self.activity_vocab = data_train.act_encoder
        
        if embedding_training:
            # role_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            role_encoder = StrictlyNonNegativeOrdinalEncoder()
            encoded_roles = role_encoder.fit_transform(data_train.data['role'].to_numpy().reshape(-1, 1))
            data_train.data['role'] = encoded_roles
            df_train['role'] = encoded_roles
            
            encoded_roles_test = role_encoder.transform(data_test.data['role'].to_numpy().reshape(-1, 1))
            data_test.data['role'] = encoded_roles_test
            df_test['role'] = encoded_roles_test
            
            data_train.role_encoder = role_encoder
            
            self.role_vocab = (role_encoder, ['role'])
            self.role_vocab_col_idx = 0
        else:
            try:
                resource_encoder_key = [enc_key for enc_key, (enc, cols) in data_train.encoders.items() if self.resource_col in cols][0]
                resource_col_enc_index = data_train.encoders[resource_encoder_key][1].index(self.resource_col)
            except KeyError as e:
                e.args("No resource column given in data preparation")
                raise
            
            self.role_vocab = data_train.encoders[resource_encoder_key]
            self.role_vocab_col_idx = resource_col_enc_index
        
        # we do splitting only for test data - we split the dataframe before entering the LSTM module
        train_case_ids = df_train[self.case_col].unique().tolist()
        val_case_ids = df_val[self.case_col].unique().tolist()
        test_case_ids = df_test[self.case_col].unique().tolist()
        
        df_train["_relative_time_s"] = df_train['tsle']
        df_train["_activity_idx"] = df_train[self.activity_col]
        df_train["_role_idx"] = df_train[self.resource_col]
        
        df_val["_relative_time_s"] = df_val['tsle']
        df_val["_activity_idx"] = df_val[self.activity_col]
        df_val["_role_idx"] = df_val[self.resource_col]
        df_val["_phase_idx"] = phase_annotated_dataset[phase_annotated_dataset[self.case_col].isin(val_case_ids)][self.phase_col].reset_index(drop=True)
        
        df_test["_relative_time_s"] = df_test['tsle']
        df_test["_activity_idx"] = df_test[self.activity_col]
        df_test["_role_idx"] = df_test[self.resource_col]
        df_test["_phase_idx"] = phase_annotated_dataset[phase_annotated_dataset[self.case_col].isin(test_case_ids)][self.phase_col].reset_index(drop=True)
        
        if include_phases:
            df_train["_phase_idx"] = phase_annotated_dataset[phase_annotated_dataset[self.case_col].isin(train_case_ids)][self.phase_col]
            df_val["_phase_idx"] = phase_annotated_dataset[phase_annotated_dataset[self.case_col].isin(val_case_ids)][self.phase_col].reset_index(drop=True)
            df_test["_phase_idx"] = phase_annotated_dataset[phase_annotated_dataset[self.case_col].isin(test_case_ids)][self.phase_col].reset_index(drop=True)
        else:
            df_train["_phase_idx"] = 0
            df_val["_phase_idx"] = 0
            df_test["_phase_idx"] = 0
            self.hidden_dim_hmm = 1
        
        df_train["_scaled_time"] = df_train['tsle']
        df_val["_scaled_time"] = df_val['tsle']
        df_test["_scaled_time"] = df_test['tsle']
        
        splits = {'train': self._build_dataset(df_train, min_prefix_len),
                  'val': self._build_dataset(df_val, min_prefix_len),
                  'test': self._build_dataset(df_test, min_prefix_len),
                  'val_suffix': self._build_dataset_suffix(df_val, min_prefix_len),
                  'test_suffix': self._build_dataset_suffix(df_test, min_prefix_len),
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
        return 1

    def _load_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], format='mixed')
        df = df.sort_values([self.case_col, self.timestamp_col]).reset_index(drop=True)
        return df

    def _split_case_ids(self, case_ids):
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(len(case_ids), generator=generator).tolist()
        shuffled = [case_ids[i] for i in perm]

        n = len(shuffled)
        n_val = max(1, int(n * self.val_split)) if n > 2 else 0
        n_test = max(1, int(n * self.test_split)) if n > 2 else 0

        val_ids = shuffled[:n_val]
        test_ids = shuffled[n_val:]
        return val_ids, test_ids

    def _build_dataset(self, split_df: pd.DataFrame, min_prefix_len: int) -> EventLogDatasetPhase:
        activities_list, roles_list, phases_list, times_list = [], [], [], []
        next_activity_list, next_role_list, next_phase_list, next_time_list = [], [], [], []

        for _, case_df in split_df.groupby(self.case_col):
            acts = case_df["_activity_idx"].tolist()
            roles = case_df["_role_idx"].tolist()
            phases = case_df["_phase_idx"].tolist()
            times = case_df["_scaled_time"].tolist()
    
            if len(acts) < min_prefix_len:
                pass

            for t in range(min_prefix_len, len(acts)):
                hist_acts = acts[:t][-self.seq_len:]
                hist_roles = roles[:t][-self.seq_len:]
                hist_phases = phases[:t][-self.seq_len:]
                hist_times = times[:t][-self.seq_len:]

                pad_len = self.seq_len - len(hist_acts)
                hist_acts = [hist_acts[0]] * pad_len + hist_acts
                hist_roles = [hist_roles[0]] * pad_len + hist_roles
                hist_phases = [hist_phases[0]] * pad_len + hist_phases
                hist_times = [0.0] * pad_len + hist_times

                activities_list.append(hist_acts)
                roles_list.append(hist_roles)
                phases_list.append(hist_phases)
                times_list.append(hist_times)

                next_activity_list.append(acts[t])
                next_role_list.append(roles[t])
                next_phase_list.append(phases[t])
                next_time_list.append(times[t])

        activities = torch.tensor(activities_list, dtype=torch.long)
        roles = torch.tensor(roles_list, dtype=torch.long)
        phases = torch.tensor(phases_list, dtype=torch.long)
        times = torch.tensor(times_list, dtype=torch.float).unsqueeze(-1)  # (N, seq_len, 1)

        next_activity = torch.tensor(next_activity_list, dtype=torch.long)
        next_role = torch.tensor(next_role_list, dtype=torch.long)
        next_phase = torch.tensor(next_phase_list, dtype=torch.long)
        next_time = torch.tensor(next_time_list, dtype=torch.float).unsqueeze(-1)  # (N, 1)

        return EventLogDatasetPhase(activities, roles, phases, times, next_activity, next_role, next_phase, next_time)

    def _build_dataset_suffix(self, split_df: pd.DataFrame, min_prefix_len: int) -> EventLogDatasetPhaseSuffix:
        activities_list, roles_list, phases_list, times_list = [], [], [], []
        activity_suffix_list, role_suffix_list, phase_suffix_list, time_suffix_list = [], [], [], []
        indices_list = []
        lengths_list = []
            
        split_df = split_df.reset_index(drop=True)
        
        MAX_CASE_LEN = split_df.groupby(self.case_col).size().max()

        idx_counter = 0
        
        for case_counter, (_, case_df) in enumerate(split_df.groupby(self.case_col)):
            acts = case_df["_activity_idx"].tolist()
            roles = case_df["_role_idx"].tolist()
            phases = case_df["_phase_idx"].tolist()
            times = case_df["_scaled_time"].tolist()
    
            for t in range(min_prefix_len, len(acts)):
                hist_acts = acts[:t][-self.seq_len:]
                hist_roles = roles[:t][-self.seq_len:]
                hist_phases = phases[:t][-self.seq_len:]
                hist_times = times[:t][-self.seq_len:]
    
                pad_len = self.seq_len - len(hist_acts)
                hist_acts = [hist_acts[0]] * pad_len + hist_acts
                hist_roles = [hist_roles[0]] * pad_len + hist_roles
                hist_phases = [hist_phases[0]] * pad_len + hist_phases
                hist_times = [0.0] * pad_len + hist_times
    
                activities_list.append(hist_acts)
                roles_list.append(hist_roles)
                phases_list.append(hist_phases)
                times_list.append(hist_times)
    
                act_suffix = acts[t:] + [acts[-1]] * (MAX_CASE_LEN - (len(acts) - t))
                role_suffix = roles[t:] + [roles[-1]] * (MAX_CASE_LEN - (len(roles) - t))
                phase_suffix = phases[t:] + [phases[-1]] * (MAX_CASE_LEN - (len(phases) - t))
                time_suffix = times[t:] + [0] * (MAX_CASE_LEN - (len(times) - t))
    
                activity_suffix_list.append(act_suffix)
                role_suffix_list.append(role_suffix)
                phase_suffix_list.append(phase_suffix)
                time_suffix_list.append(time_suffix)
    
                indices_list.append(idx_counter)
                lengths_list.append(t+1)
                
                idx_counter += 1
                
        activities = torch.tensor(activities_list, dtype=torch.long)
        roles = torch.tensor(roles_list, dtype=torch.long)
        phases = torch.tensor(phases_list, dtype=torch.long)
        times = torch.tensor(times_list, dtype=torch.float).unsqueeze(-1)  # (N, seq_len, 1)
    
        activity_suffix = torch.tensor(activity_suffix_list, dtype=torch.long)
        role_suffix = torch.tensor(role_suffix_list, dtype=torch.long)
        phase_suffix = torch.tensor(phase_suffix_list, dtype=torch.long)
        time_suffix = torch.tensor(time_suffix_list, dtype=torch.float).unsqueeze(-1)  # (N, 1)
        indices = torch.tensor(indices_list, dtype=torch.long)
        lengths = torch.tensor(lengths_list, dtype=torch.long)
    
        return EventLogDatasetPhaseSuffix(activities, roles, phases, times, activity_suffix, role_suffix, phase_suffix, time_suffix, indices, lengths)