from tqdm import tqdm
from ..models.hmm import MultivariateHMM
import torch
from torch.special import logsumexp
import pandas as pd
import os

import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

import pyro
import pyro.distributions as dist

from typing import Literal

from .posterior_distributions import inspect_posterior_distribution, inspect_state_paths, inspect_all_posterior_distributions, inspect_all_posterior_distributions_phases_only

from ..util.logging import init_logging
logger = init_logging(__name__, 'prediction.log')

class HMMPredictor():

    def __init__(self, model, pred_args):
        
        self.model = model
        self.pred_args = pred_args
        
        self.model_args_list = list()
        self.predicted_sequences = list()
        self.predicted_future_sequences = list()
        self.filtered_sequences_train = list()
        self.filtered_sequences_test = list()
        
        self.probs_trans_ho, self.state_grid = build_higher_order_transition_matrix(
                    self.model.trained_params['AutoDelta.probs_x'],
                    self.model.trained_params['AutoDelta.probs_initial'].size(0),
                    self.model.train_args['markov_order'],
                    self.model.train_args['log_prob'],
                )

    def predict(self, n_batches: int = 1):
        
        self.predicted_sequences = list()
        
        probs_transition = self.model.trained_params['AutoDelta.probs_x']
        probs_initial = self.model.trained_params['AutoDelta.probs_initial']
        hidden_dim = probs_initial.size(0)
        log_prob=self.model.model_args_test['log_prob']
        markov_order=self.model.model_args_test['markov_order']
        
        probs_emission_discrete = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.probs_y')]
        probs_emission_continuous_loc = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.loc_y')]
        probs_emission_continuous_scale = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.scale_y')]
        
        probs_emission_continuous_alpha = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.alpha_y')]
        probs_emission_continuous_theta = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.theta_y')]
        
        probs_emission_continuous_alphazero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.alphazero_y')]
        probs_emission_continuous_thetazero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.thetazero_y')]
        probs_emission_continuous_zerozero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.zerozero_y')]
        
        probs_emission_continuous_normal = [(l, s) for l, s in zip(probs_emission_continuous_loc, probs_emission_continuous_scale)]
        probs_emission_continuous_gamma = [(a, t) for a, t in zip(probs_emission_continuous_alpha, probs_emission_continuous_theta)]
        probs_emission_continuous_gammazero = [(a, t, z) for a, t, z in zip(probs_emission_continuous_alphazero, probs_emission_continuous_thetazero, probs_emission_continuous_zerozero)]
        
        probs_emission = probs_emission_discrete + probs_emission_continuous_normal + probs_emission_continuous_gamma + probs_emission_continuous_gammazero
        
        batch_indices = _get_batch_indices(self.model.model_args_test['sequences'].shape[0], n_batches)
        for idx, i in enumerate(batch_indices):
            
            seq_batch = self.model.model_args_test['sequences'][i,:,:]
            len_batch = self.model.model_args_test['lengths'][i]
            
            self.predicted_sequences = hmm_sample_future_multivariate_batched(
                # sequences=self.model.model_args_test['sequences'],
                # lengths=self.model.model_args_test['lengths'],
                sequences=seq_batch,
                lengths=len_batch,
                probs_initial=probs_initial,
                probs_transition=probs_transition,
                probs_emission=probs_emission,
                emission_types=self.model.model_args_train['emission_types'],
                markov_order=markov_order,
                n_pred_steps=self.model.model_args_test['max_pred_length'],
                num_samples=self.model.model_args_test['num_samples'],
                log_prob=log_prob,
                batch_idx=idx,
                )
            
            formatted_batch_preds = self.format_predictions_batched()
            
            self.predicted_future_sequences.extend(formatted_batch_preds)
                    
    def format_predictions_batched(self, aggregate_samples: bool = True, uncertainty: bool = True):
    
        assert isinstance(self.predicted_sequences, torch.Tensor), "predicted sequences need to be provided as batched tensor of predicted sequences (possibly num_samples > 1)"
        
        if self.model.model == MultivariateHMM:

            assert self.predicted_sequences.dim() == 4, "shape of predictions needs to be 3-dimensional (with shape [num_samples, num_sequences, n_channels, prediction_length]), even if num_samples = 1"
            
            num_sequences = self.predicted_sequences.shape[1]
            num_steps = self.predicted_sequences.shape[3]
            
            agg_emissions = [[_pick_mode(self.predicted_sequences[:, seq, 0, step]) 
                                for step in range(num_steps)] 
                                for seq in range(num_sequences)]
        else:
            raise NotImplementedError()
            
        return agg_emissions
        
    def format_predictions(self, aggregate_samples: bool = True, uncertainty: bool = True):

        assert isinstance(self.predicted_sequences, list), "predicted sequences need to be provided as list of predicted sequences (possibly num_samples > 1)"
        formatted_preds = list()

        if self.model.model == MultivariateHMM:
            
            for pred_seq in self.predicted_sequences:
                assert pred_seq.dim() == 3, "shape of predictions needs to be 3-dimensional (with shape [n_channels, num_samples, prediction_length]), even if num_samples = 1"

                if aggregate_samples:
                    agg_emissions = [_pick_mode(pred_seq[0, :, pred_idx], report_prob=uncertainty) for pred_idx in range(pred_seq.shape[-1])]
                    formatted_preds.append(agg_emissions)
                elif not aggregate_samples and pred_seq.shape[0] > 1:
                    raise NotImplementedError("keeping all samples for num_samples > 1 not yet implemented!")
                
        else:
            raise NotImplementedError()

        self.predicted_future_sequences = formatted_preds

    def filter_external(self, sequences: torch.Tensor, lengths: torch.Tensor, marginal_state_probs: bool, log: bool = False):
        """Function for filtering sequences from an external call/module
        CAUTION: SHOULD NOT BE USED INSIDE PROCESSHHMM MODULE
        """
        probs_transition = self.model.trained_params['AutoDelta.probs_x']
        probs_initial = self.model.trained_params['AutoDelta.probs_initial']
        hidden_dim = probs_initial.size(0)
        log_prob=self.model.train_args['log_prob']
        markov_order=self.model.train_args['markov_order']

        probs_emission_discrete = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.probs_y')]
        probs_emission_continuous_loc = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.loc_y')]
        probs_emission_continuous_scale = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.scale_y')]
        
        probs_emission_continuous_alpha = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.alpha_y')]
        probs_emission_continuous_theta = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.theta_y')]
        
        probs_emission_continuous_alphazero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.alphazero_y')]
        probs_emission_continuous_thetazero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.thetazero_y')]
        probs_emission_continuous_zerozero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.zerozero_y')]
        
        probs_emission_continuous_normal = [(l, s) for l, s in zip(probs_emission_continuous_loc, probs_emission_continuous_scale)]
        probs_emission_continuous_gamma = [(a, t) for a, t in zip(probs_emission_continuous_alpha, probs_emission_continuous_theta)]
        probs_emission_continuous_gammazero = [(a, t, z) for a, t, z in zip(probs_emission_continuous_alphazero, probs_emission_continuous_thetazero, probs_emission_continuous_zerozero)]
        
        probs_emission = probs_emission_discrete + probs_emission_continuous_normal + probs_emission_continuous_gamma + probs_emission_continuous_gammazero

        
        filtered_sequences_tensor, _ = hmm_filtering_multivariate_batched(sequences,
                                                               lengths,
                                                               probs_initial=probs_initial,
                                                               probs_transition=probs_transition,
                                                               probs_emission=probs_emission,
                                                               emission_types=self.model.model_args_train['emission_types'], 
                                                               markov_order=markov_order,
                                                               log_prob=log_prob,
                                                               marginal_state_probs=marginal_state_probs, # needs to be False - we need the actual paths through the previous phases for order > 1 
                                                               probs_trans_ho=self.probs_trans_ho,
                                                               state_grid=self.state_grid,
                                                               log=log
                                                               )
        
        return filtered_sequences_tensor, self.state_grid
                
    def filter(self, n_batches: int = 1):
        
        self.filtered_sequences_train = list()
        self.filtered_sequences_val = list()
        self.filtered_sequences_test = list()
        
        probs_transition = self.model.trained_params['AutoDelta.probs_x']
        probs_initial = self.model.trained_params['AutoDelta.probs_initial']
        hidden_dim = probs_initial.size(0)
        log_prob=self.model.model_args_test['log_prob']
        markov_order=self.model.model_args_test['markov_order']
        
        probs_emission_discrete = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.probs_y')]
        probs_emission_continuous_loc = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.loc_y')]
        probs_emission_continuous_scale = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.scale_y')]
        
        probs_emission_continuous_alpha = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.alpha_y')]
        probs_emission_continuous_theta = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.theta_y')]
        
        probs_emission_continuous_alphazero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.alphazero_y')]
        probs_emission_continuous_thetazero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.thetazero_y')]
        probs_emission_continuous_zerozero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.zerozero_y')]
        
        probs_emission_continuous_normal = [(l, s) for l, s in zip(probs_emission_continuous_loc, probs_emission_continuous_scale)]
        probs_emission_continuous_gamma = [(a, t) for a, t in zip(probs_emission_continuous_alpha, probs_emission_continuous_theta)]
        probs_emission_continuous_gammazero = [(a, t, z) for a, t, z in zip(probs_emission_continuous_alphazero, probs_emission_continuous_thetazero, probs_emission_continuous_zerozero)]
        
        probs_emission = probs_emission_discrete + probs_emission_continuous_normal + probs_emission_continuous_gamma + probs_emission_continuous_gammazero

        batch_indices_test = _get_batch_indices(self.model.model_args_test['sequences'].shape[0], n_batches)
        
        for idx, i in enumerate(batch_indices_test):
            seq_batch = self.model.model_args_test['sequences'][i,:,:]
            len_batch = self.model.model_args_test['lengths'][i]
        
            filter_tensor_test, _ = hmm_filtering_multivariate_batched(
                # self.model.model_args_test['sequences'], 
                # self.model.model_args_test['lengths'], 
                seq_batch, 
                len_batch, 
                probs_initial=probs_initial,
                probs_transition=probs_transition,
                probs_emission=probs_emission,
                emission_types=self.model.model_args_train['emission_types'], 
                markov_order=markov_order,
                log_prob=log_prob,
                marginal_state_probs=True,
                probs_trans_ho=self.probs_trans_ho,
                state_grid=self.state_grid,
                batch_idx=idx,
                )
        
        
            for seq_idx in range(filter_tensor_test.shape[1]):
                # seq_obs_len = self.model.model_args_test['lengths'][seq_idx] + 1
                seq_obs_len = len_batch[seq_idx] + 1
                self.filtered_sequences_test.append(filter_tensor_test[:seq_obs_len, seq_idx, :])
        
        batch_indices_val = _get_batch_indices(self.model.model_args_val['sequences'].shape[0], n_batches)
        
        for idx, i in enumerate(batch_indices_val):
            seq_batch = self.model.model_args_val['sequences'][i,:,:]
            len_batch = self.model.model_args_val['lengths'][i]
        
            filter_tensor_val, _ = hmm_filtering_multivariate_batched(
                # self.model.model_args_test['sequences'], 
                # self.model.model_args_test['lengths'], 
                seq_batch, 
                len_batch, 
                probs_initial=probs_initial,
                probs_transition=probs_transition,
                probs_emission=probs_emission,
                emission_types=self.model.model_args_train['emission_types'], 
                markov_order=markov_order,
                log_prob=log_prob,
                marginal_state_probs=True,
                probs_trans_ho=self.probs_trans_ho,
                state_grid=self.state_grid,
                batch_idx=idx,
                )
        
        
            for seq_idx in range(filter_tensor_val.shape[1]):
                # seq_obs_len = self.model.model_args_test['lengths'][seq_idx] + 1
                seq_obs_len = len_batch[seq_idx] + 1
                self.filtered_sequences_val.append(filter_tensor_val[:seq_obs_len, seq_idx, :])
        
        batch_indices_train = _get_batch_indices(self.model.model_args_train['sequences'].shape[0], n_batches)

        for idx, i in enumerate(batch_indices_train):
            seq_batch = self.model.model_args_train['sequences'][i,:,:]
            len_batch = self.model.model_args_train['lengths'][i]
            
            filter_tensor_train, _ = hmm_filtering_multivariate_batched(
                # self.model.model_args_train['sequences'], 
                # self.model.model_args_train['lengths'], 
                seq_batch, 
                len_batch, 
                probs_initial=probs_initial,
                probs_transition=probs_transition,
                probs_emission=probs_emission,
                emission_types=self.model.model_args_train['emission_types'], 
                markov_order=markov_order,
                log_prob=log_prob,
                marginal_state_probs=True,
                probs_trans_ho=self.probs_trans_ho,
                state_grid=self.state_grid,
                batch_idx=idx,
                )
            
            for seq_idx in range(filter_tensor_train.shape[1]):
                # seq_obs_len = self.model.model_args_train['lengths'][seq_idx] + 1
                seq_obs_len = len_batch[seq_idx]
                self.filtered_sequences_train.append(filter_tensor_train[:seq_obs_len, seq_idx, :])
                
    def filter_val(self, n_batches: int = 1):

        self.filtered_sequences_val = list()

        probs_transition = self.model.trained_params['AutoDelta.probs_x']
        probs_initial = self.model.trained_params['AutoDelta.probs_initial']
        hidden_dim = probs_initial.size(0)
        log_prob=self.model.model_args_test['log_prob']
        markov_order=self.model.model_args_test['markov_order']

        probs_emission_discrete = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.probs_y')]
        probs_emission_continuous_loc = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.loc_y')]
        probs_emission_continuous_scale = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.scale_y')]

        probs_emission_continuous_alpha = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.alpha_y')]
        probs_emission_continuous_theta = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.theta_y')]

        probs_emission_continuous_alphazero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.alphazero_y')]
        probs_emission_continuous_thetazero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.thetazero_y')]
        probs_emission_continuous_zerozero = [v for k,v in self.model.trained_params.items() if k.startswith('AutoDelta.zerozero_y')]

        probs_emission_continuous_normal = [(l, s) for l, s in zip(probs_emission_continuous_loc, probs_emission_continuous_scale)]
        probs_emission_continuous_gamma = [(a, t) for a, t in zip(probs_emission_continuous_alpha, probs_emission_continuous_theta)]
        probs_emission_continuous_gammazero = [(a, t, z) for a, t, z in zip(probs_emission_continuous_alphazero, probs_emission_continuous_thetazero, probs_emission_continuous_zerozero)]

        probs_emission = probs_emission_discrete + probs_emission_continuous_normal + probs_emission_continuous_gamma + probs_emission_continuous_gammazero

        batch_indices_val = _get_batch_indices(self.model.model_args_val['sequences'].shape[0], n_batches)

        for idx, i in enumerate(batch_indices_val):
            seq_batch = self.model.model_args_val['sequences'][i,:,:]
            len_batch = self.model.model_args_val['lengths'][i]

            filter_tensor_val, _ = hmm_filtering_multivariate_batched(
                # self.model.model_args_test['sequences'], 
                # self.model.model_args_test['lengths'], 
                seq_batch, 
                len_batch, 
                probs_initial=probs_initial,
                probs_transition=probs_transition,
                probs_emission=probs_emission,
                emission_types=self.model.model_args_train['emission_types'], 
                markov_order=markov_order,
                log_prob=log_prob,
                marginal_state_probs=True,
                probs_trans_ho=self.probs_trans_ho,
                state_grid=self.state_grid,
                batch_idx=idx,
                )


            for seq_idx in range(filter_tensor_val.shape[1]):
                # seq_obs_len = self.model.model_args_test['lengths'][seq_idx] + 1
                seq_obs_len = len_batch[seq_idx] + 1
                self.filtered_sequences_val.append(filter_tensor_val[:seq_obs_len, seq_idx, :])
    
    def annotate_data(self, split: Literal['train', 'test', 'all'] = 'all', argmax: bool = True, export_path: os.PathLike = None):
        
        if split=='train':
            dataset = self.model.data_train.data.copy()
            if argmax:
                if self.model.train_args['log_prob']:
                    phases = torch.cat([f_seq.exp().argmax(dim=1) for f_seq in self.filtered_sequences_train])
                else:
                    phases = torch.cat([f_seq.argmax(dim=1) for f_seq in self.filtered_sequences_train])
                assert phases.size()[0] == len(dataset), "filtered phases need to be same length as n rows of dataset"
                dataset = pd.concat((dataset, pd.Series(phases, name='phase')), axis=1)
            else:
                raise NotImplementedError
            
        elif split in ['val', 'test']:
            if split=='val':
                dataset = self.model.data_val.data.copy()
                full_sequence_indices = get_full_info_prefix_indices(self.model.model_args_val['lengths'])
                filtered_sequences = self.filtered_sequences_val
            else:
                dataset = self.model.data_test.data.copy()
                full_sequence_indices = get_full_info_prefix_indices(self.model.model_args_test['lengths'])
                filtered_sequences = self.filtered_sequences_test
            relevant_filtered_sequences = [f_seq for f_seq_idx, f_seq in enumerate(filtered_sequences) if f_seq_idx in full_sequence_indices]
            if argmax:
                if self.model.train_args['log_prob']:
                    phases = torch.cat([f_seq.exp().argmax(dim=1) for f_seq in relevant_filtered_sequences])
                else:
                    phases = torch.cat([f_seq.argmax(dim=1) for f_seq in relevant_filtered_sequences])
                assert phases.size()[0] == len(dataset), "filtered phases need to be same length as n rows of dataset"
                dataset = pd.concat((dataset, pd.Series(phases, name='phase')), axis=1)
            else:
                raise NotImplementedError
        elif split=='all':
            train_dataset = self.model.data_train.data.copy()
            val_dataset = self.model.data_val.data.copy()
            test_dataset = self.model.data_test.data.copy()
            dataset = pd.concat((train_dataset, val_dataset, test_dataset), axis=0).reset_index(drop=True)
            full_sequence_indices_val = get_full_info_prefix_indices(self.model.model_args_val['lengths'])
            full_sequence_indices_test = get_full_info_prefix_indices(self.model.model_args_test['lengths'])
            relevant_filtered_sequences_val = [f_seq for f_seq_idx, f_seq in enumerate(self.filtered_sequences_val) if f_seq_idx in full_sequence_indices_val]
            relevant_filtered_sequences_test = [f_seq for f_seq_idx, f_seq in enumerate(self.filtered_sequences_test) if f_seq_idx in full_sequence_indices_test]
            if argmax:
                if self.model.train_args['log_prob']:
                    phases_train = torch.cat([f_seq.exp().argmax(dim=1) for f_seq in self.filtered_sequences_train])
                    phases_val = torch.cat([f_seq.exp().argmax(dim=1) for f_seq in relevant_filtered_sequences_val])
                    phases_test = torch.cat([f_seq.exp().argmax(dim=1) for f_seq in relevant_filtered_sequences_test])
                else:
                    phases_train = torch.cat([f_seq.argmax(dim=1) for f_seq in self.filtered_sequences_train])
                    phases_val = torch.cat([f_seq.argmax(dim=1) for f_seq in relevant_filtered_sequences_val])
                    phases_test = torch.cat([f_seq.argmax(dim=1) for f_seq in relevant_filtered_sequences_test])
                phases = torch.cat((phases_train, phases_val, phases_test))
                assert phases.size()[0] == len(dataset), "filtered phases need to be same length as n rows of dataset"
                dataset = pd.concat((dataset, pd.Series(phases, name='phase')), axis=1)
            else:
                raise NotImplementedError
        else:
            raise ValueError("split needs to be one of ['train', 'test', 'all']")
        
        if export_path is not None:
            dataset.to_csv(os.path.join(export_path, f"annotated_dataset_{self.model.model.__name__.lower()}_{self.model.param_hash}_{sha[:8]}.csv"))
        
        return dataset
    
    def posterior_predictive_check(self, channel: str, emission_type: str, data_mode: Literal['train','test'], path_mode: Literal['all', 'random', 'seq_id'] = 'random', weighted_sampling: bool = True, num_samples: int = int(1e5)):
        
        full_sequence_indices_test = get_full_info_prefix_indices(self.model.model_args_test['lengths'])
        
        if self.model.train_args['log_prob']:
            alpha_probs_test = [self.filtered_sequences_test[i].exp() for i in full_sequence_indices_test]
            alpha_probs_train = [a.exp() for a in self.filtered_sequences_train]
        else:
            alpha_probs_test = [self.filtered_sequences_test[i] for i in full_sequence_indices_test]
            alpha_probs_train = self.filtered_sequences_train
        
        if weighted_sampling:
            if data_mode=='train':
                alpha = alpha_probs_train
            elif data_mode=='test':
                alpha = alpha_probs_test
        else:
            alpha = None
                
        inspect_posterior_distribution(self.model, channel=channel, emission_type=emission_type, alpha_probs=alpha, mode=data_mode, num_samples=num_samples)

    def complete_posterior_predictive_check(self, channels: str, channel_labels: str, emission_types: str, data_mode: Literal['train','test'], num_samples: int = int(1e5)):
        full_sequence_indices_test = get_full_info_prefix_indices(self.model.model_args_test['lengths'])
        full_sequence_indices_val = get_full_info_prefix_indices(self.model.model_args_val['lengths'])
        
        if self.model.train_args['log_prob']:
            alpha_probs_test = [self.filtered_sequences_test[i].exp() for i in full_sequence_indices_test]
            alpha_probs_train = [a.exp() for a in self.filtered_sequences_train]
            alpha_probs_val = [a.exp() for a in self.filtered_sequences_val]
        else:
            alpha_probs_test = [self.filtered_sequences_test[i] for i in full_sequence_indices_test]
            alpha_probs_val = [self.filtered_sequences_val[i] for i in full_sequence_indices_val]
            alpha_probs_train = self.filtered_sequences_train
        
        if data_mode=='train':
            alpha = alpha_probs_train
        elif data_mode=='test':
            alpha = alpha_probs_test
        elif data_mode=='val':
            alpha = alpha_probs_val
        
        inspect_all_posterior_distributions(self.model, channels=channels, channel_labels=channel_labels, emission_types=emission_types, alpha_probs=alpha, mode=data_mode, num_samples=num_samples)
    
    def complete_posterior_predictive_check_phases_only(self, channels: str, channel_labels: str, emission_types: str, data_mode: Literal['train','test'], num_samples: int = int(1e5), pred_model = None):
        full_sequence_indices_test = get_full_info_prefix_indices(self.model.model_args_test['lengths'])
        full_sequence_indices_val = get_full_info_prefix_indices(self.model.model_args_val['lengths'])

        if self.model.train_args['log_prob']:
            alpha_probs_test = [self.filtered_sequences_test[i].exp() for i in full_sequence_indices_test]
            alpha_probs_train = [a.exp() for a in self.filtered_sequences_train]
            alpha_probs_val = [a.exp() for a in self.filtered_sequences_val]
        else:
            alpha_probs_test = [self.filtered_sequences_test[i] for i in full_sequence_indices_test]
            alpha_probs_val = [self.filtered_sequences_val[i] for i in full_sequence_indices_val]
            alpha_probs_train = self.filtered_sequences_train

        if data_mode=='train':
            alpha = alpha_probs_train
        elif data_mode=='test':
            alpha = alpha_probs_test
        elif data_mode=='val':
            alpha = alpha_probs_val

        inspect_all_posterior_distributions_phases_only(self.model, channels=channels, channel_labels=channel_labels, emission_types=emission_types, alpha_probs=alpha, mode=data_mode, num_samples=num_samples, pred_model=pred_model)
        
    def plot_paths(self, data_mode: Literal['train','test'], path_mode: Literal['all', 'random', 'seq_id'] = 'random', seq_id: int = None, ma_horizon: int = None, only_past: bool = True, labels: bool = True, print_trace: bool = True):
        
        full_sequence_indices_test = get_full_info_prefix_indices(self.model.model_args_test['lengths'])
        
        if self.model.train_args['log_prob']:
            alpha_probs_test = [self.filtered_sequences_test[i].exp() for i in full_sequence_indices_test]
            alpha_probs_train = [a.exp() for a in self.filtered_sequences_train]
        else:
            alpha_probs_test = [self.filtered_sequences_test[i] for i in full_sequence_indices_test]
            alpha_probs_train = self.filtered_sequences_train
        
        if data_mode=='train':
            alpha = alpha_probs_train
            sequences = self.model.model_args_train['sequences']
            lengths = self.model.model_args_train['lengths']
        elif data_mode=='test':
            alpha = alpha_probs_test
            sequences = self.model.model_args_test['sequences'][full_sequence_indices_test]
            lengths = self.model.model_args_test['lengths'][full_sequence_indices_test]
                
        inspect_state_paths(alpha_probs=alpha, sequences=sequences, lengths=lengths, mode=path_mode, seq_id=seq_id, ma_horizon=ma_horizon, only_past=only_past, labels=labels, print_trace=print_trace)

def _pick_mode(x: torch.Tensor, report_prob: bool = True):

    values, counts = torch.unique(x, return_counts=True)
    mode = values[counts.argmax()]
    if report_prob:
        prob = counts.max() / counts.sum()
        return mode.item(), prob.item()
    return mode.item()

def _calc_mean(x: torch.Tensor, report_sd: bool = True):

    mean = x.mean()
    if report_sd:
        if len(x) > 1:
            std = x.std()
        else:
            std = torch.tensor(torch.nan)
        return mean.item(), std.item()
    return mean.item()

def hmm_filtering(obs_seq, probs_initial, probs_transition, probs_emission, markov_order, log_prob, marginal_state_probs: bool = False, probs_trans_ho = None, state_grid = None):
    T = len(obs_seq)
    hidden_dim = probs_initial.size(0) # number of hidden states

    if probs_trans_ho is None or state_grid is None:

        probs_trans_ho, state_grid = build_higher_order_transition_matrix(
            probs_transition,
            hidden_dim,
            markov_order,
            log_prob
        )

    n_paths = hidden_dim ** markov_order

    if log_prob:
        alpha = torch.full((T, n_paths), -torch.inf)

        # initial forward calculation
        joint_init = torch.zeros(n_paths)

        for i in range(markov_order):
            joint_init += probs_initial[state_grid[:, i]].log()

        joint_init += probs_emission[state_grid[:, -1], obs_seq[0].long()].log()

        alpha[0] = joint_init - logsumexp(joint_init, dim=0)

    else:

        alpha = torch.zeros(T, n_paths)

        # initial forward calculation

        joint_init = torch.ones(n_paths)

        for i in range(markov_order):
            joint_init *= probs_initial[state_grid[:, i]]

        joint_init *= probs_emission[state_grid[:, -1], obs_seq[0].long()]
        # alpha[0] = joint_init / joint_init.sum().clamp_min(1e-12)
        alpha[0] = joint_init / joint_init.sum() # turning off clamping for now

    # remaining recursion

    for t in range(1, T):

        if log_prob:
            alpha[t] = logsumexp(alpha[t - 1].unsqueeze(1) + probs_trans_ho, dim=0)
            alpha[t] += probs_emission[state_grid[:, -1], obs_seq[t].long()].log()
            alpha[t] -= logsumexp(alpha[t], dim=0)
        else:
            alpha[t] = alpha[t - 1] @ probs_trans_ho
            alpha[t] *= probs_emission[state_grid[:, -1], obs_seq[t].long()]
            alpha[t] /= alpha[t].sum()
            
    if marginal_state_probs:
        # we need to select all paths for a specific end state
        # e.g. with a 3 state HMM of order 2 and the probs of being in state 1
        # 0 -> 1, 1 -> 1, 2 -> 1 and marginalize
        
        current_state_idx = state_grid[:, -1]
        state_indices = current_state_idx.unique()
        
        if log_prob:
            out = torch.full((T, hidden_dim), -torch.inf)
            for s in state_indices:
                mask = (current_state_idx == s)
                out[:, s] = logsumexp(alpha[:, mask], dim=1)
        else:
            out = torch.zeros((T, hidden_dim))
            for s in state_indices:
                mask = (current_state_idx == s)
                out[:, s] = alpha[:, mask].sum(dim=1)
        
        alpha = out

    return alpha, state_grid

def hmm_filtering_multivariate(obs_seq, probs_initial, probs_transition, probs_emission, emission_types, markov_order, log_prob, marginal_state_probs: bool = False, probs_trans_ho = None, state_grid = None):
    
    n_channels, T = obs_seq.shape
    hidden_dim = probs_initial.size(0)
    
    assert len(probs_emission) == n_channels, \
        "probs_emission needs one tensor per channel (obs_seq.shape[1])"

    if probs_trans_ho is None or state_grid is None:
        probs_trans_ho, state_grid = build_higher_order_transition_matrix(
            probs_transition,
            hidden_dim,
            markov_order,
            log_prob,
        )

    n_paths = hidden_dim ** markov_order

    def joint_emission(states, obs_t):
        """states: (n_paths,) last-state index per path. obs_t: (n_channels,) obs at time t.
        Returns (n_paths,) joint (product/log-sum across channels) emission prob per path.
        Discrete channels contribute a pmf lookup, continuous channels a Normal pdf."""
        if log_prob:
            total = torch.zeros(states.shape[0])
            for c, (etype, params) in enumerate(zip(emission_types, probs_emission)):
                if etype == 'discrete':
                    total += params[states, obs_t[c].long()].log()
                elif etype == 'gamma':
                    alpha_c, theta_c = params
                    total += torch.distributions.Gamma(alpha_c[states], theta_c[states]).log_prob(obs_t[c])
                elif etype == 'continuous':
                    loc_c, scale_c = params
                    total += torch.distributions.Normal(loc_c[states], scale_c[states]).log_prob(obs_t[c])
                else:
                    raise ValueError(f"unknown emission type: {etype}")
            return total
        else:
            total = torch.ones(states.shape[0])
            for c, (etype, params) in enumerate(zip(emission_types, probs_emission)):
                if etype == 'discrete':
                    total *= params[states, obs_t[c].long()]
                elif etype == 'gamma':
                    alpha_c, theta_c = params
                    total *= torch.distributions.Gamma(alpha_c[states], theta_c[states]).log_prob(obs_t[c]).exp()
                elif etype == 'continuous':
                    loc_c, scale_c = params
                    total *= torch.distributions.Normal(loc_c[states], scale_c[states]).log_prob(obs_t[c]).exp()
                else:
                    raise ValueError(f"unknown emission type: {etype}")
            return total


    if log_prob:
        alpha = torch.full((T, n_paths), -torch.inf)

        joint_init = torch.zeros(n_paths)
        for i in range(markov_order):
            joint_init += probs_initial[state_grid[:, i]].log()

        joint_init += joint_emission(state_grid[:, -1], obs_seq[:, 0])

        alpha[0] = joint_init - logsumexp(joint_init, dim=0)

    else:
        alpha = torch.zeros(T, n_paths)

        joint_init = torch.ones(n_paths)
        for i in range(markov_order):
            joint_init *= probs_initial[state_grid[:, i]]

        joint_init *= joint_emission(state_grid[:, -1], obs_seq[:, 0])
        alpha[0] = joint_init / joint_init.sum()

    for t in range(1, T):
        if log_prob:
            alpha[t] = logsumexp(alpha[t - 1].unsqueeze(1) + probs_trans_ho, dim=0)
            alpha[t] += joint_emission(state_grid[:, -1], obs_seq[:, t])
            alpha[t] -= logsumexp(alpha[t], dim=0)
        else:
            alpha[t] = alpha[t - 1] @ probs_trans_ho
            alpha[t] *= joint_emission(state_grid[:, -1], obs_seq[:, t])
            alpha[t] /= alpha[t].sum()
    
    if marginal_state_probs:
        # we need to select all paths for a specific end state
        # e.g. with a 3 state HMM of order 2 and the probs of being in state 1
        # 0 -> 1, 1 -> 1, 2 -> 1 and marginalize
        
        current_state_idx = state_grid[:, -1]
        state_indices = current_state_idx.unique()
        
        if log_prob:
            out = torch.full((T, hidden_dim), -torch.inf)
            for s in state_indices:
                mask = (current_state_idx == s)
                out[:, s] = logsumexp(alpha[:, mask], dim=1)
        else:
            out = torch.zeros((T, hidden_dim))
            for s in state_indices:
                mask = (current_state_idx == s)
                out[:, s] = alpha[:, mask].sum(dim=1)
        
        alpha = out

    return alpha, state_grid

def hmm_filtering_multivariate_batched(sequences, lengths, probs_initial, probs_transition, probs_emission, emission_types, markov_order, log_prob, marginal_state_probs: bool = False, probs_trans_ho = None, state_grid = None, batch_idx: int = None, log: bool = True):
    
    num_sequences, n_channels, T = sequences.shape
    
    if batch_idx is not None:
        batch_output = f"(Batch {batch_idx}) "
    else:
        batch_output = ""
    
    assert len(probs_emission) == n_channels, "probs_emission needs one tensor per channel (obs_seq.shape[1])"
    
    if probs_trans_ho is None or state_grid is None:
        probs_trans_ho, state_grid = build_higher_order_transition_matrix(
            probs_transition,
            hidden_dim,
            markov_order,
            log_prob,
        )
        
    hidden_dim = probs_initial.size(0)
    n_paths = hidden_dim ** markov_order
    
    if log_prob:
        alpha = torch.full((T, num_sequences, n_paths), -torch.inf)
        joint_init = torch.zeros(n_paths)
        for i in range(markov_order):
            joint_init += probs_initial[state_grid[:, i]].log()
        
        joint_init = joint_init.unsqueeze(0).expand(num_sequences, n_paths).clone()  # (n_paths) -> (num_sequences, n_paths)
        
        # alpha for first index
        joint_init = joint_init + joint_emission_batched(
            state_grid[:, -1], sequences[:, :, 0], emission_types, probs_emission, log_prob
        )
        alpha[0] = joint_init - logsumexp(joint_init, dim=1, keepdim=True)
        
        if log:
            progress = tqdm(range(1, T), desc=f'{batch_output}Filtering for T={T} maximum sequence length')
        else:
            progress = range(1, T)
        for t in progress:
            alpha_ctx = alpha[t - 1].view(num_sequences, *([hidden_dim] * markov_order))
            step = alpha_ctx.unsqueeze(-1) + probs_transition.log().unsqueeze(0)
            new_ctx = logsumexp(step, dim=1)
            alpha[t] = new_ctx.reshape(num_sequences, n_paths)
            alpha[t] = alpha[t] + joint_emission_batched(
                state_grid[:, -1], sequences[:, :, t], emission_types, probs_emission, log_prob
            )
            alpha[t] = alpha[t] - logsumexp(alpha[t], dim=1, keepdim=True)

    else:
        alpha = torch.zeros((T, num_sequences, n_paths))
        joint_init = torch.ones(n_paths)
        for i in range(markov_order):
            joint_init *= probs_initial[state_grid[:, i]].log()
        
        joint_init = joint_init.unsqueeze(0).expand(num_sequences, n_paths).clone()  # (n_paths) -> (num_sequences, n_paths)
        
        # alpha for first index
        joint_init = joint_init * joint_emission_batched(
            state_grid[:, -1], sequences[:, :, 0], emission_types, probs_emission, log_prob
        )
        alpha[0] = joint_init / joint_init.sum(dim=1, keepdim=True)
        
        # alpha for following indices
        if log:
            progress = tqdm(range(1, T), desc=f'{batch_output}Filtering for T={T} maximum sequence length')
        else:
            progress = range(1, T)
        for t in progress:
            alpha[t] = alpha[t - 1] @ probs_trans_ho  # (num_sequences, n_paths) @ (n_paths, n_paths)
            alpha[t] = alpha[t] * joint_emission_batched(
                state_grid[:, -1], sequences[:, t], emission_types, probs_emission, log_prob
            )
            alpha[t] = alpha[t] / alpha[t].sum(dim=1, keepdim=True)

    if marginal_state_probs:
        # we need to select all paths for a specific end state
        # e.g. with a 3 state HMM of order 2 and the probs of being in state 1
        # 0 -> 1, 1 -> 1, 2 -> 1 and marginalize
        current_state_idx = state_grid[:, -1]
        state_indices = current_state_idx.unique()
 
        if log_prob:
            out = torch.full((T, num_sequences, hidden_dim), -torch.inf)
            for s in state_indices:
                mask = (current_state_idx == s)
                out[:, :, s] = logsumexp(alpha[:, :, mask], dim=2)
        else:
            out = torch.zeros((T, num_sequences, hidden_dim))
            for s in state_indices:
                mask = (current_state_idx == s)
                out[:, :, s] = alpha[:, :, mask].sum(dim=2)
        alpha = out

    return alpha, state_grid

def joint_emission_batched(states, sequences, emission_types, probs_emission, log_prob):
    """
    states: (n_paths,) last-state index per path.
    sequences:  (B, n_channels) observations at time t for all sequences in the batch.
    Returns (B, n_paths) joint emission prob/log-prob per (sequence, path).
    """
    num_sequences = sequences.shape[0]
    n_paths = states.shape[0]
 
    if log_prob:
        total = torch.zeros(num_sequences, n_paths)
    else:
        total = torch.ones(num_sequences, n_paths)
 
    for c, (etype, params) in enumerate(zip(emission_types, probs_emission)):
        obs_c = sequences[:, c]
        if etype == 'discrete':
            # gather params[states[j], obs_c[n]] for every (n, j) pair at once
            states_exp = states.unsqueeze(0).expand(num_sequences, n_paths)
            obs_exp = obs_c.long().unsqueeze(1).expand(num_sequences, n_paths)
            emission_c = params[states_exp, obs_exp]  # (num_sequences, n_paths)
            total = total + emission_c.log() if log_prob else total * emission_c
        elif etype == 'gamma':
            alpha_c, theta_c = params
            dist = torch.distributions.Gamma(alpha_c[states].unsqueeze(0), theta_c[states].unsqueeze(0))  # (1, n_paths)
            lp = dist.log_prob(obs_c.unsqueeze(1))  # (num_sequences, n_paths)
            total = total + lp if log_prob else total * lp.exp()
        elif etype == 'zeroinflatedgamma':
            alpha_c, theta_c, zero_c = params

            zero_states = zero_c[states].unsqueeze(0)
            alpha_states = alpha_c[states].unsqueeze(0)
            theta_states = theta_c[states].unsqueeze(0)

            obs_exp = obs_c.unsqueeze(1)
            is_zero = (obs_exp == 0)

            safe_obs = obs_exp.clamp_min(1e-8)
            gamma_dist = torch.distributions.Gamma(alpha_states, theta_states)
            gamma_lp = gamma_dist.log_prob(safe_obs)

            log_zero = torch.log(zero_states.clamp_min(1e-12))
            log_one_minus_zero = torch.log((1 - zero_states).clamp_min(1e-12))

            lp = torch.where(
                is_zero,
                log_zero.expand_as(gamma_lp),
                log_one_minus_zero + gamma_lp,
            )

            total = total + lp if log_prob else total * lp.exp()
        elif etype == 'continuous':
            loc_c, scale_c = params
            dist = torch.distributions.Normal(loc_c[states].unsqueeze(0), scale_c[states].unsqueeze(0))
            lp = dist.log_prob(obs_c.unsqueeze(1))
            total = total + lp if log_prob else total * lp.exp()
        else:
            raise ValueError(f"unknown emission type: {etype}")
 
    return total


def hmm_sample_future(obs_seq, probs_initial, probs_transition, probs_emission, n_pred_steps, num_samples, markov_order, log_prob, probs_trans_ho = None, state_grid = None):
    
    alpha, state_grid = hmm_filtering(obs_seq, probs_initial, probs_transition, probs_emission, markov_order, log_prob, marginal_state_probs=False, probs_trans_ho=probs_trans_ho, state_grid=state_grid)
    
    if log_prob:
        current_belief = alpha[-1].exp()
    else:
        current_belief = alpha[-1]
    future_obs = []

    with pyro.plate("samples", num_samples, dim=-1):
        tuple_idx = pyro.sample(
            "x_tuple_0",
            dist.Categorical(current_belief)
        )
        state_tuple = state_grid[tuple_idx]

        for t in range(n_pred_steps):
            x_next = pyro.sample(
                f"x_future_{t}",
                dist.Categorical(probs_transition[tuple(state_tuple.T)])
            )

            y_next = pyro.sample(
                f"y_future_{t}",
                dist.Categorical(probs_emission[x_next])
            )

            future_obs.append(y_next)
            state_tuple = torch.cat([state_tuple[:, 1:], x_next.unsqueeze(-1)], dim=1)

    return torch.stack(future_obs, dim=1)

def hmm_sample_future_multivariate_batched(sequences, lengths, probs_initial, probs_transition, probs_emission, emission_types, n_pred_steps, num_samples, markov_order, log_prob, probs_trans_ho = None, state_grid = None, batch_idx = None):
    
    num_sequences, n_channels, T = sequences.shape
    alpha, state_grid = hmm_filtering_multivariate_batched(sequences, lengths, probs_initial, probs_transition, probs_emission, emission_types, markov_order, log_prob, marginal_state_probs=False, probs_trans_ho=probs_trans_ho, state_grid=state_grid, batch_idx=batch_idx)
    future_obs = torch.zeros((num_samples, num_sequences, n_channels, n_pred_steps))
    
    # we pick the last unpadded state of our sequences 
    if log_prob:
        current_beliefs = alpha[lengths-1, torch.arange(num_sequences)].exp()
    else:
        current_beliefs = alpha[lengths-1, torch.arange(num_sequences)]

    with pyro.plate("sequences", num_sequences, dim=-1) as seq_idx:
        with pyro.plate("samples", num_samples, dim=-2) as sample_seq_idx:
            tuple_idx = pyro.sample(
                "x_tuple_0",
                dist.Categorical(current_beliefs[seq_idx])
            )
            state_tuple = state_grid[tuple_idx]

            for t in tqdm(range(n_pred_steps), desc=f'Predicting n={n_pred_steps} steps into the future'):
                x_next = pyro.sample(
                    f"x_future_{t}",
                    dist.Categorical(probs_transition[tuple(state_tuple.permute(2, 0, 1))])
                )
                
                for d, emission_type in enumerate(emission_types):
                    if emission_type == 'discrete':
                        future_obs[:, :, d, t] = pyro.sample(
                            f"y_{d}_{t}",
                            dist.Categorical(probs_emission[d][x_next])
                            )
                    elif emission_type == 'gamma':
                        future_obs[:, :, d, t] = pyro.sample(
                            f"y_{d}_{t}",
                            dist.Gamma(probs_emission[d][0][x_next], probs_emission[d][1][x_next])
                            )
                    elif emission_type == 'zeroinflatedgamma':
                        future_zero_obs = pyro.sample(f"zero_{d}_{t}", dist.Bernoulli(probs_emission[d][2][x_next]))
                        
                        future_non_zero_obs = pyro.sample(
                            f"y_{d}_{t}",
                            dist.Gamma(probs_emission[d][0][x_next], probs_emission[d][1][x_next])
                            )
                        future_obs[:, :, d, t] = torch.where(
                            future_zero_obs.bool(),
                            torch.zeros_like(future_non_zero_obs),
                            future_non_zero_obs,
                        )
                        
                    elif emission_type == 'continuous':
                        future_obs[:, :, d, t] = pyro.sample(
                            f"y_{d}_{t}",
                            dist.Normal(probs_emission[d][0][x_next], probs_emission[d][1][x_next])
                            )

                state_tuple = torch.cat([state_tuple[:, :, 1:], x_next.unsqueeze(-1)], dim=2)

    return future_obs

def hmm_sample_future_multivariate(obs_seq, probs_initial, probs_transition, probs_emission, emission_types, n_pred_steps, num_samples, markov_order, log_prob, probs_trans_ho = None, state_grid = None):
    
    alpha, state_grid = hmm_filtering_multivariate(obs_seq, probs_initial, probs_transition, probs_emission, emission_types, markov_order, log_prob, marginal_state_probs=False, probs_trans_ho=probs_trans_ho, state_grid=state_grid)
    
    if log_prob:
        current_belief = alpha[-1].exp()
    else:
        current_belief = alpha[-1]
    future_obs = torch.zeros((obs_seq.shape[0], num_samples, n_pred_steps))

    with pyro.plate("samples", num_samples, dim=-1):
        tuple_idx = pyro.sample(
            "x_tuple_0",
            dist.Categorical(current_belief)
        )
        state_tuple = state_grid[tuple_idx]

        for t in range(n_pred_steps):
            x_next = pyro.sample(
                f"x_future_{t}",
                dist.Categorical(probs_transition[tuple(state_tuple.T)])
            )

            for d, emission_type in enumerate(emission_types):
                if emission_type == 'discrete':
                    future_obs[d, :, t] = pyro.sample(
                        f"y_{d}_{t}",
                        dist.Categorical(probs_emission[d][x_next])
                        )
                elif emission_type == 'gamma':
                    future_obs[d, :, t] = pyro.sample(
                        f"y_{d}_{t}",
                        dist.Gamma(probs_emission[d][0][x_next], probs_emission[d][1][x_next])
                        )
                elif emission_type == 'continuous':
                    future_obs[d, :, t] = pyro.sample(
                        f"y_{d}_{t}",
                        dist.Normal(probs_emission[d][0][x_next], probs_emission[d][1][x_next])
                        )

            state_tuple = torch.cat([state_tuple[:, 1:], x_next.unsqueeze(-1)], dim=1)

    return future_obs

def build_higher_order_transition_matrix(probs_transition, hidden_dim, markov_order, log_prob):
    """
    Builds augmented transition matrix A where:
    alpha_t = alpha_{t-1} @ A

    Returns:
        A: (K, K)
        state_grid: (K, markov_order)
    """
    K = hidden_dim ** markov_order

    # enumeration of state tuples
    state_grid = torch.cartesian_prod(*[torch.arange(hidden_dim) for _ in range(markov_order)])

    if markov_order == 1:
        state_grid.unsqueeze_(1)

    # indices for shifting tuples
    prev_prefix = state_grid[:, 1:]
    next_suffix = state_grid[:, :-1]


    # match transitions
    if log_prob:
        A = torch.full((K, K), -torch.inf)

        for i in range(K):
            log_trans_probs = probs_transition[tuple(state_grid[i])].log()
            matches = (prev_prefix[i] == next_suffix).all(dim=1)
            A[i, matches] = log_trans_probs[state_grid[matches, -1]]

    else:
        A = torch.zeros(K, K)

        for i in range(K):
            trans_probs = probs_transition[tuple(state_grid[i])]
            matches = (prev_prefix[i] == next_suffix).all(dim=1)
            A[i, matches] = trans_probs[state_grid[matches, -1]]

    return A, state_grid

def get_full_info_prefix_indices(lengths):
    
    full_info_prefix_indices = list()
    for i in range(1, lengths.size()[0]):
        if lengths[i] <= lengths[i-1]:
            full_info_prefix_indices.append(i-1)

    full_info_prefix_indices.append(i)
    
    return full_info_prefix_indices

def _get_batch_indices(n_indices, n_batches):
    
    batch_size = int(n_indices/n_batches) + 1
    all_indices = [_ for _ in range(n_indices)]
    batch_indices = [all_indices[n:n+batch_size] for n in range(0, n_indices, batch_size)]
    return batch_indices