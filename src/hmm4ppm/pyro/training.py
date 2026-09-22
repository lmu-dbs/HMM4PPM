from collections.abc import Callable

import git
repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

import os
import pickle
import io

import pandas as pd
import pyro
import torch
from torch.nn.functional import pad
from tqdm import tqdm

from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDelta, init_to_mean
from pyro import poutine
from sklearn.preprocessing import OrdinalEncoder
from ..encoding.util import StrictlyNonNegativeOrdinalEncoder, GammaScaler, ZeroInflatedGammaScaler

import matplotlib.pyplot as plt

from ..models.hmm import PyroHMM, MultivariateHMM
from ..data.sequencedata import SequenceData

from ..util.logging import init_logging
logger = init_logging(__name__, 'training.log')

from ..util.params import hash_param_dict, save_hash_dict

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class HMMTrainer():

    def __init__(self, model: PyroHMM, train_args: dict, encoding_params: dict, export_path: os.PathLike = None, load_only: bool = False):
        """
        Args:
            model (Callable): A pyro model. In this case a HMM implementation.
            model_args (dict): All model parameters that belong in the call of the model (and are not set with default values)
            train_args (dict):  train args describe main parameters for training. Necessary entries into the dictionary are
                                number of hidden states in the HMM ('hidden_dim'), learning rate ('learning_rate'), 
                                batch size for training ('batch_size') and number of inference steps ('num_steps').
                                GPU support is enabled with providing 'cuda' = True. Default for cuda is None/False.
            encoding_params (dict): encoding params describe encoders and respective data columns encoded with those encoders along the
                                    data preparation.
        """
        self.model = model
        self.train_args = train_args
        self.encoding_params = encoding_params

        if not load_only:
            if train_args.get("cuda"):
                self.train_args["cuda"] = train_args["cuda"]
            else:
                self.train_args["cuda"] = False

        self.guide = AutoDelta(poutine.block(self.model.model, expose_fn=lambda msg: msg["name"].startswith(("probs_", "loc_", "scale_", "alpha", "theta", "zero"))), init_loc_fn=init_to_mean)
        
        self.optim = Adam({"lr": train_args["learning_rate"]})
        self.elbo = TraceEnum_ELBO(max_plate_nesting=2,
                              strict_enumeration_warning=True
                              )
        self.losses = list()
        self.trained_params = dict()

        self._padding_size = 1
        
        if export_path is not None:
            self.param_hash, serialized = hash_param_dict({k: v for k, v in train_args.items() if k not in ['reuse_fitted', 'max_pred_length', 'num_samples', 'cuda']})
            save_hash_dict(self.param_hash, serialized, os.path.join(export_path, 'params.jsonl'))
            self.export_path = export_path
        
        self.param_slug = None
    
    def fit(self, seed: int = 0, load_only: bool = False):
        
        self.param_slug = f'{self.param_hash}'
        
        param_filepath = os.path.join(self.export_path, f"trained_params_{self.model.__name__.lower()}_{self.param_slug}.pkl")
        
        # performs the training of the HMM model
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        
        self.svi = SVI(poutine.scale(self.model.model, scale = 1/self.model_args_train['lengths'].sum()/self.model_args_train['sequences'].shape[1]), 
                       poutine.scale(self.guide, scale = 1/self.model_args_train['lengths'].sum()/self.model_args_train['sequences'].shape[1]), 
                       self.optim, 
                       self.elbo)

        # check if params were trained and saved before
        if os.path.exists(param_filepath) and self.train_args['reuse_fitted']:
            with open(param_filepath, 'rb') as f:
                self.trained_params = CPU_Unpickler(f).load()
                # self.trained_params = pickle.load(f)
                
            logger.info("Found and loaded trained params for config")
        
        else:
            if load_only:
                raise ValueError("specified load_only but no trained HMM was found for given parameter settings")
            
            pbar = tqdm(range(self.train_args["num_steps"]))
            pbar.set_description_str(f"Step {'':4} | loss={'':13} | min. loss={'':13} | max. loss={'':13}")

            steps_performed = 0
            interrupt_addendum = ""

            for step in pbar:
                try:
                    last_good_params = {
                        name: pyro.param(name).detach().clone()
                        for name in pyro.get_param_store().get_all_param_names()
                    }
                    loss = self.svi.step(**self.model_args_train)
                    self.losses.append(loss)
                    
                    # we save params in each step to be able to go back to last step without error/exception
                    self.trained_params = last_good_params
                    
                    pbar.set_description_str(f"Step {step:4d} | loss={loss:13.4f} | min. loss={min(self.losses):13.4f} | max. loss={max(self.losses):13.4f}")
                    steps_performed += 1
                except ValueError as e:
                    # e.args = (f"SVI training stopped early due to error",)
                    # raise
                    logger.warning(f"SVI training stopped early due to ValueError")
                    break
                except KeyboardInterrupt as e:
                    logger.warning(f"SVI training interupted by user - continuing with latest trained params after step {steps_performed}")
                    interrupt_addendum = f"_interrupted{steps_performed}"
                    break

            with open(os.path.join(self.export_path, f"trained_params_{self.model.__name__.lower()}_{self.param_hash}{interrupt_addendum}.pkl"), 'wb') as f:
                pickle.dump(self.trained_params, f, protocol=pickle.HIGHEST_PROTOCOL)

            plt.figure(figsize=(5, 2))
            plt.plot(self.losses)
            plt.xlabel("SVI step")
            plt.ylabel("ELBO loss")
            
            if self.export_path is not None:
                plt.savefig(os.path.join(self.export_path, f"train_loss_{self.model.__name__.lower()}_{self.param_hash}{interrupt_addendum}.png"))
            else:
                plt.savefig(f"train_loss_{self.model.__name__.lower()}_{self.param_hash}{interrupt_addendum}.png")
    
    def load_data(self, train: SequenceData, val: SequenceData, test: SequenceData):
        self.data_train = train
        self.data_val = val
        self.data_test = test

    def prepare_train(self, additional_cols_to_pad: list[str]):
        """This prepares the model_args for the respective model
        """

        # padding trace data by one START and one END token
        self.data_train.pad_columns(cols_to_pad=[self.data_train.activity_identifier] + additional_cols_to_pad, n_pad=1)

        self.data_train.data[self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].ffill()
        self.data_train.data[self.data_train.timestamp_identifier] = self.data_train.data.groupby(self.data_train.case_identifier)[self.data_train.timestamp_identifier].bfill()
        
        # generating time features
        # TSLE: time since last event
        # TSMN: time since midnight
        # TSCS: time since case start
        self.data_train.generate_time_features(["tsle", "tsmn", "tscs"])

        self.train_max_trace_len = self.data_train._get_max_trace_len()
        
        act_idx = self.data_train.data.groupby(self.data_train.case_identifier).apply(
            lambda x: pd.Series(range(-(self._padding_size-1), 
                                      len(x)-(self._padding_size-1))))
        
        self.data_train.data['activity_idx'] = act_idx.reset_index(drop=True)
        
        # setup encoding and transformers
        self.data_train.setup_encoders(self.encoding_params)
        # self.data_train.setup_transformers(self.transform_params)

        # transform
        # self.data_train.transform()
        
        # encode activities
        self.data_train.encode_activities()

        # encode/scale remaining attribute data
        self.data_train.encode_attributes()


        self.prepare_model_args_train()
        
        logger.info('Training data prepared!')
    
    def prepare_test(self, act_encoder: OrdinalEncoder, additional_cols_to_pad: list[str], val_set: bool = False, attributes: list[str] = None, filter_sequences: bool = True):
        logger.info('Preparing test data...')
        
        if val_set:
            data = self.data_val
        else:
            data = self.data_test

        # padding trace data by one START and one END token
        data.pad_columns(cols_to_pad=[data.activity_identifier] + additional_cols_to_pad, n_pad=1)

        data.data[data.timestamp_identifier] = data.data.groupby(data.case_identifier)[data.timestamp_identifier].ffill()
        data.data[data.timestamp_identifier] = data.data.groupby(data.case_identifier)[data.timestamp_identifier].bfill()
        
        # generating time features
        # TSLE: time since last event
        # TSMN: time since midnight
        # TSCS: time since case start
        data.generate_time_features(["tsle", "tsmn", "tscs"])

        if val_set:
            self.val_max_trace_len = data._get_max_trace_len()
        else:
            self.test_max_trace_len = data._get_max_trace_len()
            
        act_idx = data.data.groupby(data.case_identifier).apply(
            lambda x: pd.Series(range(-(self._padding_size-1), 
                                      len(x)-(self._padding_size-1))))
        
        data.data['activity_idx'] = act_idx.reset_index(drop=True)
        
        # encode activities
        data.encode_activities(act_encoder=act_encoder)

        # encode/scale remaining attribute data
        data.encoders = self.data_train.encoders
        data.encode_attributes(already_fitted=True)

        all_activities = data.data[[data.case_identifier, data.activity_identifier]]
        next_activities = list()
        activity_suffixes = list()

        for name, grouped in all_activities.groupby(data.case_identifier):
            # extract next activities
            case_next_activities = grouped[data.activity_identifier][2:].tolist()
            next_activities.extend(case_next_activities)

            # extract activity suffixes
            case_activity_suffixes = list()
            for prefix_idx in range(2, len(grouped)):
                p_suffix = grouped[data.activity_identifier][prefix_idx:].tolist()
                case_activity_suffixes.append(p_suffix)
            
            activity_suffixes.extend(case_activity_suffixes)

        data.next_activities = next_activities
        data.activity_suffixes = activity_suffixes

        self.prepare_model_args_test(val_set=val_set)

        logger.info(f"{'Test' if not val_set else 'Validation'} data prepared!")

    def prepare_model_args_train(self):

        trace_cols = [self.data_train.activity_identifier] + self.data_train.attribute_identifiers
        self.data_train.extract_traces(columns=trace_cols)
        self.start_activity = self.data_train.start_activity
        self.end_activity = self.data_train.end_activity
        
        model_args = dict()

        if self.model in [MultivariateHMM]:

            # transform individual traces into padded tensor
            tensor_sequences = list()
            lengths = list()
            for trace in self.data_train.traces:
                tensor_sequence = torch.tensor([trace[key] for key in trace_cols])
                tensor_sequences.append(tensor_sequence)
                lengths.append(tensor_sequence.shape[1])
            
            lengths = torch.tensor(lengths)
            max_tensor_length = lengths.max()
            pad_len = int(self.train_args["pad_buffer"] * max_tensor_length) + 1

            # pad all tensor_sequences to pad_len
            logger.warning("WE ARE PADDING WITH END ACTIVITY FOR EVERYTHING!")
            for idx, seq in enumerate(tensor_sequences):
                # seq = pad(seq, (0, pad_len - seq.shape[1]), value=self.data_train.end_activity)
                seq = pad(seq, (0, pad_len - seq.shape[1]), mode='replicate')
                tensor_sequences[idx] = seq
            
            sequences = torch.stack(tensor_sequences, dim=0)
            
            if isinstance(self.data_train.act_encoder, StrictlyNonNegativeOrdinalEncoder):
                n_categories_per_variable = [len(self.data_train.act_encoder.categories_[0]) + 2]
            elif isinstance(self.data_train.act_encoder, OrdinalEncoder):
                n_categories_per_variable = [len(self.data_train.act_encoder.categories_[0])]
            encoder_signals = {e_val[0]:e_val[1] for e_val in self.data_train.encoders.values()}
            for signal_name in self.data_train.traces[0].keys():
                if signal_name not in [self.data_train.case_identifier, self.data_train.activity_identifier]:
                    
                    signal_encoder, internal_idx = [(e, signals.index(signal_name)) for e, signals in encoder_signals.items() if signal_name in signals][0]
                    
                    if isinstance(signal_encoder, StrictlyNonNegativeOrdinalEncoder) and 'categories_' in signal_encoder.__dict__.keys():
                        n_categories_per_variable.append(len(signal_encoder.categories_[internal_idx]) + 2)
                    elif isinstance(signal_encoder, OrdinalEncoder) and 'categories_' in signal_encoder.__dict__.keys():
                        n_categories_per_variable.append(len(signal_encoder.categories_[internal_idx]))
                    elif isinstance(signal_encoder, GammaScaler):
                        n_categories_per_variable.append(-2)
                    elif isinstance(signal_encoder, ZeroInflatedGammaScaler):
                        n_categories_per_variable.append(-3)
                    else:
                        n_categories_per_variable.append(-1)
            
            variable_n_cat = {s: n for s, n in zip([n for n in self.data_train.traces[0].keys() if n!=self.data_train.case_identifier], n_categories_per_variable)}
                    
            emission_types = ['discrete' if n > 0 else 'continuous' if n == -1 else 'gamma' if n == -2 else 'zeroinflatedgamma' for n in n_categories_per_variable]
            
            hidden_dim = self.train_args["hidden_dim"]
            markov_order = self.train_args["markov_order"]
            pred_length = None
            max_pred_length = None
            batch_size = self.train_args["hmm_batch_size"]
            include_prior = True
            mask_padded = self.train_args["mask_padded"]

            model_args["sequences"] = sequences
            model_args["lengths"] = lengths
            model_args["n_categories_per_variable"] = n_categories_per_variable
            model_args["emission_types"] = emission_types
            model_args["hidden_dim"] = hidden_dim
            model_args["markov_order"] = markov_order
            model_args["batch_size"] = batch_size
            model_args["include_prior"] = include_prior
            model_args["mask_padded"] = mask_padded

        else:
            raise NotImplementedError(f"Training loop preparation for model architecture {self.model.__name__} is not implemented yet!")

        self.model_args_train = model_args
    
    def prepare_model_args_test(self, val_set: bool = False):
        
        if val_set:
            data = self.data_val
        else:
            data = self.data_test

        trace_cols = [data.activity_identifier] + data.attribute_identifiers
        data.extract_traces(columns=trace_cols)

        model_args = dict()

        if self.model in [MultivariateHMM]:

            # transform individual traces into padded tensor
            tensor_sequences = list()
            lengths = list()
            for trace in data.traces:
                tensor_sequence = torch.tensor([trace[key] for key in trace_cols])
                tensor_sequences.append(tensor_sequence)
                lengths.append(tensor_sequence.shape[1])
            
            lengths = torch.tensor(lengths)
            max_tensor_length = lengths.max()
            pad_len = int(self.train_args["pad_buffer"] * max_tensor_length) + 1
            
            # pad all tensor_sequences to max_tensor_length
            for idx, seq in enumerate(tensor_sequences):
                seq = pad(seq, (0, max_tensor_length - seq.shape[1]))
                tensor_sequences[idx] = seq
            
            sequences = torch.stack(tensor_sequences, dim=0)

            # with 2, l we get prefixes from START, first activity onwards until the next to last activity in the sequence
            # we need to adjust repeat_interleave as well
            repeated_sequences = torch.repeat_interleave(sequences, lengths - 2, dim=0)
            prefix_lengths = torch.cat([torch.arange(2, l) for l in lengths])
            
            if isinstance(self.data_train.act_encoder, StrictlyNonNegativeOrdinalEncoder):
                n_categories_per_variable = [len(self.data_train.act_encoder.categories_[0]) + 2]
            elif isinstance(self.data_train.act_encoder, OrdinalEncoder):
                n_categories_per_variable = [len(self.data_train.act_encoder.categories_[0])]
            # n_categories_per_variable = [len(self.data_train.act_encoder.categories_[0])]
            encoder_signals = {e_val[0]:e_val[1] for e_val in self.data_train.encoders.values()}
            for signal_name in self.data_train.traces[0].keys():
                if signal_name not in [self.data_train.case_identifier, self.data_train.activity_identifier]:
                    
                    signal_encoder, internal_idx = [(e, signals.index(signal_name)) for e, signals in encoder_signals.items() if signal_name in signals][0]
                    
                    if isinstance(signal_encoder, StrictlyNonNegativeOrdinalEncoder) and 'categories_' in signal_encoder.__dict__.keys():
                        n_categories_per_variable.append(len(signal_encoder.categories_[internal_idx]) + 2)
                    elif isinstance(signal_encoder, OrdinalEncoder) and 'categories_' in signal_encoder.__dict__.keys():
                        n_categories_per_variable.append(len(signal_encoder.categories_[internal_idx]))
                    elif isinstance(signal_encoder, GammaScaler):
                        n_categories_per_variable.append(-2)
                    elif isinstance(signal_encoder, GammaScaler):
                        n_categories_per_variable.append(-3)
                    else:
                        n_categories_per_variable.append(-1)
            
            emission_types = ['discrete' if n > 0 else 'continuous' if n == -1 else 'gamma' if n == -2 else 'zeroinflatedgamma' for n in n_categories_per_variable]
            
            hidden_dim = self.train_args["hidden_dim"]
            markov_order = self.train_args["markov_order"]
            max_pred_length = self.train_args["max_pred_length"]
            num_samples = self.train_args["num_samples"]
            log_prob = self.train_args["log_prob"]
            batch_size = None
            include_prior = False

            model_args["sequences"] = repeated_sequences
            model_args["lengths"] = prefix_lengths
            model_args["n_categories_per_variable"] = n_categories_per_variable
            # model_args["emission_types"] = emission_types
            model_args["hidden_dim"] = hidden_dim
            model_args["markov_order"] = markov_order
            # model_args["pred_length"] = pred_length
            model_args["max_pred_length"] = max_pred_length
            model_args["batch_size"] = batch_size
            model_args["include_prior"] = include_prior
            model_args["num_samples"] = num_samples
            model_args["log_prob"] = log_prob

        else:
            raise NotImplementedError(f"Testing loop preparation for model architecture {self.model.__name__} is not implemented yet!")

        if val_set:
            self.model_args_val = model_args
        else:
            self.model_args_test = model_args