import time
import os
import mlflow
import numpy as np
import torch
import random
from itertools import chain
import copy
import json
from typing import Literal

from scipy.stats import wasserstein_distance

from hmm4ppm.data.sequencedata import SequenceData
from hmm4ppm.util.config_utils import read_config
from hmm4ppm.pyro.posterior_distributions import get_posterior_distribution
from hmm4ppm.pyro.prediction import get_full_info_prefix_indices

from hmm4ppm.pyro.training import HMMTrainer
from hmm4ppm.pyro.prediction import HMMPredictor
from hmm4ppm.models.hmm import HMMFactory

from hmm4ppm.util.logging import init_logging
logger = init_logging(__name__, 'main.log')

from util.paths import CONFIG_PATH, DATA_PATH, EXPORT_PATH
from util.combinations import param_combinations

from hmm4ppm.util.params import load_all

FIG_EXP_DIR = os.path.join(EXPORT_PATH, 'plots')
PARAMS_EXP_DIR = os.path.join(EXPORT_PATH, 'params')
DATASET_EXP_DIR = os.path.join(EXPORT_PATH, 'datasets')

# criterion = 'loglik'
criterion = 'wasserstein'

def main():

    general_config = read_config(os.path.join(CONFIG_PATH, "general_config.yml"))
    data_configs = read_config(os.path.join(CONFIG_PATH, "data_configs.yml"))
    model_configs = read_config(os.path.join(CONFIG_PATH, "model_configs.yml"))
    mlflow_config = read_config(os.path.join(CONFIG_PATH, "mlflow_config.yml"))

    os.makedirs(FIG_EXP_DIR, exist_ok=True)
    os.makedirs(PARAMS_EXP_DIR, exist_ok=True)
    os.makedirs(DATASET_EXP_DIR, exist_ok=True)

    # mlflow.set_tracking_uri(mlflow_config["uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    for dataset in general_config["dataset"]:
        
        os.makedirs(os.path.join(PARAMS_EXP_DIR, dataset), exist_ok=True)
        os.makedirs(os.path.join(DATASET_EXP_DIR, dataset), exist_ok=True)
        
        try:
            data_config = data_configs[dataset]
        except KeyError as e:
            e.args = (f"desired datset {dataset} not found in data_config.yml",)
            raise

        phase_model_config = model_configs[general_config['model_config']]['phase_config']
        combinations_generator = param_combinations(phase_model_config['model_params'])
        additional_params = dict()
        
        additional_params['seed'] = general_config['seed']
        additional_params['dataset'] = dataset
        additional_params['resource_col'] = data_config['resource_identifier']

        markov_orders = [1, 2, 3]

        random.seed(additional_params['seed'])
        np.random.seed(additional_params['seed'])

        times = dict()
        times["start_time"] = time.perf_counter()
        times["data_prep_time"] = time.perf_counter()

        if not data_config.get('read_params'):
            data_config['read_params'] = dict()

        data = SequenceData.from_csv(
            load_path=os.path.join(DATA_PATH, data_config["file_name"]),
            case_identifier=data_config["case_identifier"],
            activity_identifier=data_config["activity_identifier"],
            resource_identifier=data_config["resource_identifier"],
            timestamp_identifier=data_config["timestamp_identifier"],
            read_params=data_config.get('read_params'),
            encoding_params=data_config.get('encoding_params'),
        )

        if general_config['cv_folds'] > 1:
            
            for markov_order in markov_orders:
            
                # sample with same seeds for each markov_order to prevent leakage
                random.seed(additional_params['seed'])
                np.random.seed(additional_params['seed'])

                folds, outer_train, outer_test = data.train_test_split(train_pct=general_config.get('train_pct'), val_pct=general_config.get('val_pct'), cv=general_config.get('cv_folds'))
                times['data_prep_time'] = time.perf_counter()

                # model_params = dict(zip(list(model_config.keys()), [param for param in combination]))
                if phase_model_config is None:
                    raise KeyError('desired model config not found in model_config.yml')

                fold_models = list()

                for fold_idx, fold in enumerate(folds):

                    data_train, data_test = fold
                    times['run_start_time'] = time.perf_counter()
                    
                    best_hmm, best_params = select_best_hmm_per_order(dataset=dataset, markov_order=markov_order, data_train=data_train, data_val=data_val, data_test=data_test, criterion=criterion, fold_idx=fold_idx)

        else:                
            
            for markov_order in markov_orders:
                
                # sample with same seeds for each markov_order to prevent leakage
                random.seed(additional_params['seed'])
                np.random.seed(additional_params['seed'])
                
                data_train, data_val, data_test = data.train_test_split(train_pct=general_config.get('train_pct'), val_pct=general_config.get('val_pct'), cv=general_config.get('cv_folds'))
                times['data_prep_time'] = time.perf_counter()
                
                if phase_model_config is None:
                    raise KeyError('desired model config not found in model_config.yml')
                
                times['run_start_time'] = time.perf_counter()
    
                best_hmm, best_params = select_best_hmm_per_order(dataset=dataset, markov_order=markov_order, data_train=data_train, data_val=data_val, data_test=data_test, criterion=criterion)

def select_best_hmm_per_order(dataset: str, markov_order: int, data_train: SequenceData, data_val: SequenceData, data_test: SequenceData, criterion: Literal['loglik', 'wasserstein'], fold_idx: int|None = None):
    
    if fold_idx is not None:
        param_dict_path = os.path.join(PARAMS_EXP_DIR, dataset, f"fold_{fold_idx}", 'params.jsonl')
    else:
        param_dict_path = os.path.join(PARAMS_EXP_DIR, dataset, 'params.jsonl')
        
    best_params_path = os.path.join(PARAMS_EXP_DIR, dataset, f'best_params_{criterion}.jsonl')
    
    try:
        best_params_slug_dict = {}
        with open(best_params_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    dict_line = json.loads(line)
                    if markov_order==[int(_) for _ in dict_line.keys()][0]: # only load params for specified markov order
                        best_params_slug_dict.update(json.loads(line))
        if len(best_params_slug_dict) == 0:
            best_param_slug = None
            best_params = None
        else:
            best_param_slug = [_ for _ in best_params_slug_dict[str(markov_order)].keys()][0]
            best_params = best_params_slug_dict[str(markov_order)][best_param_slug]
    except FileNotFoundError:
        best_param_slug = None
        best_params = None
        
    if best_params is not None and best_param_slug is not None:
        best_hmm = load_trained_hmm(best_params, best_param_slug, dataset, data_train, data_val, data_test)
    else:
    
        param_dict = {}
        with open(param_dict_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    dict_line = json.loads(line)
                    if markov_order==[_ for _ in dict_line.values()][0]['markov_order']: # only load params for specified markov order
                        param_dict.update(json.loads(line))
            
        # load the HMM params from the exported trained_params
        # hmms = list()
        slugs = list()
        criterion_values = list()
        
        # sort params by hidden_dim (ascending) - we want to pick highest loglik with lowest amount of states
        param_dict = {k: v for k, v in sorted(param_dict.items(), key=lambda x: x[1]['hidden_dim']) if v['hidden_dim'] >= 3}
        
        for param_idx, (param_slug, params) in enumerate(param_dict.items()):
        
            loaded_hmm = load_trained_hmm(params, param_slug, dataset, copy.deepcopy(data_train), copy.deepcopy(data_val), copy.deepcopy(data_test))
            
            slugs.append(param_slug)
        
            # calculate model ranking criterion
            
            if criterion=='loglik':
            
                logger.info(f"Calculating log-likelihood for HMM {param_idx + 1} of {len(param_dict)}...")
                
                loglik = calc_loglik(loaded_hmm)
                
                logger.info(f"Log-likelihood calculated: {loglik}")
                criterion_values.append(loglik)
            elif criterion=='wasserstein':
                
                logger.info(f"Calculating alpha probabilities (val) for HMM {param_idx + 1} of {len(param_dict)}...")
                            
                if loaded_hmm is None:
                    criterion_values.append(-float("inf"))
                else:
                    loaded_hmm_predictor = HMMPredictor(loaded_hmm, pred_args={})
                    
                    loaded_hmm_predictor.filter_val(n_batches=5)
                    
                    full_sequence_indices_val = get_full_info_prefix_indices(loaded_hmm_predictor.model.model_args_val['lengths'])
                    
                    if loaded_hmm.train_args['log_prob']:
                        alpha_probs_val = [loaded_hmm_predictor.filtered_sequences_val[i].exp() for i in full_sequence_indices_val]
                        
                    else:
                        alpha_probs_val = [loaded_hmm_predictor.filtered_sequences_val[i] for i in full_sequence_indices_val]
                    
                    logger.info(f"Calculating wasserstein distance for HMM {param_idx + 1} of {len(param_dict)}...")
                    
                    distance = calc_neg_posterior_error(loaded_hmm, alpha_probs_val=alpha_probs_val)
                    
                    logger.info(f"Wasserstein distance calculated: {distance}")
                    criterion_values.append(distance)
        
        # select best model for ranking criterion for each order    
        max_criterion = max(criterion_values)
        max_idx = criterion_values.index(max_criterion)
        best_param_slug = slugs[max_idx]
        best_params = param_dict[best_param_slug]
        
        best_hmm = load_trained_hmm(best_params, best_param_slug, dataset, data_train, data_val, data_test)
        
        # save best params to file
        last_file_state = load_all(best_params_path)

        if str(markov_order) not in last_file_state.keys():
            slug_params_dict = {best_param_slug: best_params}
            serialized = json.dumps(slug_params_dict, sort_keys=True, default=str).encode('utf-8') 
            with open(best_params_path, 'a') as f:
                f.write(json.dumps({markov_order: json.loads(serialized)}) + '\n')
    
    return best_hmm, best_params

def calc_loglik(hmm):
    # we can drop number of parameters if we keep training the same and do not compare across different markov orders
    # we only check all HMMs of the same markov order
    
    if hmm is None:
        return -float("inf")
    
    eval_args = hmm.model_args_val.copy()
    eval_args["include_prior"] = False
    eval_args["batch_size"] = None
    eval_args.pop('max_pred_length')
    eval_args.pop('num_samples')
    eval_args.pop('log_prob')
    eval_args['emission_types'] = hmm.model_args_train['emission_types']
    eval_args['mask_padded'] = hmm.model_args_train['mask_padded']
    
    total_loglik = 0.0
    with torch.no_grad():
        for batch_sequences, batch_lengths in chunk_sequences(eval_args['sequences'], eval_args['lengths'], chunk_size=int(eval_args['sequences'].shape[0]/10)):
            batch_args = dict(eval_args)
            batch_args["sequences"] = batch_sequences
            batch_args["lengths"] = batch_lengths
            total_loglik += -hmm.elbo.loss(hmm.model.model, hmm.guide, **batch_args)
    
    return total_loglik

def calc_neg_posterior_error(hmm, alpha_probs_val):
    
    if hmm is None:
        return -float("inf")
    
    posterior_errors = list()
    signals = [hmm.data_train.activity_identifier, hmm.data_train.resource_identifier, 'tsle', 'tscs'] # TODO where do we get these from?
    emission_types = ['discrete', 'discrete', 'zeroinflatedgamma', 'zeroinflatedgamma'] # TODO where do we get these from?
    
    for channel, emission_type in zip(signals, emission_types):
        
        val_posterior_dist, true_dist = get_posterior_distribution(hmm, channel, emission_type, mode='val', alpha_probs=alpha_probs_val)
        posterior_errors.append(wasserstein_distance(true_dist, val_posterior_dist))

    posterior_error = sum(posterior_errors)

    neg_posterior_error = -posterior_error

    return neg_posterior_error

def load_trained_hmm(model_params: dict, param_slug: str, dataset: str, data_train: SequenceData, data_val: SequenceData, data_test: SequenceData) -> HMMTrainer|None:
    
    model = HMMFactory.create(model_params['model_name'])
    
    model_params['max_pred_length'] = -1
    model_params['num_samples'] = -1
    model_params['reuse_fitted'] = True
    
    processhhmm = HMMTrainer(model=model, train_args=model_params, encoding_params=model_params["encoding_params"], export_path=os.path.join(PARAMS_EXP_DIR, dataset), load_only=True)
    
    processhhmm.load_data(data_train, data_val, data_test)
    
    pad_cols = list(set(chain(*model_params["encoding_params"].values())).difference(set(['tsle', 'tsmn', 'tscs', 'activity_idx'])))
    processhhmm.prepare_train(additional_cols_to_pad=pad_cols)
    processhhmm.prepare_test(act_encoder=data_train.act_encoder, additional_cols_to_pad=pad_cols, val_set=True)
    processhhmm.prepare_test(act_encoder=data_train.act_encoder, additional_cols_to_pad=pad_cols)
    
    logger.info("Loading HMM")
    
    processhhmm.param_hash = param_slug
    
    try:
        processhhmm.fit(load_only=True)
    except ValueError:
        logger.warning("HMM could not be loaded - no trained HMM found for given parameters - skipping")
        return None
    
    logger.info("HMM loaded")

    return processhhmm

def chunk_sequences(sequences, lengths, chunk_size):
    num_sequences = sequences.shape[0]
    for start in range(0, num_sequences, chunk_size):
        end = min(start + chunk_size, num_sequences)
        yield sequences[start:end], lengths[start:end]

if __name__=='__main__':
    main()