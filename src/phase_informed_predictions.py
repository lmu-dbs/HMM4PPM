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
from hmm4ppm.eval.evaluator import Evaluator

from hmm4ppm.models.ppm import PPMFactory, PPMModel

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
        ppm_model_config = model_configs[general_config['model_config']]['ppm_config']
        combinations_generator = param_combinations(phase_model_config['model_params'])
        config_combinations = [c_comb for c_comb in combinations_generator]
        additional_params = dict()
        
        cv_hashes = [random.getrandbits(128) for _ in range(0, len(config_combinations))]
        additional_params['seed'] = general_config['seed']
        additional_params['dataset'] = dataset
        additional_params['resource_col'] = data_config['resource_identifier']

        if model_configs[general_config['model_config']]['ppm_config']['model_type'] == ['BESTVanillaPhases']:
            markov_orders = [1]
        else:
            markov_orders = [1, 2, 3]
        
        eval_args = {'num_samples': phase_model_config['model_params']['num_samples'][0]}

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
                    fold_models.append((best_hmm, best_params))
                        
                for fold_idx, (best_hmm, best_params) in enumerate(fold_models):

                    best_hmm.model_args_test.update(eval_args)
                    
                    processhhmm, hmm_predictor, annotated_dataset = perform_loaded_run_train(best_hmm, times, export_path=os.path.join(PARAMS_EXP_DIR, dataset))
                    
                    intermediate_calc_times = {'hmm_prep_duration': times['hmm_prep_time_end'] - times['hmm_prep_time_start'], 
                                            'hmm_train_duration': times['fitting_time'] - times['hmm_prep_time_end'], 
                                            'hmm_filtering_duration': times['filtering_time_end'] - times['filtering_time_start'], 
                                            'hmm_annotation_duration': times['annotation_time_end'] - times['annotation_time_start'], 
                                            }
                    
                    ppm_model_config['model_params'].update({'model_type': ppm_model_config['model_type']})
                    
                    ppm_combinations_generator = param_combinations(ppm_model_config['model_params'])
                    ppm_config_combinations = [c_comb for c_comb in ppm_combinations_generator]
                    
                    for comb_idx, ppm_model_params in enumerate(ppm_config_combinations):
                        with mlflow.start_run():
                    
                            mlflow.log_metrics(intermediate_calc_times)
                    
                            # log model params
                            mlflow.log_params(best_params)
                            mlflow.log_params(additional_params)
                            mlflow.log_params(ppm_model_params)
                    
                            if ppm_model_params['model_type'] == 'BESTVanillaPhases':
                                include_phases_list = [False]
                            else:
                                include_phases_list = [True, False]
                    
                            for include_phases in include_phases_list:
                    
                                ppm_model_params['include_phases'] = include_phases
                    
                                data_train_, data_val_, data_test_ = copy.deepcopy(data_train), copy.deepcopy(data_val), copy.deepcopy(data_test)
                    
                                ppm_model = perform_run_train_ppm(ppm_model_params, general_config, data_train_, data_val_, data_test_, annotated_dataset, processhhmm, hmm_predictor, times, additional_params=additional_params)
                    
                                mlflow.log_metric(f"ppm_fitting_duration{'_phases' if include_phases else ''}", times['ppm_fitting_end_time'] - times['ppm_fitting_start_time'])
                    
                                if ppm_model_params['model_type'] == 'LSTM':
                                    mlflow.log_metric(f"epochs_trained{'_phases' if include_phases else ''}", ppm_model.trainer.epochs_trained)
                    
                                run_log_params_metrics = {'params': dict(),
                                                        'metrics': dict()}
                                if include_phases:
                                    model_params_eval_list = [
                                        {'perfect_activity_info': False,
                                        'use_hmm_phase_filtering': False,
                                        'include_phases': include_phases,
                                        'perfect_phase_info': False,
                                        },
                                        ]
                    
                                    if ppm_model_params['model_type'] == 'LSTM':
                                        model_params_eval_list.append({'perfect_activity_info': False, 
                                                                        'use_hmm_phase_filtering': True, 
                                                                        'include_phases': include_phases, 
                                                                        'perfect_phase_info': False,
                                                                        })
                                else:
                                    model_params_eval_list = [
                                        {'perfect_activity_info': False,
                                        'use_hmm_phase_filtering': False,
                                        'include_phases': include_phases,
                                        'perfect_phase_info': False,
                                        },
                                        ]
                    
                    
                    
                                for model_params_eval in model_params_eval_list:
                    
                                    remaining_model_params = {k: v for k, v in ppm_model_params.items() if k not in ['perfect_activity_info',
                                                                                                                    'use_hmm_phase_filtering',
                                                                                                                    'include_phases',
                                                                                                                    'perfect_phase_info']}
                    
                                    model_params_eval.update(remaining_model_params)
                    
                                    logger.info(f"Results for dataset {general_config.get('dataset')} (include phases: {include_phases}, hmm phase filtering: {model_params_eval['use_hmm_phase_filtering']}, perfect phase info: {model_params_eval['perfect_phase_info']}, perfect activity info: {model_params_eval['perfect_activity_info']}")
                                    perform_run_test_ppm(ppm_model, model_params_eval, general_config, times, run_log_params_metrics)
                    
                                    # logging final times
                                    final_calc_times = {f"ppm_prediction_duration{'_phases' if include_phases else ''}{'_hmm' if model_params_eval['use_hmm_phase_filtering'] else ''}": times['prediction_ppm_end_time'] - times['prediction_ppm_start_time'], 
                                                        f"ppm_evaluation_duration{'_phases' if include_phases else ''}{'_hmm' if model_params_eval['use_hmm_phase_filtering'] else ''}": times['ppm_evaluation_end_time'] - times['ppm_evaluation_start_time'], 
                                                        }
                    
                                    mlflow.log_metrics(final_calc_times)
                    
                                    # logging params and metrics
                                    recoded_params = {f"{k}{'_phases' if include_phases else ''}{'_hmm' if model_params_eval['use_hmm_phase_filtering'] else ''}": v for k, v in run_log_params_metrics['params'].items()}
                                    recoded_metrics = {f"{k}{'_phases' if include_phases else ''}{'_hmm' if model_params_eval['use_hmm_phase_filtering'] else ''}": v for k, v in run_log_params_metrics['metrics'].items()}
                                    mlflow.log_params(recoded_params)
                                    mlflow.log_metrics(recoded_metrics)

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
                
                best_hmm.model_args_test.update(eval_args)

                processhhmm, hmm_predictor, annotated_dataset = perform_loaded_run_train(best_hmm, times, export_path=os.path.join(PARAMS_EXP_DIR, dataset))

                intermediate_calc_times = {'hmm_filtering_duration': times['filtering_time_end'] - times['filtering_time_start'], 
                                           'hmm_annotation_duration': times['annotation_time_end'] - times['annotation_time_start'], 
                                           }

                ppm_model_config['model_params'].update({'model_type': ppm_model_config['model_type']})
                
                ppm_combinations_generator = param_combinations(ppm_model_config['model_params'])
                ppm_config_combinations = [c_comb for c_comb in ppm_combinations_generator]
                
                for comb_idx, ppm_model_params in enumerate(ppm_config_combinations):
                    with mlflow.start_run():
                        
                        mlflow.log_metrics(intermediate_calc_times)
                        
                        # log model params
                        mlflow.log_params(best_params)
                        mlflow.log_params(additional_params)
                        mlflow.log_params(ppm_model_params)
                        
                        if ppm_model_params['model_type'] == 'BESTVanillaPhases':
                            include_phases_list = [False]
                        else:
                            include_phases_list = [True, False]
                        
                        for include_phases in include_phases_list:
                            
                            ppm_model_params['include_phases'] = include_phases
                            
                            data_train_, data_val_, data_test_ = copy.deepcopy(data_train), copy.deepcopy(data_val), copy.deepcopy(data_test)
                        
                            ppm_model = perform_run_train_ppm(ppm_model_params, general_config, data_train_, data_val_, data_test_, annotated_dataset, processhhmm, hmm_predictor, times, additional_params=additional_params)
                            
                            mlflow.log_metric(f"ppm_fitting_duration{'_phases' if include_phases else ''}", times['ppm_fitting_end_time'] - times['ppm_fitting_start_time'])
                            
                            if ppm_model_params['model_type'] == 'LSTM':
                                mlflow.log_metric(f"epochs_trained{'_phases' if include_phases else ''}", ppm_model.trainer.epochs_trained)
                            
                            run_log_params_metrics = {'params': dict(),
                                                    'metrics': dict()}
                            if include_phases:
                                model_params_eval_list = [
                                    {'perfect_activity_info': False,
                                    'use_hmm_phase_filtering': False,
                                    'include_phases': include_phases,
                                    'perfect_phase_info': False,
                                    },
                                    ]
                                
                                if ppm_model_params['model_type'] == 'LSTM':
                                    model_params_eval_list.append({'perfect_activity_info': False, 
                                                                    'use_hmm_phase_filtering': True, 
                                                                    'include_phases': include_phases, 
                                                                    'perfect_phase_info': False,
                                                                    })
                            else:
                                model_params_eval_list = [
                                    {'perfect_activity_info': False,
                                    'use_hmm_phase_filtering': False,
                                    'include_phases': include_phases,
                                    'perfect_phase_info': False,
                                    },
                                    ]
                                
                            
                            
                            for model_params_eval in model_params_eval_list:
                                
                                remaining_model_params = {k: v for k, v in ppm_model_params.items() if k not in ['perfect_activity_info',
                                                                                                                'use_hmm_phase_filtering',
                                                                                                                'include_phases',
                                                                                                                'perfect_phase_info']}
                                
                                model_params_eval.update(remaining_model_params)
                                
                                logger.info(f"Results for dataset {general_config.get('dataset')} (include phases: {include_phases}, hmm phase filtering: {model_params_eval['use_hmm_phase_filtering']}, perfect phase info: {model_params_eval['perfect_phase_info']}, perfect activity info: {model_params_eval['perfect_activity_info']}")
                                perform_run_test_ppm(ppm_model, model_params_eval, general_config, times, run_log_params_metrics)
                                
                                # logging final times
                                final_calc_times = {f"ppm_prediction_duration{'_phases' if include_phases else ''}{'_hmm' if model_params_eval['use_hmm_phase_filtering'] else ''}": times['prediction_ppm_end_time'] - times['prediction_ppm_start_time'], 
                                                    f"ppm_evaluation_duration{'_phases' if include_phases else ''}{'_hmm' if model_params_eval['use_hmm_phase_filtering'] else ''}": times['ppm_evaluation_end_time'] - times['ppm_evaluation_start_time'], 
                                                    }
                                
                                mlflow.log_metrics(final_calc_times)
                                
                                # logging params and metrics
                                recoded_params = {f"{k}{'_phases' if include_phases else ''}{'_hmm' if model_params_eval['use_hmm_phase_filtering'] else ''}": v for k, v in run_log_params_metrics['params'].items()}
                                recoded_metrics = {f"{k}{'_phases' if include_phases else ''}{'_hmm' if model_params_eval['use_hmm_phase_filtering'] else ''}": v for k, v in run_log_params_metrics['metrics'].items()}
                                mlflow.log_params(recoded_params)
                                mlflow.log_metrics(recoded_metrics)
                        
                logger.info("All done!")

def perform_run_train(data_train, data_val, data_test, model_params_train, times, export_path = None) -> HMMTrainer:
    
    model = HMMFactory.create(model_params_train['model_name'])
    
    processhhmm = HMMTrainer(model=model, train_args=model_params_train, encoding_params=model_params_train["encoding_params"], export_path=export_path)
    
    times['hmm_prep_time_start'] = time.perf_counter()
    
    processhhmm.load_data(data_train, data_val, data_test)
    
    pad_cols = list(set(chain(*model_params_train["encoding_params"].values())).difference(set(['tsle', 'tsmn', 'tscs', 'activity_idx'])))
    processhhmm.prepare_train(additional_cols_to_pad=pad_cols)
    processhhmm.prepare_test(act_encoder=data_train.act_encoder, additional_cols_to_pad=pad_cols, val_set=True)
    processhhmm.prepare_test(act_encoder=data_train.act_encoder, additional_cols_to_pad=pad_cols)
    
    times['hmm_prep_time_end'] = time.perf_counter()
    
    logger.info("HMM fitting started")
    
    processhhmm.fit()

    logger.info("HMM fitting completed")
    
    times['fitting_time'] = time.perf_counter()
    
    N_BATCHES = 8
    
    # taking the sample to predict future elements for from the test set
    predictor = HMMPredictor(processhhmm, pred_args={})
    
    logger.info("HMMPredictor initialized")
    
    times['filtering_time_start'] = time.perf_counter()
    predictor.filter(n_batches=N_BATCHES)
    times['filtering_time_end'] = time.perf_counter()
    
    times['annotation_time_start'] = time.perf_counter()
    
    if export_path is not None:
        annotated_dataset = predictor.annotate_data(split='all', argmax=True, export_path=export_path)
        
    times['annotation_time_end'] = time.perf_counter()
    
    logger.info("data annotation complete")

    return processhhmm, predictor, annotated_dataset

def perform_loaded_run_train(processhhmm, times, export_path):
    
    N_BATCHES = 8
    
    # taking the sample to predict future elements for from the test set
    predictor = HMMPredictor(processhhmm, pred_args={})
    
    logger.info("HMMPredictor initialized")
    
    times['filtering_time_start'] = time.perf_counter()
    predictor.filter(n_batches=N_BATCHES)
    times['filtering_time_end'] = time.perf_counter()
    
    times['annotation_time_start'] = time.perf_counter()
    
    if export_path is not None:
        annotated_dataset = predictor.annotate_data(split='all', argmax=True, export_path=export_path)
    
    times['annotation_time_end'] = time.perf_counter()
    
    logger.info("Data annotation complete")
    
    return processhhmm, predictor, annotated_dataset

def perform_run_train_ppm(ppm_model_params, general_config, data_train, data_val, data_test, annotated_dataset, hmm_model, hmm_predictor, times, additional_params):
    
    logger.info(f"PPM training started - model type: {ppm_model_params['model_type']}")
    
    if ppm_model_params['model_type']=='LSTM':
        log_parser_args = {'case_col': data_train.case_identifier,
                        'activity_col': data_train.activity_identifier,
                        'timestamp_col': data_train.timestamp_identifier,
                        'resource_col': data_train.resource_identifier if not ppm_model_params['embedding_training'] else 'role',
                        'phase_col': 'phase',
                        'hidden_dim_hmm': hmm_model.model_args_train['hidden_dim'], 
                        'seq_len': ppm_model_params['seq_len']}
    
        ppm_model_params['log_parser_args'] = log_parser_args
    
    if ppm_model_params['model_type']=='Tree':
        log_parser_args = {'case_col': data_train.case_identifier,
                        'activity_col': data_train.activity_identifier,
                        'timestamp_col': data_train.timestamp_identifier,
                        'resource_col': data_train.resource_identifier,
                        'phase_col': 'phase',
                        'hidden_dim_hmm': hmm_model.model_args_train['hidden_dim'], 
                        'seq_len': ppm_model_params['seq_len']}
    
        ppm_model_params['log_parser_args'] = log_parser_args
        ppm_model_params['filter_tokens'] = ppm_model_params['filter_sequences']
        ppm_model_params['ncores'] = general_config['ncores']
    
    if ppm_model_params['model_type'] in ['BESTHMMPhases', 'BESTVanillaPhases']:
        ppm_model_params['max_pattern_size'] = ppm_model_params['max_pattern_size_train']
        ppm_model_params['eval_pattern_size'] = ppm_model_params['max_pattern_size_train']
        ppm_model_params['filter_tokens'] = ppm_model_params['filter_sequences']
        ppm_model_params['ncores'] = general_config['ncores']
    
    if ppm_model_params['model_type']=='vanderSpoel':
        ppm_model_params['filter_tokens'] = ppm_model_params['filter_sequences']
        ppm_model_params['ncores'] = general_config['ncores']
    
    ppm_model_params['hmm_model'] = hmm_model
    ppm_model_params['hmm_predictor'] = hmm_predictor
    
    ppm_model_params['resource_col'] = additional_params['resource_col']
    ppm_model_params['seed'] = additional_params['seed']
    ppm_model_params['dataset_name'] = additional_params['dataset']

    ppm_model = PPMFactory.create(ppm_model_params['model_type'], **ppm_model_params)

    times['ppm_prep_start_time'] = time.perf_counter()
    
    ppm_model.load_data(data_train, data_val, data_test, annotated_dataset)
    
    ppm_model.prepare_data()
    
    times['ppm_prep_end_time'] = time.perf_counter()
    times['ppm_fitting_start_time'] = time.perf_counter()
    
    ppm_model.fit()
    
    times['ppm_fitting_end_time'] = time.perf_counter()
    
    return ppm_model

def perform_run_test_ppm(ppm_model: PPMModel, model_params_eval, general_config, times, param_metric_dict, export_path = None) -> None:
    
    times['prediction_ppm_start_time'] = time.perf_counter()
    
    ppm_model.predict(**model_params_eval)
    
    times['prediction_ppm_end_time'] = time.perf_counter()
    
    times['ppm_evaluation_start_time'] = time.perf_counter()
    
    formatted_predictions = ppm_model.format_predictions()
    
    next_activity_predictions = [ap[0][0] if isinstance(ap[0] if len(ap) > 0 else None, tuple) else ap[0] if len(ap) > 0 else None for ap in formatted_predictions]
    activity_suffix_predictions = [[ap[i][0] if len(ap) > 0 else None for i in range(len(ap))] if isinstance(ap[0] if len(ap) > 0 else None, tuple) else ap for ap in formatted_predictions]

    cropped_activity_suffix_predictions = list()
    for asp in activity_suffix_predictions:
        try:
            end_idx = list(asp).index(ppm_model.data_test.end_activity)
        except ValueError:
            end_idx = len(asp) - 1
        casp = asp[:(end_idx + 1)]
        cropped_activity_suffix_predictions.append(list(casp))

    next_activity_actuals = ppm_model.data_test.next_activities
    activity_suffix_actuals = ppm_model.data_test.activity_suffixes

    acc_evaluator = Evaluator(pred=next_activity_predictions, actual=next_activity_actuals)
    ndls_evaluator = Evaluator(pred=cropped_activity_suffix_predictions, actual=activity_suffix_actuals)

    nap_acc = acc_evaluator.calc_accuracy_score()
    nap_balanced_acc = acc_evaluator.calc_balanced_accuracy_score()
    logger.info(f'NAP accuracy: {nap_acc:.4f}')
    logger.info(f'NAP balanced accuracy: {nap_balanced_acc:.4f}')

    ndls = ndls_evaluator.calc_ndls(ncores=general_config['ncores'])
    logger.info(f"SFX similarity: {ndls:.4f}")
    
    times['ppm_evaluation_end_time'] = time.perf_counter()
    
    param_metric_dict['metrics']['nap_accuracy'] = nap_acc
    param_metric_dict['metrics']['nap_balanced_accuracy'] = nap_balanced_acc
    param_metric_dict['metrics']['sfx_similarity'] = ndls    

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
            
            # hmms.append(loaded_hmm)
            slugs.append(param_slug)
        
            # calculate model ranking criterion
            
            if criterion=='loglik':
            
                logger.info(f"Calculating log-likelihood for HMM {param_idx + 1} of {len(param_dict)}...")
                
                loglik = calc_loglik(loaded_hmm)
                
                logger.info(f"Log-likelihood calculated: {loglik}")
                criterion_values.append(loglik)
            elif criterion=='wasserstein':
                
                logger.info(f"Calculating alpha probabilities (val) for HMM {param_idx + 1} of {len(param_dict)}...")
                
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
        # best_hmm = hmms[max_idx]
        # best_hmm = best_hmm
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