import time
import os
import mlflow
import numpy as np
import random
from itertools import chain

from hmm4ppm.data.sequencedata import SequenceData
from hmm4ppm.util.config_utils import read_config

from hmm4ppm.pyro.training import HMMTrainer
from hmm4ppm.pyro.prediction import HMMPredictor
from hmm4ppm.models.hmm import HMMFactory

from hmm4ppm.util.logging import init_logging
logger = init_logging(__name__, 'main.log')

from util.paths import CONFIG_PATH, DATA_PATH, EXPORT_PATH
from util.combinations import param_combinations

import pm4py
import pandas as pd

from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer

FIG_EXP_DIR = os.path.join(EXPORT_PATH, 'plots')
PARAMS_EXP_DIR = os.path.join(EXPORT_PATH, 'params')
DATASET_EXP_DIR = os.path.join(EXPORT_PATH, 'datasets')

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

    if general_config['dataset'] != ['BPI2012_WC'] or general_config['model_config'] != 'process_models_BPI2012_WC_config':
        logger.warning("This script is based on dataset BPI2012_WC and config 'process_models_BPI2012_WC_config' given in configs/model_configs.yml - please change configs/general_config.yml accordingly")

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
        config_combinations = [c_comb for c_comb in combinations_generator]
        additional_params = dict()

        additional_params['seed'] = general_config['seed']
        additional_params['dataset'] = dataset
        additional_params['resource_col'] = data_config['resource_identifier']
        
        eval_args = {'num_samples': phase_model_config['model_params']['num_samples'][0]}

        for comb_idx, model_params in enumerate(config_combinations):

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
                    
            data_train, data_val, data_test = data.train_test_split(train_pct=general_config.get('train_pct'), val_pct=general_config.get('val_pct'), cv=general_config.get('cv_folds'))
            times['data_prep_time'] = time.perf_counter()
            
            if phase_model_config is None:
                raise KeyError('desired model config not found in model_config.yml')
            
            times['run_start_time'] = time.perf_counter()
            
            model_params.update({'encoding_params': data_config['encoding_params'],
                                        'transform_params': data_config['transform_params']})
            
            processhhmm, hmm_predictor, annotated_dataset = perform_run_train(data_train, data_val, data_test, model_params, times, export_path=os.path.join(PARAMS_EXP_DIR, dataset))

            processhhmm.model_args_test.update(eval_args)
            
            # posterior distributions for channels Activity, Resource, TSLE, TSCS
            
            hmm_predictor.complete_posterior_predictive_check_phases_only([processhhmm.data_train.activity_identifier,
                                                                       processhhmm.data_train.resource_identifier,
                                                                       'tsle',
                                                                       'tscs',
                                                                       ], 
                                                                      ['Activity', 'Resource', 'TSLE', 'TSCS'], 
                                                                      ['discrete', 'discrete', 'zeroinflatedgamma', 'zeroinflatedgamma'], 
                                                                      'train', 
                                                                      pred_model=processhhmm)
            
            
            # process models
            raw_df = pd.read_csv(os.path.join(DATA_PATH, data_config["file_name"]), sep=',')
            
            # custom code for paper visualizations            
            # deleting lifecycle:transition information appended to activity column within BPI12_conversions.py
            raw_df[processhhmm.data_train.activity_identifier] = raw_df[processhhmm.data_train.activity_identifier].apply(lambda x: '_'.join(x.split('_')[:-1]) if '_' in x else x)
            annotated_dataset['phase'] = annotated_dataset['phase'].apply(lambda x: f"Phase {x}")
            
            df = pm4py.format_dataframe(raw_df, case_id=processhhmm.data_train.case_identifier,
                                        activity_key=processhhmm.data_train.activity_identifier,
                                        timestamp_key=processhhmm.data_train.timestamp_identifier)
            df_phases = pm4py.format_dataframe(annotated_dataset, case_id=processhhmm.data_train.case_identifier,
                                            activity_key='phase',
                                            timestamp_key=processhhmm.data_train.timestamp_identifier)

            log = pm4py.convert_to_event_log(df)
            log_phases = pm4py.convert_to_event_log(df_phases)
    
            # process model (Heuristics Net (pm4py)) for original activity data
            heu_net = heuristics_miner.apply_heu(log)
            gviz = hn_visualizer.apply(heu_net)
            gviz.attr(ranksep='0.15')
            gviz.attr(nodesep='0.2')
            gviz.attr(size='8,8', ratio='compress')
            gviz.node_attr.update(height="0.5", margin="0.05,0.02", fontsize='20', fontname='Helvetica-Bold')
            hn_visualizer.view(gviz)
            
            # process model (Heuristics Net (pm4py)) for process phase data
            heu_net_phases = heuristics_miner.apply_heu(log_phases)
            gviz_phases = hn_visualizer.apply(heu_net_phases)
            gviz_phases.attr(ranksep='0.15')
            gviz_phases.attr(nodesep='0.2')
            gviz_phases.attr(size='8,8', ratio='compress')
            gviz_phases.node_attr.update(height="0.5", margin="0.05,0.02", fontsize='20', fontname='Helvetica-Bold')
            hn_visualizer.view(gviz_phases)        

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

if __name__=='__main__':
    main()