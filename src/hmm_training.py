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

FIG_EXP_DIR = os.path.join(EXPORT_PATH, 'plots')
PARAMS_EXP_DIR = os.path.join(EXPORT_PATH, 'params')
DATASET_EXP_DIR = os.path.join(EXPORT_PATH, 'datasets')

def main():

    general_config = read_config(os.path.join(CONFIG_PATH, "general_config.yml"))
    data_configs = read_config(os.path.join(CONFIG_PATH, "data_configs.yml"))
    model_configs = read_config(os.path.join(CONFIG_PATH, "model_configs.yml"))
    # mlflow_config = read_config(os.path.join(CONFIG_PATH, "mlflow_config.yml"))

    if general_config['model_config'] != 'hmm_training_config':
        logger.warning('Exhaustive HMM parameter training should be performed with model_config: hmm_training_config!')

    FIG_EXP_DIR = os.path.join(EXPORT_PATH, 'plots')
    PARAMS_EXP_DIR = os.path.join(EXPORT_PATH, 'params')
    DATASET_EXP_DIR = os.path.join(EXPORT_PATH, 'datasets')

    os.makedirs(FIG_EXP_DIR, exist_ok=True)
    os.makedirs(PARAMS_EXP_DIR, exist_ok=True)
    os.makedirs(DATASET_EXP_DIR, exist_ok=True)

    # mlflow.set_tracking_uri(mlflow_config["uri"])
    # mlflow.set_experiment(mlflow_config["experiment_name"])

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

        cv_hashes = [random.getrandbits(128) for _ in range(0, len(config_combinations))]
        additional_params['seed'] = general_config['seed']
        additional_params['dataset'] = dataset
        additional_params['resource_col'] = data_config['resource_identifier']

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

            if general_config['cv_folds'] > 1:

                folds = data.train_test_split(train_pct=general_config.get('train_pct'), val_pct=general_config.get('val_pct'), cv=general_config.get('cv_folds'))
                times['data_prep_time'] = time.perf_counter()
                base_cv_hash = cv_hashes[comb_idx]

                # model_params = dict(zip(list(model_config.keys()), [param for param in combination]))
                if phase_model_config is None:
                    raise KeyError('desired model config not found in model_config.yml')

                fold_models = list()

                for fold_idx, fold in enumerate(folds):

                    data_train, data_val, data_test = fold
                    times['run_start_time'] = time.perf_counter()

                    model_params.update({'encoding_params': data_config['encoding_params'],
                                         'transform_params': data_config['transform_params']})

                    fold_models.append(perform_run_train(data_train, data_val, data_test, model_params, times, os.path.join(PARAMS_EXP_DIR, dataset, f"fold_{fold_idx}")))
                    
                    calc_times = {'hmm_prep_duration': times['hmm_prep_time_end'] - times['hmm_prep_time_start'], 
                                  'hmm_train_duration': times['fitting_time'] - times['hmm_prep_time_end'], 
                                  'hmm_filtering_duration': times['filtering_time_end'] - times['filtering_time_start'], 
                                  'hmm_annotation_duration': times['annotation_time_end'] - times['annotation_time_start'], 
                                  }
                    
                    mlflow.log_metrics(calc_times)

            else:                
                data_train, data_val, data_test = data.train_test_split(train_pct=general_config.get('train_pct'), val_pct=general_config.get('val_pct'), cv=general_config.get('cv_folds'))
                times['data_prep_time'] = time.perf_counter()

                if phase_model_config is None:
                    raise KeyError('desired model config not found in model_config.yml')

                times['run_start_time'] = time.perf_counter()

                model_params.update({'encoding_params': data_config['encoding_params'],
                                     'transform_params': data_config['transform_params']})

                for key, value in model_params.items():
                    if value == 'None':
                        model_params[key] = None

                processhhmm, hmm_predictor, annotated_dataset = perform_run_train(data_train, data_val, data_test, model_params, times, os.path.join(PARAMS_EXP_DIR, dataset))

                calc_times = {'hmm_prep_duration': times['hmm_prep_time_end'] - times['hmm_prep_time_start'], 
                              'hmm_train_duration': times['fitting_time'] - times['hmm_prep_time_end'], 
                              'hmm_filtering_duration': times['filtering_time_end'] - times['filtering_time_start'], 
                              'hmm_annotation_duration': times['annotation_time_end'] - times['annotation_time_start'], 
                              }
                
                mlflow.log_metrics(calc_times)

                logger.info("All done!")
                

def perform_run_train(data_train, data_val, data_test, model_params_train, times, export_path = None) -> HMMTrainer:
    
    os.makedirs(export_path, exist_ok=True)
    
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