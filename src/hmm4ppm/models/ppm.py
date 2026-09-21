from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
import time

import pyro
import pyro.distributions as dist

from tqdm import tqdm
import copy
import os

from torch.utils.data import DataLoader

from ..data.lstm_log_parser_phase import EventLogParserPhase
from ..data.tree_log_parser_phase import TreeEventLogParserPhase
from .lstm_role_embedding import embedding_training

from hmm4ppm.eval.evaluator import normalized_damerau_levenshtein_similarity
from hmm4ppm.pyro.prediction import HMMPredictor
from hmm4ppm.util.logging import init_logging

from hmm4ppm.models.best import BESTPredictorHMMPhases, BESTPredictorVanillaPhases
from hmm4ppm.models.vanderspoel import VDSPredictorHMMPhases
from hmm4ppm.models.tree_models import TreePredictor

logger = init_logging(__name__, 'ppm.log')

torch.backends.mkldnn.enabled = False

class PPMModel(ABC):

    def __init__(self, **model_args):

        self.model_args = dict()

        for k, v in model_args.items():
            model_args[k] = v

    @abstractmethod
    def load_data(self, **model_args):
        pass
    
    @abstractmethod
    def prepare_data(self, **model_args):
        pass
    
    @abstractmethod
    def fit(self, **model_args):
        pass

    @abstractmethod
    def predict(self, **pred_args):
        pass

    @abstractmethod
    def format_predictions(self, **format_args):
        pass

@dataclass
class CamargoLSTMPhaseTrainConfig:
    # data
    num_activities: int = 20
    num_roles: int = 6
    num_phases: int = 5
    seq_len: int = 5
    ac_rl_weights: tuple[torch.Tensor, torch.Tensor] = None
    num_time_features: int = 1

    # model
    hidden_size: int = 100
    variant: str = "shared_categorical"  # "specialized" | "shared_categorical" | "full_shared"
    dropout: float = 0.2

    # optimization
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    phase_loss_weight: float = 1.0  # relative weight of phase CE loss
    time_loss_weight: float = 1.0  # relative weight of MAE time loss vs. CE losses

    # training control
    early_stopping_patience: int = 5
    grad_clip_norm: float = 5.0
    seed: int = 42

    # misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str = "best_camargo_lstm.pt"
    log_every: int = 1
 
class CamargoLSTMPhase(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_roles: int,
        num_phases: int,
        num_time_features: int,
        activity_emb_dim: int = None,
        role_emb_dim: int = None,
        phase_emb_dim: int = None,
        ac_rl_weights: tuple[torch.Tensor, torch.Tensor] = (None, None), 
        hidden_size: int = 100,
        variant: str = "shared_categorical",  # "specialized" | "shared_categorical" | "full_shared"
        dropout: float = 0.2,
    ):
        super().__init__()
        assert variant in ("specialized", "shared_categorical", "full_shared")
        self.variant = variant
        self.hidden_size = hidden_size

        if ac_rl_weights != (None, None):
            
            activity_emb_dim, role_emb_dim = ac_rl_weights[0].shape[1], ac_rl_weights[1].shape[1]
            self.activity_embedding = nn.Embedding.from_pretrained(torch.tensor(ac_rl_weights[0], dtype=torch.float32),
                                                                   freeze=True)
            self.role_embedding = nn.Embedding.from_pretrained(torch.tensor(ac_rl_weights[1], dtype=torch.float32),
                                                                   freeze=True)
        else:
            activity_emb_dim = activity_emb_dim or math.ceil(num_activities ** 0.25)
            role_emb_dim = role_emb_dim or math.ceil(num_roles ** 0.25)
            self.activity_embedding = nn.Embedding(num_activities, activity_emb_dim)
            self.role_embedding = nn.Embedding(num_roles, role_emb_dim)
        
        phase_emb_dim = phase_emb_dim or min(5, num_phases)
        self.phase_embedding = nn.Embedding(num_phases, phase_emb_dim)
        
        time_input_dim = num_time_features  # relative time is a single scalar per time step
        
        print(f"model dimensions:\n\tactivity embedding: {activity_emb_dim}\n\trole embedding: {role_emb_dim}\n\tphase embedding: {phase_emb_dim}\n\ttimes dim: {time_input_dim}\n\t")

        if variant == "specialized":
            self.lstm1_activity = nn.LSTM(activity_emb_dim, hidden_size, batch_first=True)
            self.lstm1_role = nn.LSTM(role_emb_dim, hidden_size, batch_first=True)
            self.lstm1_phase = nn.LSTM(phase_emb_dim, hidden_size, batch_first=True)
            self.lstm1_time = nn.LSTM(time_input_dim, hidden_size, batch_first=True)

        elif variant == "shared_categorical":
            cat_input_dim = activity_emb_dim + role_emb_dim + phase_emb_dim
            self.lstm1_categorical = nn.LSTM(cat_input_dim, hidden_size, batch_first=True)
            self.lstm1_time = nn.LSTM(time_input_dim, hidden_size, batch_first=True)

        elif variant == "full_shared":
            full_input_dim = activity_emb_dim + role_emb_dim + phase_emb_dim + time_input_dim
            self.lstm1_full = nn.LSTM(full_input_dim, hidden_size, batch_first=True)

        self.dropout1 = nn.Dropout(dropout)

        self.lstm2_activity = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm2_role = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm2_phase = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm2_time = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.dropout2 = nn.Dropout(dropout)

        self.out_activity = nn.Linear(hidden_size, num_activities)
        self.out_role = nn.Linear(hidden_size, num_roles)
        self.out_phase = nn.Linear(hidden_size, num_phases)
        self.out_time = nn.Linear(hidden_size, num_time_features)

    def forward(self, activities, roles, phases, times):
        act_emb = self.activity_embedding(activities)
        role_emb = self.role_embedding(roles)
        phase_emb = self.phase_embedding(phases)

        if self.variant == "specialized":
            act_h1, _ = self.lstm1_activity(act_emb)
            role_h1, _ = self.lstm1_role(role_emb)
            phase_h1, _ = self.lstm1_phase(phase_emb)
            time_h1, _ = self.lstm1_time(times)

        elif self.variant == "shared_categorical":
            cat_in = torch.cat([act_emb, role_emb, phase_emb], dim=-1)
            cat_h1, _ = self.lstm1_categorical(cat_in)
            act_h1, role_h1, phase_h1 = cat_h1, cat_h1, cat_h1  # shared representation feeds both branches
            time_h1, _ = self.lstm1_time(times)

        elif self.variant == "full_shared":
            full_in = torch.cat([act_emb, role_emb, phase_emb, times], dim=-1)
            full_h1, _ = self.lstm1_full(full_in)
            act_h1, role_h1, phase_h1, time_h1 = full_h1, full_h1, full_h1, full_h1

        act_h1 = self.dropout1(act_h1)
        role_h1 = self.dropout1(role_h1)
        phase_h1 = self.dropout1(phase_h1)
        time_h1 = self.dropout1(time_h1)

        act_h2, _ = self.lstm2_activity(act_h1)
        role_h2, _ = self.lstm2_role(role_h1)
        phase_h2, _ = self.lstm2_phase(phase_h1)
        time_h2, _ = self.lstm2_time(time_h1)

        act_last = self.dropout2(act_h2[:, -1, :])
        role_last = self.dropout2(role_h2[:, -1, :])
        phase_last = self.dropout2(phase_h2[:, -1, :])
        time_last = self.dropout2(time_h2[:, -1, :])

        activity_logits = self.out_activity(act_last)
        role_logits = self.out_role(role_last)
        phase_logits = self.out_phase(phase_last)
        time_pred = self.out_time(time_last)

        return activity_logits, role_logits, phase_logits, time_pred

class Trainer:
    def __init__(self, cfg: CamargoLSTMPhaseTrainConfig, 
                 model: nn.Module = None, 
                 ac_rl_weights: tuple[torch.Tensor, torch.Tensor] = None):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        self.device = torch.device(cfg.device)
        self.model = (model or CamargoLSTMPhase(
            num_activities=cfg.num_activities,
            num_roles=cfg.num_roles,
            num_phases=cfg.num_phases,
            num_time_features=cfg.num_time_features,
            ac_rl_weights=cfg.ac_rl_weights,
            hidden_size=cfg.hidden_size,
            variant=cfg.variant,
            dropout=cfg.dropout,
        )).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.mae_loss = nn.L1Loss()

        self.history = {"train_loss": [], "val_loss": [], "val_activity_acc": [], "val_role_acc": [], "val_phase_acc": []}

    def _compute_loss(self, activity_logits, role_logits, phase_logits, time_pred, act_t, role_t, phase_t, time_t):
        loss_act = self.ce_loss(activity_logits, act_t)
        loss_role = self.ce_loss(role_logits, role_t)
        loss_phase = self.ce_loss(phase_logits, phase_t)
        loss_time = self.mae_loss(time_pred, time_t)
        total = loss_act + loss_role + self.cfg.phase_loss_weight * loss_phase + self.cfg.time_loss_weight * loss_time
        return total, {
            "loss_activity": loss_act.item(),
            "loss_role": loss_role.item(),
            "loss_phase": loss_phase.item(),
            "loss_time": loss_time.item(),
        }

    def _run_batch(self, batch, train: bool):
        activities, roles, phases, times, act_t, role_t, phase_t, time_t = [x.to(self.device) for x in batch]

        if train:
            self.optimizer.zero_grad()

        activity_logits, role_logits, phase_logits, time_pred = self.model(activities, roles, phases, times)
        loss, parts = self._compute_loss(activity_logits, role_logits, phase_logits, time_pred, act_t, role_t, phase_t, time_t)

        if train:
            loss.backward()
            if self.cfg.grad_clip_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()

        activity_correct = (activity_logits.argmax(dim=-1) == act_t).sum().item()
        role_correct = (role_logits.argmax(dim=-1) == role_t).sum().item()
        phase_correct = (phase_logits.argmax(dim=-1) == phase_t).sum().item()

        return loss.item(), activity_correct, role_correct, phase_correct, activities.size(0), parts

    def _run_epoch(self, loader: DataLoader, train: bool):
        self.model.train(train)
        total_loss, total_act_correct, total_role_correct, total_phase_correct, total_n = 0.0, 0, 0, 0, 0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                loss, act_correct, role_correct, phase_correct, n, _ = self._run_batch(batch, train=train)
                total_loss += loss * n
                total_act_correct += act_correct
                total_role_correct += role_correct
                total_phase_correct += phase_correct
                total_n += n

        return {
            "loss": total_loss / total_n,
            "activity_acc": total_act_correct / total_n,
            "role_acc": total_role_correct / total_n,
            "phase_acc": total_phase_correct / total_n,
        }
    
    def _run_epoch_suffix(self, suffix_loader: DataLoader, train: bool, max_pred_len: int, use_hmm_phase_filtering: bool, include_phases: bool, end_activity: int, perfect_phase_info: bool, perfect_activity_info: bool, hmm_predictor: HMMPredictor):
        self.model.train(train)
        total_loss, total_n = 0.0, 0
    
        context = torch.enable_grad() if train else torch.no_grad()
        all_act_ndls = list()
        all_role_ndls = list()
        all_phase_ndls = list()
        with context:
            for suffix_batch in tqdm(suffix_loader, desc=f'Predicting for suffix batches...'):
                loss, act_ndls, role_ndls, phase_ndls, n, _ = self._run_batch_suffix(suffix_batch, train=train, max_pred_len=max_pred_len, use_hmm_phase_filtering=use_hmm_phase_filtering, include_phases=include_phases, end_activity=end_activity, perfect_phase_info=perfect_phase_info, perfect_activity_info=perfect_activity_info, hmm_predictor=hmm_predictor)
                total_loss += loss * n
                all_act_ndls.extend(act_ndls)
                all_role_ndls.extend(role_ndls)
                all_phase_ndls.extend(phase_ndls)
                total_n += n
    
        return {
            "suffix_loss": total_loss / total_n,
            "activity_ndls": sum(all_act_ndls) / total_n,
            "role_ndls": sum(all_role_ndls) / total_n,
            "phase_ndls": sum(all_phase_ndls) / total_n,
        }
    
    def _predict_sequences(self, suffix_loader: DataLoader, max_pred_len: int, use_hmm_phase_filtering: bool, include_phases: bool, perfect_phase_info: bool, perfect_activity_info: bool, hmm_predictor: HMMPredictor):
        context = torch.no_grad()
        
        _predicted_sequences = {'act': [], 
                               'role': [], 
                               'phase': [], 
                               'time': [], 
                               }

        with context:
            for suffix_batch in tqdm(suffix_loader, desc=f'Predicting for suffix batches...'):
                predicted_sequences_batch = self._predict_sequences_batch(suffix_batch, max_pred_len=max_pred_len, use_hmm_phase_filtering=use_hmm_phase_filtering, include_phases=include_phases, perfect_phase_info=perfect_phase_info, perfect_activity_info=perfect_activity_info, hmm_predictor=hmm_predictor)
                
                _predicted_sequences['act'].append(predicted_sequences_batch['act'])
                _predicted_sequences['role'].append(predicted_sequences_batch['role'])
                _predicted_sequences['phase'].append(predicted_sequences_batch['phase'])
                _predicted_sequences['time'].append(predicted_sequences_batch['time'])
    
        predicted_sequences = dict()

        for k, v in _predicted_sequences.items():
            predicted_sequences[k] = torch.cat(v)
    
        return predicted_sequences
    
    def _run_batch_suffix(self, batch_suffix, train: bool, max_pred_len: int, use_hmm_phase_filtering: bool, include_phases: bool, end_activity: int, perfect_phase_info: bool, perfect_activity_info: bool, hmm_predictor: HMMPredictor):
        
        activities, roles, phases, times, act_suffix, role_suffix, phase_suffix, time_suffix, indices, lengths = [x.to(self.device) for x in batch_suffix]
        
        if train:
            self.optimizer.zero_grad()

        pred_act_suffix = torch.zeros((batch_suffix[0].shape[0], max_pred_len, )).long()
        pred_role_suffix = torch.zeros((batch_suffix[1].shape[0], max_pred_len, )).long()
        pred_phase_suffix = torch.zeros((batch_suffix[2].shape[0], max_pred_len, )).long()
        pred_time_suffix = torch.zeros((batch_suffix[3].shape[0], max_pred_len, )).float()
        
        suffix_loss = 0
        
        for pred_idx in range(max_pred_len):
            act_t = act_suffix[:,pred_idx]
            role_t = role_suffix[:,pred_idx]
            phase_t = phase_suffix[:,pred_idx]
            time_t = time_suffix[:,pred_idx,:]

            # predict next event and update data
            activity_logits, role_logits, phase_logits, time_pred = self.model(activities, roles, phases, times)
            loss, parts = self._compute_loss(activity_logits, role_logits, phase_logits, time_pred, act_t, role_t, phase_t, time_t)
            
            suffix_loss += loss
            
            if perfect_activity_info:
                pred_act_suffix[:, pred_idx] = act_t
            else:
                pred_act_suffix[:, pred_idx] = activity_logits.argmax(dim=-1)
            
            pred_role_suffix[:, pred_idx] = role_logits.argmax(dim=-1)
            
            if perfect_phase_info:
                pred_phase_suffix[:, pred_idx] = phase_t  # perfect information forecast of phase
            else:
                pred_phase_suffix[:, pred_idx] = phase_logits.argmax(dim=-1)
            pred_time_suffix[:, pred_idx] = time_pred.squeeze(1)
            
            # hybrid HMM/PMM phase extrapolation
            if use_hmm_phase_filtering and include_phases:
                hmm_phases = update_phase_information(indices, lengths, hmm_predictor=hmm_predictor)
                pred_phase_suffix[:, pred_idx] = hmm_phases
            
            activities = torch.cat((activities[:,1:], pred_act_suffix[:, pred_idx].reshape(-1,1)), dim=1)
            roles = torch.cat((roles[:,1:], pred_role_suffix[:, pred_idx].reshape(-1,1)), dim=1)
            phases = torch.cat((phases[:,1:], pred_phase_suffix[:, pred_idx].reshape(-1,1)), dim=1)
            times = torch.cat((times[:,1:,:], pred_time_suffix[:, pred_idx].reshape(-1,1,1)), dim=1)

        if train:
            loss.backward()
            if self.cfg.grad_clip_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()

        pred_mask = first_index(pred_act_suffix, end_activity)
        act_mask = first_index(act_suffix, end_activity)

        activity_ndls = ndls_tensor(pred_act_suffix, act_suffix, pred_mask, act_mask)
        role_ndls = ndls_tensor(pred_role_suffix, role_suffix, pred_mask, act_mask)
        phase_ndls = ndls_tensor(pred_phase_suffix, phase_suffix, pred_mask, act_mask)

        return suffix_loss.item(), activity_ndls, role_ndls, phase_ndls, activities.size(0), parts
    
    def _predict_sequences_batch(self, batch_suffix, max_pred_len: int, use_hmm_phase_filtering: bool, include_phases: bool, perfect_phase_info: bool, perfect_activity_info: bool, hmm_predictor: HMMPredictor):

        activities, roles, phases, times, act_suffix, role_suffix, phase_suffix, time_suffix, indices, lengths = [x.to(self.device) for x in batch_suffix]

        pred_act_suffix = torch.zeros((batch_suffix[0].shape[0], max_pred_len, )).long()
        pred_role_suffix = torch.zeros((batch_suffix[1].shape[0], max_pred_len, )).long()
        pred_phase_suffix = torch.zeros((batch_suffix[2].shape[0], max_pred_len, )).long()
        pred_time_suffix = torch.zeros((batch_suffix[3].shape[0], max_pred_len, )).float()

        predicted_sequences_batch = {'act': pred_act_suffix,
                                     'role': pred_role_suffix,
                                     'phase': pred_phase_suffix,
                                     'time': pred_time_suffix,
                                     }
        
        if use_hmm_phase_filtering and include_phases:
            
            # get alpha probs for whole batch once
            filtered_sequences_tensor = get_filtered_tensor(indices, hmm_predictor=hmm_predictor)
            lengths = hmm_predictor.model.model_args_test['lengths'][indices]
            current_beliefs = filtered_sequences_tensor[lengths, torch.arange(filtered_sequences_tensor.shape[1])]
        
        for pred_idx in range(max_pred_len):
            act_t = act_suffix[:,pred_idx]
            role_t = role_suffix[:,pred_idx]
            phase_t = phase_suffix[:,pred_idx]
            time_t = time_suffix[:,pred_idx,:]

            # predict next event and update data
            activity_logits, role_logits, phase_logits, time_pred = self.model(activities, roles, phases, times)
            
            if perfect_activity_info:
                pred_act_suffix[:, pred_idx] = act_t
            else:
                pred_act_suffix[:, pred_idx] = activity_logits.argmax(dim=-1)

            pred_role_suffix[:, pred_idx] = role_logits.argmax(dim=-1)

            if perfect_phase_info:
                pred_phase_suffix[:, pred_idx] = phase_t  # perfect information forecast of phase
            else:
                pred_phase_suffix[:, pred_idx] = phase_logits.argmax(dim=-1)
            pred_time_suffix[:, pred_idx] = time_pred.squeeze(1)

            if use_hmm_phase_filtering and include_phases:
                hmm_phases = update_phase_information_from_alpha(sequence_indices=indices, alpha=current_beliefs, n_pred_steps=pred_idx+1, hmm_predictor=hmm_predictor)
                pred_phase_suffix[:, pred_idx] = hmm_phases[:,-1]

            activities = torch.cat((activities[:,1:], pred_act_suffix[:, pred_idx].reshape(-1,1)), dim=1)
            roles = torch.cat((roles[:,1:], pred_role_suffix[:, pred_idx].reshape(-1,1)), dim=1)
            phases = torch.cat((phases[:,1:], pred_phase_suffix[:, pred_idx].reshape(-1,1)), dim=1)
            times = torch.cat((times[:,1:,:], pred_time_suffix[:, pred_idx].reshape(-1,1,1)), dim=1)

        return predicted_sequences_batch

    def fit(self, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            elapsed = time.time() - t0

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_activity_acc"].append(val_metrics["activity_acc"])
            self.history["val_role_acc"].append(val_metrics["role_acc"])
            self.history["val_phase_acc"].append(val_metrics["phase_acc"])

            if epoch % self.cfg.log_every == 0:
                logger.info(
                    f"[epoch {epoch:3d}/{self.cfg.epochs}] "
                    f"train_loss={train_metrics['loss']:.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}  "
                    f"val_act_acc={val_metrics['activity_acc']:.3f}  "
                    f"val_role_acc={val_metrics['role_acc']:.3f}  "
                    f"val_phase_acc={val_metrics['phase_acc']:.3f}  "
                    f"({elapsed:.1f}s) "
                    f" - no val improvement for {patience_counter} epochs (patience = {self.cfg.early_stopping_patience})"
                )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                self.epochs_trained = epoch
                torch.save(best_state, self.cfg.checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch} (no val improvement for "
                          f"{self.cfg.early_stopping_patience} epochs).")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.history

    def evaluate(self, test_dataset, test_suffix_dataset = None, end_activity = None, use_hmm_phase_filtering: bool = False, include_phases: bool = False, perfect_phase_info: bool = False, perfect_activity_info: bool = False, hmm_predictor: HMMPredictor = None, eval: bool = True):
        
        if eval:
            test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size, shuffle=False)
            metrics = self._run_epoch(test_loader, train=False)
            print(
                f"[test] loss={metrics['loss']:.4f}  "
                f"activity_acc={metrics['activity_acc']:.3f}  "
                f"role_acc={metrics['role_acc']:.3f}  "
                f"phase_acc={metrics['phase_acc']:.3f}"
            )
            
            if test_suffix_dataset is not None:
                max_pred_len = test_suffix_dataset.activity_suffix.shape[1]
                if end_activity is None:
                    raise ValueError("need a specified end_activity value for predicted sequence cropping")
                test_suffix_loader = DataLoader(test_suffix_dataset, batch_size=self.cfg.batch_size, shuffle=False)
                metrics_ndls = self._run_epoch_suffix(test_suffix_loader, train=False, max_pred_len=max_pred_len, use_hmm_phase_filtering=use_hmm_phase_filtering, include_phases=include_phases, end_activity=end_activity, perfect_phase_info=perfect_phase_info, perfect_activity_info=perfect_activity_info, hmm_predictor=hmm_predictor)
                print(
                    f"[test suffix] loss={metrics_ndls['suffix_loss']:.4f}  "
                    f"activity_ndls={metrics_ndls['activity_ndls']:.3f}  "
                    f"role_ndls={metrics_ndls['role_ndls']:.3f}  "
                    f"phase_ndls={metrics_ndls['phase_ndls']:.3f}"
                )
                
                metrics.update(metrics_ndls)
            
            return metrics
        else:
            if test_suffix_dataset is not None:
                max_pred_len = test_suffix_dataset.activity_suffix.shape[1]
                if end_activity is None:
                    raise ValueError("need a specified end_activity value for predicted sequence cropping")
                test_suffix_loader = DataLoader(test_suffix_dataset, batch_size=self.cfg.batch_size, shuffle=False)
                # self._run_epoch_suffix(test_suffix_loader, train=False, max_pred_len=max_pred_len, end_activity=end_activity, perfect_phase_info=perfect_phase_info, perfect_activity_info=perfect_activity_info, hmm_predictor=hmm_predictor, eval=eval)
                predicted_tensor_sequences = self._predict_sequences(test_suffix_loader, max_pred_len=max_pred_len, use_hmm_phase_filtering=use_hmm_phase_filtering, include_phases=include_phases, perfect_phase_info=perfect_phase_info, perfect_activity_info=perfect_activity_info, hmm_predictor=hmm_predictor)
                
            return predicted_tensor_sequences
                
    def load_checkpoint(self, path: str = None):
        path = path or self.cfg.checkpoint_path
        self.model.load_state_dict(torch.load(path, map_location=self.device))

class LSTM(PPMModel):
    def __init__(self, 
                 variant: Literal['specialized','shared','shared_categorical'], 
                 hidden_size: int, 
                 epochs: int, 
                 embedding_training: bool, 
                 n_epochs_embedding: int, 
                 resource_col: str, 
                 include_phases: bool, 
                 perfect_phase_info: bool, 
                 perfect_activity_info: bool, 
                 load_trained: bool, 
                 hmm_model, 
                 hmm_predictor, 
                 min_prefix_len: int, 
                 dropout: float, 
                 lr: float,
                 batch_size: int, 
                 phase_loss_weight: float, 
                 early_stopping_patience: int, 
                 seed: int, 
                 device: Literal['cpu', 'cuda'] = 'cpu', 
                 log_parser_args: dict = {}, 
                 reuse_fitted: bool = False, 
                 dataset_name: str|None = None, 
                 **model_args):
                
        super().__init__(**model_args)
        
        self.variant = variant
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.embedding_training = embedding_training
        self.n_epochs_embedding = n_epochs_embedding
        self.resource_col = resource_col
        
        self.device = device
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.phase_loss_weight = phase_loss_weight
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        
        self.include_phases = include_phases
        self.perfect_phase_info = perfect_phase_info
        self.perfect_activity_info = perfect_activity_info
        self.load_trained = load_trained
        self.hmm_model = hmm_model
        
        self.min_prefix_len = min_prefix_len
        
        self.log_parser = EventLogParserPhase(**log_parser_args)
        
        self.reuse_fitted = reuse_fitted
        self.dataset_name = dataset_name

        self.hmm_predictor = hmm_predictor

    def load_data(self, data_train, data_val, data_test, phase_annotated_df):
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.phase_annotated_df = phase_annotated_df
    
    def prepare_data(self):
        self.dataset = self.log_parser.load_and_prepare_data(data_train=self.data_train, 
                                                             data_val=self.data_val,
                                                             data_test=self.data_test, 
                                                             phase_annotated_dataset=self.phase_annotated_df, 
                                                             min_prefix_len=self.min_prefix_len, 
                                                             include_phases=self.include_phases,
                                                             embedding_training=self.embedding_training)
        
        self.hmm_end_activity = self.log_parser.activity_vocab.transform(np.array('END').reshape(-1, 1))[0][0]
    
    def fit(self, **model_args):
        
        if self.embedding_training:
            
            self.ac_weights, self.rl_weights, roles_df_train, roles_df_test = embedding_training(data_train=self.data_train, 
                                                                                                 data_test=self.data_test, 
                                                                                                 activity_identifier=self.data_train.activity_identifier, 
                                                                                                 resource_identifier=self.resource_col, 
                                                                                                 n_epochs_embedding=self.n_epochs_embedding)
            self.data_train.data = roles_df_train
            self.data_test.data = roles_df_test
        else:
            self.ac_weights, self.rl_weights = None, None
        
        self.cfg = self.load_config()
        
        logger.info(f"Config: {self.cfg}")
        
        self.train_ds = self.dataset['train']
        self.val_ds = self.dataset['val']
        self.test_ds = self.dataset['test']
        self.val_ds_suffix = self.dataset['val_suffix']
        self.test_ds_suffix = self.dataset['test_suffix']
        
        logger.info(f"train={len(self.train_ds)}  val={len(self.val_ds)}  test={len(self.test_ds)}")
        
        logger.info(f"LSTM training started")
        
        self.trainer = Trainer(self.cfg)
        if self.reuse_fitted and os.path.exists(self.cfg.checkpoint_path):
            logger.info("Found and loaded trained params for config")
            self.trainer.load_checkpoint(self.cfg.checkpoint_path)
            self.trainer.epochs_trained = -1
        else:
            self.trainer.fit(self.train_ds, self.val_ds)
        
        logger.info(f"LSTM training finished")
    
    def predict(self, **pred_args):
        self.predicted_tensor_sequences = self.trainer.evaluate(self.test_ds, self.test_ds_suffix, self.hmm_end_activity, pred_args['use_hmm_phase_filtering'], pred_args['include_phases'], pred_args['perfect_phase_info'], pred_args['perfect_activity_info'], hmm_predictor=self.hmm_predictor, eval=False)
        
    def format_predictions(self, **format_args):
        self.predicted_future_sequences = self.predicted_tensor_sequences['act']
        
        return self.predicted_future_sequences
    
    def load_config(self):
        checkpoint_path_str_list = [f"best_camargo_lstm_{self.dataset_name}_{self.hmm_model.param_slug}",
                                    f"{'_phases_' + str(self.phase_loss_weight/10).split('.')[-1] if self.include_phases else ''}",
                                    f"_h_{self.hidden_size}",
                                    f"_s_{self.log_parser.seq_len}",
                                    f"_v_{self.variant}",
                                    f"_e_{self.epochs}",
                                    f"_lr_{self.lr}",
                                    f".pt"]
        return CamargoLSTMPhaseTrainConfig(num_activities=self.log_parser.num_activities, 
                            num_roles=self.log_parser.num_roles, 
                            num_phases=self.log_parser.num_phases,
                            num_time_features=self.log_parser.num_time_features, 
                            seq_len=self.log_parser.seq_len,
                            ac_rl_weights=(self.ac_weights, self.rl_weights),
                            hidden_size=self.hidden_size, 
                            variant=self.variant, 
                            dropout=self.dropout, 
                            epochs=self.epochs, 
                            lr=self.lr, 
                            weight_decay=0.0, 
                            batch_size=self.batch_size, 
                            phase_loss_weight=self.phase_loss_weight, 
                            time_loss_weight=1.0, 
                            early_stopping_patience=self.early_stopping_patience, 
                            grad_clip_norm=5.0, 
                            seed=self.seed, 
                            device='cpu', 
                            checkpoint_path=''.join(checkpoint_path_str_list),
                            log_every=1)

class BESTHMMPhases(PPMModel):
    def __init__(self, hmm_model, hmm_predictor, **model_args):
                
        super().__init__(**model_args)

        self.best_predictor = BESTPredictorHMMPhases(**model_args)
        self.hmm_model = hmm_model
        self.hmm_predictor = hmm_predictor

    def load_data(self, data_train, data_val, data_test, phase_annotated_df):
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.phase_annotated_df = phase_annotated_df
        
        self.best_predictor.load_data(data_train, data_val, data_test, phase_annotated_df)
    
    def prepare_data(self):
        
        self.best_predictor.prepare_train()
        self.best_predictor.prepare_val() # we do not need act_encoder - already encoded in HMM training
        self.best_predictor.prepare_test() # we do not need act_encoder - already encoded in HMM training
        
        self.hmm_end_activity = self.data_train.act_encoder.transform(np.array('END').reshape(-1, 1))[0][0]
    
    def fit(self, **model_args):
        self.best_predictor.fit()
    
    def predict(self, **pred_args):
        self.best_predictor.predict(**pred_args)
    
    def format_predictions(self, **format_args):
        return self.best_predictor.predicted_sequences
    
class BESTVanillaPhases(PPMModel):
    def __init__(self, hmm_model, hmm_predictor, **model_args):
                
        super().__init__(**model_args)

        self.best_predictor = BESTPredictorVanillaPhases(**model_args)
        self.hmm_model = hmm_model
        self.hmm_predictor = hmm_predictor

    def load_data(self, data_train, data_val, data_test, phase_annotated_df):
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.phase_annotated_df = phase_annotated_df
        
        self.best_predictor.load_data(data_train, data_val, data_test, phase_annotated_df)
    
    def prepare_data(self):
        
        self.best_predictor.prepare_train()
        self.best_predictor.prepare_val() # we do not need act_encoder - already encoded in HMM training
        self.best_predictor.prepare_test() # we do not need act_encoder - already encoded in HMM training
        
        self.hmm_end_activity = self.data_train.act_encoder.transform(np.array('END').reshape(-1, 1))[0][0]
    
    def fit(self, **model_args):
        self.best_predictor.fit()
    
    def predict(self, **pred_args):
        self.best_predictor.predict(**pred_args)
    
    def format_predictions(self, **format_args):
        return self.best_predictor.predicted_sequences

class VDSHMMPhases(PPMModel):
    def __init__(self, hmm_model, hmm_predictor, **model_args):
                
        super().__init__(**model_args)

        self.vds_predictor = VDSPredictorHMMPhases(hmm_model=hmm_model, hmm_predictor=hmm_predictor, **model_args)
        self.hmm_model = hmm_model
        self.hmm_predictor = hmm_predictor

    def load_data(self, data_train, data_val, data_test, phase_annotated_df):
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.phase_annotated_df = phase_annotated_df
        
        self.vds_predictor.load_data(data_train, data_val, data_test, phase_annotated_df)
    
    def prepare_data(self):
        
        self.vds_predictor.prepare_train()
        self.vds_predictor.prepare_val() # we do not need act_encoder - already encoded in HMM training
        self.vds_predictor.prepare_test() # we do not need act_encoder - already encoded in HMM training
        
        self.hmm_end_activity = self.data_train.act_encoder.transform(np.array('END').reshape(-1, 1))[0][0]
    
    def fit(self, **model_args):
        self.vds_predictor.fit()
    
    def predict(self, **pred_args):
        self.vds_predictor.predict(**pred_args)
    
    def format_predictions(self, **format_args):
        return self.vds_predictor.predicted_sequences
    
class TreeHMMPhases(PPMModel):
    def __init__(self, include_phases, predict_phase, pred_model, hmm_model, hmm_predictor, min_prefix_len, log_parser_args, seq_len, **model_args):
                
        super().__init__(**model_args)

        self.tree_predictor = TreePredictor(include_phases=include_phases, predict_phase=predict_phase, pred_model=pred_model, hmm_model=hmm_model, hmm_predictor=hmm_predictor, seq_len=seq_len, **model_args)
        self.hmm_model = hmm_model
        self.hmm_predictor = hmm_predictor
        
        self.include_phases = include_phases
        self.min_prefix_len = min_prefix_len
        self.log_parser = TreeEventLogParserPhase(**log_parser_args)

    def load_data(self, data_train, data_val, data_test, phase_annotated_df):
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.phase_annotated_df = phase_annotated_df
        
        self.tree_predictor.load_data(data_train, data_val, data_test, phase_annotated_df)
    
    def prepare_data(self):
        
        self.tree_predictor.prepare_train()
        self.tree_predictor.prepare_val() # we do not need act_encoder - already encoded in HMM training
        self.tree_predictor.prepare_test() # we do not need act_encoder - already encoded in HMM training
        
        self.tree_predictor.seq_data = self.log_parser.load_and_prepare_data(data_train=self.data_train, 
                                                             data_val=self.data_val, 
                                                             data_test=self.data_test, 
                                                             phase_annotated_dataset=self.phase_annotated_df, 
                                                             min_prefix_len=self.min_prefix_len, 
                                                             include_phases=self.include_phases)
    
        self.hmm_end_activity = self.data_train.act_encoder.transform(np.array('END').reshape(-1, 1))[0][0]
        
    def fit(self, **model_args):
        self.tree_predictor.fit()
    
    def predict(self, **pred_args):
        self.tree_predictor.predict(**pred_args)
    
    def format_predictions(self, **format_args):
        return self.tree_predictor.predicted_sequences

class PPMFactory:
    _model_registry = {
        "LSTM": LSTM,
        "BESTHMMPhases": BESTHMMPhases,
        "BESTVanillaPhases": BESTVanillaPhases,
        "vanderSpoel": VDSHMMPhases,
        "Tree": TreeHMMPhases,
    }

    @classmethod
    def create(cls, model: str, *args, **kwargs):
        try:
            return cls._model_registry[model](*args, **kwargs)
        except KeyError:
            raise ValueError(f"Unknown model '{model}'")

def ndls_tensor(pred: torch.Tensor, actual: torch.Tensor, pred_mask: torch.Tensor, act_mask: torch.Tensor):
    
    ndls = list()
    for seq_idx in range(actual.shape[0]):
        
        pred_list = pred[seq_idx,:].tolist()
        actual_list = actual[seq_idx,:].tolist()
        
        try:
            # TODO
            # phases contain zeros
            # select a different padding value (-1?)
            actual_cropped = actual_list[:act_mask[seq_idx]+1] # we padded suffixes with zeros
        except ValueError:
            actual_cropped = actual_list
        
        try:
            pred_cropped = pred_list[:pred_mask[seq_idx]+1]
        except ValueError:
            pred_cropped = pred_list
    
        ndls.append(normalized_damerau_levenshtein_similarity(pred_cropped, actual_cropped))
    
    return ndls

def first_index(tensor, value, dim = 1):
    mask = (tensor == value)
    exists = mask.any(dim=dim)
    idx = mask.int().argmax(dim=dim)  # first True index per row (ties -> lowest index)
    return [i.item() if e else tensor.shape[dim]-1 for i, e in zip(idx, exists)]

def update_phase_information(sequence_indices: torch.Tensor, sequence_lengths: torch.Tensor, hmm_predictor: HMMPredictor):
    
    num_samples = hmm_predictor.model.model_args_test['num_samples']
    num_sequences = sequence_indices.shape[0]
    probs_transition = hmm_predictor.model.trained_params['AutoDelta.probs_x']
    filtered_sequences = list()
    sequences = hmm_predictor.model.model_args_test['sequences'][sequence_indices,:,:]
    lengths = hmm_predictor.model.model_args_test['lengths'][sequence_indices]
    filtered_sequences_tensor, state_grid = hmm_predictor.filter_external(sequences=sequences, lengths=lengths, log=False)
    
    for seq_idx in range(filtered_sequences_tensor.shape[1]):
        # seq_obs_len = self.model.model_args_test['lengths'][seq_idx] + 1
        seq_obs_len = lengths[seq_idx] + 1
        filtered_sequences.append(filtered_sequences_tensor[:seq_obs_len, seq_idx, :])
        
    # we have the actual alpha values for the indexed prefixes
    # now extrapolate once with those and sample emissions
    # merge emissions with the PPM model predictions (if specified)
    # then filter again and report the next phase
    
    # we pick the last unpadded state of our sequences 
    if hmm_predictor.model.model_args_test['log_prob']:
        current_beliefs = filtered_sequences_tensor[lengths-1, torch.arange(num_sequences)].exp()
    else:
        current_beliefs = filtered_sequences_tensor[lengths-1, torch.arange(num_sequences)]
    
    with pyro.plate("sequences", num_sequences, dim=-1) as seq_idx:
        with pyro.plate("samples", num_samples, dim=-2) as sample_seq_idx:

            tuple_idx = pyro.sample(
                "x_tuple_0",
                dist.Categorical(current_beliefs[seq_idx]) # CHECK IF current_beliefs IS CORRECTLY HANDLED WITH NESTED PLATE STRUCTURE - DO WE NEED FURTHER INDEXING IN OTHER SAMPLE STATEMENTS?
            )
            state_tuple = state_grid[tuple_idx]  # (num_samples, num_sequences, markov_order)

            next_phases = pyro.sample(
                f"x_future",
                dist.Categorical(probs_transition[tuple(state_tuple.permute(2, 0, 1))]) # (num_samples, num_sequences, )
            )
    
    next_phase_mode, _ = next_phases.mode(dim=0)
    
    # # draw phases from filtered_sequences
    # phases = torch.stack([f_seq[-1,:].exp().argmax() for f_seq in filtered_sequences])
    
    return next_phase_mode

def get_filtered_tensor(sequence_indices: torch.Tensor, hmm_predictor: HMMPredictor):
    
    sequences = hmm_predictor.model.model_args_test['sequences'][sequence_indices,:,:]
    lengths = hmm_predictor.model.model_args_test['lengths'][sequence_indices]
    filtered_sequences_tensor, _ = hmm_predictor.filter_external(sequences=sequences, lengths=lengths, marginal_state_probs=False, log=False)
    # filtered_sequences_tensor is of shape n_states^markov_order
    # needs to be marginalized after further filtering steps
    
    return filtered_sequences_tensor

def update_phase_information_from_alpha(sequence_indices: torch.Tensor, alpha: torch.Tensor, n_pred_steps: int, hmm_predictor: HMMPredictor):
    
    num_samples = hmm_predictor.model.model_args_test['num_samples']
    num_sequences = sequence_indices.shape[0]
    probs_transition = hmm_predictor.model.trained_params['AutoDelta.probs_x']
    
    # we pick the last unpadded state of our sequences 
    if hmm_predictor.model.model_args_test['log_prob']:
        alpha = alpha.exp()
    else:
        alpha = alpha
    
    with pyro.plate("sequences", num_sequences, dim=-1) as seq_idx:
        with pyro.plate("samples", num_samples, dim=-2) as sample_seq_idx:

            tuple_idx = pyro.sample(
                "x_tuple_0",
                dist.Categorical(alpha[seq_idx]) # CHECK IF current_beliefs IS CORRECTLY HANDLED WITH NESTED PLATE STRUCTURE - DO WE NEED FURTHER INDEXING IN OTHER SAMPLE STATEMENTS?
            )
            state_tuple = hmm_predictor.state_grid[tuple_idx]  # (num_samples, num_sequences, markov_order)
            full_phases_tensor = state_tuple[:, :, probs_transition.dim():] # just empty but broadcastable tensor for all next phases

            for t in range(n_pred_steps):
                next_phases = pyro.sample(
                    f"x_future_{t}",
                    dist.Categorical(probs_transition[tuple(state_tuple.permute(2, 0, 1))]) # (num_samples, num_sequences, )
                )
                
                state_tuple = torch.cat([state_tuple[:, :, 1:], next_phases.unsqueeze(-1)], dim=2)
                full_phases_tensor = torch.cat([full_phases_tensor, next_phases.unsqueeze(-1)], dim=2)
    
    next_n_phases_mode, _ = full_phases_tensor.mode(dim=0)
    
    return next_n_phases_mode.squeeze(0)