"""Collection of methods for easy and fast posterior distribution checks for pyro HMM emissions"""

import seaborn as sns
from ..models.hmm import PyroHMM
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

def get_posterior_distribution(model: PyroHMM, channel: str, emission_type: str, mode: Literal['train', 'val', 'test'], alpha_probs: list[torch.Tensor], num_samples: int = int(1e5)):
    
    # gather data for channel
    channel_data = gather_data(model=model, channel=channel, mode=mode)
    
    # gather params for channel
    params = gather_params(model, channel, emission_type)
    
    if emission_type in ['continuous', 'gamma', 'zeroinflatedgamma']: # channel is continuous (normal distribution with loc and scale, gamma with alpha and theta, zeroinflatedgamma with zerozero, alphazero, thetazero)
        state_distribution = get_state_distribution(alpha_probs, model.model_args_train['hidden_dim'])
        posterior_data = posterior_cont(emission_type, params, state_distribution=state_distribution, num_samples=num_samples)
    
    elif emission_type=='discrete': # channel is discrete with emission probs matrix
        state_distribution = get_state_distribution(alpha_probs, model.model_args_train['hidden_dim'])
        posterior_data = posterior_disc(params, state_distribution=state_distribution, num_samples=num_samples)
    else:
        raise ValueError("parameters are of unexpected class object")
    
    return posterior_data, channel_data

def inspect_posterior_distribution(model: PyroHMM, channel: str, emission_type: str, mode: Literal['train', 'test'], alpha_probs: list[torch.Tensor] = None, num_samples: int = int(1e5)):
    
    # gather data for channel
    channel_data = gather_data(model=model, channel=channel, mode=mode)

    # gather params for channel
    params = gather_params(model, channel, emission_type)

    if emission_type in ['continuous', 'gamma', 'zeroinflatedgamma']: # channel is continuous (normal distribution with loc and scale, gamma with alpha and theta, zeroinflatedgamma with zerozero, alphazero, thetazero)
    # if isinstance(params, dict): # channel is continuous (normal distribution with loc and scale)
        if alpha_probs is not None:
            state_distribution = get_state_distribution(alpha_probs, model.model_args_train['hidden_dim'])
            kde(channel_data, emission_type, params, mode=mode, state_distribution=state_distribution, num_samples=num_samples)
        else:
            kde(channel_data, emission_type, params, mode=mode, num_samples=num_samples)

    elif emission_type=='discrete': # channel is discrete with emission probs matrix
    # elif isinstance(params, torch.Tensor): # channel is discrete with emission probs matrix
        # gather posterior state distribution for weighted sampling of categorical emissions per state
        if alpha_probs is not None:
            state_distribution = get_state_distribution(alpha_probs, model.model_args_train['hidden_dim'])
            hist(channel_data, params, mode=mode, state_distribution=state_distribution, num_samples=num_samples)
        else:
            hist(channel_data, params, mode=mode, num_samples=num_samples)
    else:
        raise ValueError("parameters are of unexpected class object")

def inspect_all_posterior_distributions(model: PyroHMM, channels: list[str], channel_labels: list[str], emission_types: list[str], mode: Literal['train', 'test'], alpha_probs: list[torch.Tensor] = None, num_samples: int = int(1e5)):
    # gather data for channel
    channels_data = [gather_data(model=model, channel=channel, mode=mode) for channel in channels]
    
    # gather params for channel
    all_channel_params = [gather_params(model, channel, emission_type) for channel, emission_type in zip(channels, emission_types)]
    
    nrows = len(channels)
    ncols = model.train_args["hidden_dim"] + 1
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12*ncols, 6*nrows))
    
    axes = axes.flatten()
    # posterior plots
    posterior_plot_indices = [ncols*row for row in range(nrows)]
    state_wise_index_ranges = posterior_plot_indices + [ncols*nrows]
    state_wise_plot_indices = [list(range(state_wise_index_ranges[idx]+1, state_wise_index_ranges[idx+1]))  for idx in range(nrows)]
    
    state_distribution = get_state_distribution(alpha_probs, model.model_args_train['hidden_dim'])
    
    for channel_idx, posterior_plot_idx in enumerate(posterior_plot_indices):
        ax = axes[posterior_plot_idx]
        channel_data = channels_data[channel_idx]
        emission_type = emission_types[channel_idx]
        params = all_channel_params[channel_idx]
        
        if channel_idx==0:
            state_titles = True
        else:
            state_titles = False
            
        axis_labels = True
        
        current_channel_label = channel_labels[channel_idx]
        
        if emission_type in ['continuous', 'gamma', 'zeroinflatedgamma']: # channel is continuous (normal distribution with loc and scale, gamma with alpha and theta, zeroinflatedgamma with zerozero, alphazero, thetazero)
            kde_subplots(channel_data, emission_type, params, mode=mode, state_distribution=state_distribution, num_samples=num_samples, ax=ax, state_titles=state_titles, axis_labels=axis_labels, channel_labels=current_channel_label)
        elif emission_type=='discrete':
            hist_subplots(channel_data, params, mode=mode, state_distribution=state_distribution, num_samples=num_samples, ax=ax, state_titles=state_titles, axis_labels=axis_labels, channel_labels=current_channel_label)
        else:
            raise ValueError("parameters are of unexpected class object")
        
    for channel_idx, channel_state_wise_plot_indices in enumerate(state_wise_plot_indices):
        ax = [axes[idx] for idx in channel_state_wise_plot_indices]
        channel_data = channels_data[channel_idx]
        emission_type = emission_types[channel_idx]
        params = all_channel_params[channel_idx]    
        
        if channel_idx==0:
            state_titles = True
        else:
            state_titles = False
        
        axis_labels = False
        
        channel_labels = None
            
        if emission_type in ['continuous', 'gamma', 'zeroinflatedgamma']: # channel is continuous (normal distribution with loc and scale, gamma with alpha and theta, zeroinflatedgamma with zerozero, alphazero, thetazero)
            kde_subplots(channel_data, emission_type, params, mode=mode, num_samples=num_samples, ax=ax, state_titles=state_titles, axis_labels=axis_labels)
        elif emission_type=='discrete':
            hist_subplots(channel_data, params, mode=mode, num_samples=num_samples, ax=ax, state_titles=state_titles, axis_labels=axis_labels)
        else:
            raise ValueError("parameters are of unexpected class object")
    
            
    plt.tight_layout()

def inspect_all_posterior_distributions_phases_only(model: PyroHMM, channels: list[str], channel_labels: list[str], emission_types: list[str], mode: Literal['train', 'test'], alpha_probs: list[torch.Tensor] = None, num_samples: int = int(1e5), pred_model = None):
    # gather data for channel
    channels_data = [gather_data(model=model, channel=channel, mode=mode) for channel in channels]
    
    # gather params for channel
    all_channel_params = [gather_params(model, channel, emission_type) for channel, emission_type in zip(channels, emission_types)]
    
    nrows = model.train_args["hidden_dim"]
    ncols = len(channels)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12*ncols, 5*nrows))
    
    axes = axes.flatten()
    # posterior plots
    row_start_indices = [ncols*row for row in range(nrows)]
    # state_wise_index_ranges = row_start_indices + [ncols*nrows]
    state_wise_plot_indices = [[_ + increment for _ in row_start_indices] for increment in range(ncols)]
    
    state_distribution = get_state_distribution(alpha_probs, model.model_args_train['hidden_dim'])
    
    for channel_idx, channel_state_wise_plot_indices in enumerate(state_wise_plot_indices):
        ax = [axes[idx] for idx in channel_state_wise_plot_indices]
        channel_data = channels_data[channel_idx]
        emission_type = emission_types[channel_idx]
        params = all_channel_params[channel_idx]    
        
        if channel_idx==0:
            state_titles = True
        else:
            state_titles = False
        
        if channel_idx==0:
            axis_labels = True
        else:
            axis_labels = False
            
        if emission_type in ['continuous', 'gamma', 'zeroinflatedgamma']: # channel is continuous (normal distribution with loc and scale, gamma with alpha and theta, zeroinflatedgamma with zerozero, alphazero, thetazero)
            kde_subplots(channel_data, emission_type, params, mode=mode, num_samples=num_samples, ax=ax, state_titles=state_titles, axis_labels=axis_labels, pred_model=pred_model, channel_labels=channel_labels[channel_idx])
        elif emission_type=='discrete':
            hist_subplots(channel_data, params, mode=mode, num_samples=num_samples, ax=ax, state_titles=state_titles, axis_labels=axis_labels, pred_model=pred_model, channel_labels=channel_labels[channel_idx])
        else:
            raise ValueError("parameters are of unexpected class object")
        
    # for channels in horizontal direction
    fig.set_size_inches(8, 6)
    for ax in fig.axes:
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
        ax.title.set_fontsize(12)       
    fig.subplots_adjust(wspace=5)
    fig.tight_layout(pad=0.0, w_pad=0.5, h_pad=-0.2)
    fig.show()
    fig.savefig("phase_posterior_distributions_BPI2012_WC_5_phases.png")

def inspect_state_paths(alpha_probs: list[torch.Tensor], sequences: torch.Tensor, lengths : torch.Tensor, mode: Literal['all', 'random', 'seq_id'] = 'random', seq_id: int = None, ma_horizon: int = None, only_past: bool = True, labels: bool = True, print_trace: bool = True):
    
    lengths_interleaved = torch.arange(lengths.sum()) - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
    seq_ids = torch.repeat_interleave(torch.arange(len(lengths)), lengths)
    
    if mode == 'all':

        plt.figure()
        argmax_states = get_argmax_states(alpha_probs=alpha_probs)
        paths_frame = pd.DataFrame.from_dict({'seq_id': seq_ids, 'idx': lengths_interleaved, 'state': argmax_states.squeeze(1)})
        sns.lineplot(data=paths_frame, x="idx", y="state", hue="seq_id", alpha=0.4, legend=False)
    
    elif mode in ['random', 'seq_id']:
        
        if mode == 'random':
            seq_id = np.random.choice(seq_ids.unique())
        elif seq_id is None:
            raise ValueError("provide a seq_id (int) if mode is chosen to be 'seq_id'")
            
        state_probs = alpha_probs[seq_id]
        
        if ma_horizon is not None:
            state_probs = calc_ma_state_probs(state_probs, horizon=ma_horizon, pad=True, only_past=only_past)
        
        if print_trace:
            print(sequences[seq_id])
        
        path_probs_long = pd.concat([pd.Series(state_probs[:, col_id]) for col_id in range(state_probs.shape[1])], ignore_index=False).reset_index(drop=False)
        path_probs_long["state"] = np.repeat([k for k in range(state_probs.shape[1])], state_probs.shape[0])
        path_probs_long.columns = ["idx", "prob", "state"]
        
        fig, ax = plt.subplots(figsize=(8,5))

        hue_order = sorted(path_probs_long['state'].unique())
        x_order = sorted(path_probs_long['idx'].unique())

        sns.histplot(
            data=path_probs_long,
            x='idx',
            weights='prob',
            hue='state',
            multiple='stack',
            discrete=True,
            ax=ax,
        )

        ax.set_title(f'Posterior latent state probabilities for seq_id {seq_id}')

        if labels:
            for x_val in x_order:
                sub = (
                    path_probs_long[path_probs_long['idx'] == x_val]
                    .set_index('state')
                    .reindex(hue_order)
                    .fillna(0)
                )
                cum = 1
                for cat, row in sub.iterrows():
                    h = row['prob']
                    if h > 0:
                        y = cum - h / 2
                        ax.text(
                            x_val, y, str(cat),
                            ha='center', va='center',
                            fontsize=12, fontweight='bold', color='black'
                        )
                    cum -= h
    
    else:
        raise ValueError("mode needs to be one of ['all', 'random', 'seq_id']")

def kde(channel_data: pd.DataFrame, emission_type: str, params: dict, mode: Literal['train', 'test'], state_distribution: torch.Tensor = None, num_samples: int = int(1e5)):

    if state_distribution is not None:
        col_true = '#0394fc' if mode=='train' else '#e0401f'
        col_posterior = '#c1d41e' if mode=='train' else '#1bbf6a'
        
        fig, ax = plt.subplots(figsize=(12, 6))
        if emission_type=='continuous':
            n_states = params['loc'].size()[0]
        elif emission_type=='gamma':
            n_states = params['alpha'].size()[0]
        elif emission_type=='zeroinflatedgamma':
            n_states = params['alphazero'].size()[0]
        palette = sns.color_palette("cubehelix", n_colors=n_states)
        
        all_kde_data = list()
        # fig, ax = plt.subplots(figsize=(12, 6))
        for state_id in range(n_states):
            state_num_samples = int(num_samples*state_distribution[state_id])
            
            if state_num_samples > 0:
                if emission_type=='continuous':
                    distribution_data = np.random.normal(loc=params['loc'][state_id], scale=params['scale'][state_id], size=(state_num_samples, 1)).flatten()
                elif emission_type=='gamma':
                    distribution_data = np.random.gamma(shape=params['alpha'][state_id], scale=1/params['theta'][state_id], size=(state_num_samples, 1)).flatten()
                elif emission_type=='zeroinflatedgamma':
                    n_zeros = int(state_num_samples * params['zerozero'][state_id])
                    n_non_zeros = state_num_samples - n_zeros
                    zero_distribution_data = np.zeros((n_zeros, 1)).flatten()
                    non_zero_distribution_data = np.random.gamma(shape=params['alphazero'][state_id], scale=1/params['thetazero'][state_id], size=(n_non_zeros, 1)).flatten()
                    distribution_data = np.concatenate((zero_distribution_data, non_zero_distribution_data))
                all_kde_data.append(distribution_data)


        all_kde_data = torch.concat([torch.tensor(d) for d in all_kde_data])

        combined_data = pd.concat((pd.DataFrame.from_dict({'val': channel_data, 'type':'true'}), pd.DataFrame.from_dict({'val': all_kde_data, 'type':'posterior'})), axis=0)
        
        sns.kdeplot(data=combined_data, x='val', hue='type', palette=[col_true, col_posterior], common_norm=False, 
                    # fill=True, 
                    ax=ax) # global channel distribution

        ax.set_title(f"Weighted posterior model distribution of channel {channel_data.name} ({mode})")
        plt.tight_layout()
        plt.show()
    
    else:

        fig, ax = plt.subplots(figsize=(12, 6))
        if emission_type=='continuous':
            n_states = params['loc'].size()[0]
        elif emission_type=='gamma':
            n_states = params['alpha'].size()[0]
        elif emission_type=='zeroinflatedgamma':
            n_states = params['alphazero'].size()[0]
        palette = sns.color_palette("cubehelix", n_colors=n_states)

        sns.kdeplot(data=channel_data, color='black', 
                    # fill=True, 
                    ax=ax,
                    ) # global channel distribution

        global_xlim = ax.get_xlim()
        global_ylim = ax.get_ylim()

        for state_id in range(n_states):
            if emission_type=='continuous':
                distribution_data = np.random.normal(loc=params['loc'][state_id], scale=params['scale'][state_id], size=(num_samples, 1)).flatten()
            elif emission_type=='gamma':
                distribution_data = np.random.gamma(shape=params['alpha'][state_id], scale=1/params['theta'][state_id], size=(num_samples, 1)).flatten()
            elif emission_type=='zeroinflatedgamma':
                # TODO
                # add zero distribution obs from Bernoulli with zerozero param
                n_zeros = int(num_samples * params['zerozero'][state_id])
                n_non_zeros = num_samples - n_zeros
                zero_distribution_data = np.zeros((n_zeros, 1)).flatten()
                non_zero_distribution_data = np.random.gamma(shape=params['alphazero'][state_id], scale=1/params['thetazero'][state_id], size=(n_non_zeros, 1)).flatten()
                distribution_data = np.concatenate((zero_distribution_data, non_zero_distribution_data))
            sns.kdeplot(data=distribution_data, color=palette[state_id], label=f"State {state_id}", 
                        fill=True, 
                        ax=ax)
            
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        ax.set_title(f"State-wise posterior distribution of channel {channel_data.name} ({mode})")
        ax.legend()
        plt.tight_layout()
        plt.show()
    
def kde_subplots(channel_data: pd.DataFrame, emission_type: str, params: dict, mode: Literal['train', 'test'], state_distribution: torch.Tensor = None, num_samples: int = int(1e5), ax = None, state_titles: bool = False, axis_labels: bool = False, channel_labels: list[str]|None = None, pred_model=None):

    channel_identifier = channel_data.name
    if pred_model is not None:
        if emission_type=='continuous':
            enc_name = 'StandardScaler'
        elif emission_type=='gamma':
            enc_name = 'GammaScaler'
        elif emission_type=='zeroinflatedgamma':
            enc_name = 'ZeroInflatedGammaScaler'
        channel_encoder, encoded_names = pred_model.data_train.encoders[enc_name]
        
        if enc_name =='ZeroInflatedGammaScaler':
            raw_min_val = channel_encoder.mins[encoded_names.index(channel_identifier)]
            raw_max_val = channel_encoder.maxs[encoded_names.index(channel_identifier)]
        else:
            raise NotImplementedError('raw min/max ranges only implemented for ZeroInflatedGammaScaler')
    
    else:
        raw_min_val = None
        raw_max_val = None

    if state_distribution is not None:
        col_true = '#0394fc' if mode=='train' else '#e0401f'
        col_posterior = '#c1d41e' if mode=='train' else '#1bbf6a'
        
        if emission_type=='continuous':
            n_states = params['loc'].size()[0]
        elif emission_type=='gamma':
            n_states = params['alpha'].size()[0]
        elif emission_type=='zeroinflatedgamma':
            n_states = params['alphazero'].size()[0]
        palette = sns.color_palette("cubehelix", n_colors=n_states)

        all_kde_data = list()
        for state_id in range(n_states):
            state_num_samples = int(num_samples*state_distribution[state_id])
            
            if state_num_samples > 0:
                if emission_type=='continuous':
                    distribution_data = np.random.normal(loc=params['loc'][state_id], scale=params['scale'][state_id], size=(state_num_samples, 1)).flatten()
                elif emission_type=='gamma':
                    distribution_data = np.random.gamma(shape=params['alpha'][state_id], scale=1/params['theta'][state_id], size=(state_num_samples, 1)).flatten()
                elif emission_type=='zeroinflatedgamma':
                    n_zeros = int(state_num_samples * params['zerozero'][state_id])
                    n_non_zeros = state_num_samples - n_zeros
                    zero_distribution_data = np.zeros((n_zeros, 1)).flatten()
                    non_zero_distribution_data = np.random.gamma(shape=params['alphazero'][state_id], scale=1/params['thetazero'][state_id], size=(n_non_zeros, 1)).flatten()
                    distribution_data = np.concatenate((zero_distribution_data, non_zero_distribution_data))
                all_kde_data.append(distribution_data)


        all_kde_data = torch.concat([torch.tensor(d) for d in all_kde_data])

        combined_data = pd.concat((pd.DataFrame.from_dict({'val': channel_data, 'type':'true'}), pd.DataFrame.from_dict({'val': all_kde_data, 'type':'posterior'})), axis=0)
        
        sns.kdeplot(data=combined_data, x='val', hue='type', palette=[col_true, col_posterior], common_norm=False, 
                    # fill=True, 
                    ax=ax) # global channel distribution
        
        if state_titles:
            ax.set_title(f"Posterior distribution")
            
        if not axis_labels:
            ax.set_ylabel("")
        
        if channel_labels is not None:
            ax.set_ylabel(f"Density\n{channel_labels}")
        
        ax.get_legend().set_title(None)
        
        return combined_data
    
    else:

        # other plot for xlim/ylim
        fig, other_ax = plt.subplots(figsize=(12, 6))
        if emission_type=='continuous':
            n_states = params['loc'].size()[0]
        elif emission_type=='gamma':
            n_states = params['alpha'].size()[0]
        elif emission_type=='zeroinflatedgamma':
            n_states = params['alphazero'].size()[0]
        palette = sns.color_palette("cubehelix", n_colors=n_states)

        sns.kdeplot(data=channel_data, color='black', 
                    # fill=True, 
                    ax=other_ax,
                    ) # global channel distribution

        global_xlim = other_ax.get_xlim()
        global_ylim = other_ax.get_ylim()

        for state_id in range(n_states):
            if emission_type=='continuous':
                distribution_data = np.random.normal(loc=params['loc'][state_id], scale=params['scale'][state_id], size=(num_samples, 1)).flatten()
            elif emission_type=='gamma':
                distribution_data = np.random.gamma(shape=params['alpha'][state_id], scale=1/params['theta'][state_id], size=(num_samples, 1)).flatten()
            elif emission_type=='zeroinflatedgamma':
                n_zeros = int(num_samples * params['zerozero'][state_id])
                n_non_zeros = num_samples - n_zeros
                zero_distribution_data = np.zeros((n_zeros, 1)).flatten()
                non_zero_distribution_data = np.random.gamma(shape=params['alphazero'][state_id], scale=1/params['thetazero'][state_id], size=(n_non_zeros, 1)).flatten()
                distribution_data = np.concatenate((zero_distribution_data, non_zero_distribution_data))
            sns.kdeplot(data=distribution_data, color=palette[state_id], label=f"Phase {state_id}", 
                        fill=True, 
                        ax=ax[state_id])
            
            ax[state_id].set_xlim(global_xlim)
            ax[state_id].set_ylim(global_ylim)
                    
            if state_titles:
                ax[state_id].set_ylabel(f"Phase {state_id}")
            
            if not axis_labels:
                ax[state_id].set_ylabel("")
                
            if channel_labels is not None:
                if state_id == 0:
                    ax[state_id].set_title(channel_labels)
            
            if raw_max_val is not None and raw_min_val is not None:
                min_val = channel_data.min()
                max_val = channel_data.max()
                positions = np.linspace(min_val, max_val, 4)
                ax[state_id].set_xticks(positions)
                raw_labels = np.linspace(raw_min_val, raw_max_val, 4) / 60/60/24 # CAUTION - this is formatting continuous channel data as temporal information (format from seconds to days)
                ax[state_id].set_xticklabels([f"{l:.2f}" for l in raw_labels])
        
        return None

def posterior_cont(emission_type: str, params: dict, state_distribution: torch.Tensor = None, num_samples: int = int(1e5)):

    if emission_type=='continuous':
        n_states = params['loc'].size()[0]
    elif emission_type=='gamma':
        n_states = params['alpha'].size()[0]
    elif emission_type=='zeroinflatedgamma':
        n_states = params['alphazero'].size()[0]

    all_kde_data = list()
    for state_id in range(n_states):
        state_num_samples = int(num_samples*state_distribution[state_id])
        
        if state_num_samples > 0:
            if emission_type=='continuous':
                distribution_data = np.random.normal(loc=params['loc'][state_id], scale=params['scale'][state_id], size=(state_num_samples, 1)).flatten()
            elif emission_type=='gamma':
                distribution_data = np.random.gamma(shape=params['alpha'][state_id], scale=1/params['theta'][state_id], size=(state_num_samples, 1)).flatten()
            elif emission_type=='zeroinflatedgamma':
                n_zeros = int(state_num_samples * params['zerozero'][state_id])
                n_non_zeros = state_num_samples - n_zeros
                zero_distribution_data = np.zeros((n_zeros, 1)).flatten()
                non_zero_distribution_data = np.random.gamma(shape=params['alphazero'][state_id], scale=1/params['thetazero'][state_id], size=(n_non_zeros, 1)).flatten()
                distribution_data = np.concatenate((zero_distribution_data, non_zero_distribution_data))
            all_kde_data.append(distribution_data)

    posterior_data = torch.concat([torch.tensor(d) for d in all_kde_data])

    return posterior_data

def hist(channel_data: pd.DataFrame, params: torch.Tensor, mode: Literal['train', 'test'], state_distribution: torch.Tensor = None, num_samples: int = int(1e5)):

    if state_distribution is not None:
        col_true = '#0394fc' if mode=='train' else '#e0401f'
        col_posterior = '#c1d41e' if mode=='train' else '#1bbf6a'        
                
        fig, ax = plt.subplots(figsize=(12, 6))
        n_states = params.size()[0]
        palette = sns.color_palette("cubehelix", n_colors=n_states)

        # sns.histplot(data=channel_data.astype(int), color='black', stat='density', discrete=True) # global channel distribution

        all_hist_data = list()

        # fig, ax = plt.subplots(figsize=(12, 6))
        for state_id in range(n_states):
            state_probs = params[state_id, :]
            state_num_samples = int(num_samples*state_distribution[state_id])
            
            if state_num_samples > 0:
                hist_data = torch.multinomial(state_probs, num_samples=state_num_samples, replacement=True)
                all_hist_data.append(hist_data)


        all_hist_data = pd.Series(torch.concat(all_hist_data))
        
        combined_data = pd.concat((pd.DataFrame.from_dict({'cat': channel_data, 'type':'true'}), pd.DataFrame.from_dict({'cat': all_hist_data, 'type':'posterior'})), axis=0)
        
        # sns.histplot(data=all_hist_data, stat='density', discrete=True)
        sns.histplot(data=combined_data, x='cat', stat='density', hue='type', multiple='dodge', palette=[col_true, col_posterior], discrete=True, common_norm=False, shrink=0.8)
        
        ax.set_title(f"Weighted posterior model distribution of channel {channel_data.name} ({mode})")
        plt.tight_layout()
        plt.show()
    else:

        fig, ax = plt.subplots(figsize=(12, 6))
        n_states = params.size()[0]
        palette = sns.color_palette("cubehelix", n_colors=n_states)

        sns.histplot(data=channel_data.astype(int), color='black', stat='density', discrete=True) # global channel distribution

        for state_id in range(n_states):
            
            fig = plt.figure()
            state_probs = params[state_id, :]
            hist_data = torch.multinomial(state_probs, num_samples=num_samples, replacement=True)
            
            sns.histplot(data=hist_data, stat='density', 
                            binwidth=1, 
                            color=palette[state_id], 
                            label=f"State {state_id}", 
                            fill=True, 
                            )

            plt.title(f"State {state_id} - Channel {channel_data.name}")
            plt.tight_layout()
            plt.show()
            
def hist_subplots(channel_data: pd.DataFrame, params: torch.Tensor, mode: Literal['train', 'test'], state_distribution: torch.Tensor = None, num_samples: int = int(1e5), ax = None, state_titles: bool = False, axis_labels: bool = False, channel_labels: list[str]|None = None, pred_model = None):

    # CAUTION - only works for one discrete channel (else we would need all discrete channel data to get the categories)
    channel_identifier = channel_data.name
    if pred_model is not None:
        if channel_identifier == pred_model.data_train.activity_identifier:
            channel_encoder = pred_model.data_train.act_encoder
        else:
            channel_encoder = pred_model.data_train.encoders['OrdinalEncoder'][0]
        order = channel_encoder.categories_[0].tolist()
        order_reordered = ['MISSING', 'UNSEEN'] + ['START'] + [_ for _ in order if _ not in ['START', 'END']] + ['END']
        
        channel_data = channel_encoder.inverse_transform(np.array(channel_data).reshape(-1, 1))
        channel_data = pd.Categorical(pd.DataFrame(channel_data).iloc(axis=1)[0], categories=order_reordered, ordered=True)
        
    else:
        channel_encoder = None

    if state_distribution is not None:
        col_true = '#0394fc' if mode=='train' else '#e0401f'
        col_posterior = '#c1d41e' if mode=='train' else '#1bbf6a'        
                
        n_states = params.size()[0]
        palette = sns.color_palette("cubehelix", n_colors=n_states)

        all_hist_data = list()

        for state_id in range(n_states):
            state_probs = params[state_id, :]
            state_num_samples = int(num_samples*state_distribution[state_id])
            
            if state_num_samples > 0:
                hist_data = torch.multinomial(state_probs, num_samples=state_num_samples, replacement=True)
                all_hist_data.append(hist_data)


        all_hist_data = pd.Series(torch.concat(all_hist_data))
        
        combined_data = pd.concat((pd.DataFrame.from_dict({'cat': channel_data, 'type':'true'}), pd.DataFrame.from_dict({'cat': all_hist_data, 'type':'posterior'})), axis=0)
        
        sns.histplot(data=combined_data, x='cat', stat='density', hue='type', multiple='dodge', palette=[col_true, col_posterior], discrete=True, common_norm=False, shrink=0.8, ax=ax)
        
        if state_titles:
            ax.set_title(f"Posterior distribution")
            
        if not axis_labels:
            ax.set_ylabel("")
        
        if channel_labels is not None:
            ax.set_ylabel(f"Density\n{channel_labels}")
        
        ax.get_legend().set_title(None)
        
        return combined_data
    else:

        fig, other_ax = plt.subplots(figsize=(12, 6))
        n_states = params.size()[0]
        palette = sns.color_palette("cubehelix", n_colors=n_states)
        
        try:
            channel_data = channel_data.astype(int)
        except ValueError:
            pass
        
        sns.histplot(data=channel_data, color='black', stat='density', discrete=True) # global channel distribution

        global_xlim = other_ax.get_xlim()
        global_ylim = other_ax.get_ylim()

        for state_id in range(n_states):
            
            state_probs = params[state_id, :]
            hist_data = torch.multinomial(state_probs, num_samples=num_samples, replacement=True)
            
            if channel_encoder is not None:
                hist_data = channel_encoder.inverse_transform(np.array(hist_data).reshape(-1, 1))
                hist_data = pd.Categorical(pd.DataFrame(hist_data).iloc(axis=1)[0], categories=order_reordered, ordered=True)
            
            if len(hist_data.unique())==1:
                val = int(hist_data.unique())
                sns.histplot(data=hist_data, stat='density', 
                                bins=1,
                                binwidth=1,
                                binrange=(val - 0.5, val + 0.5),
                                color=palette[state_id], 
                                label=f"Phase {state_id}", 
                                fill=True, 
                                ax=ax[state_id]
                                )
            else:
                sns.histplot(data=hist_data, stat='density', 
                                binwidth=1,
                                color=palette[state_id], 
                                label=f"Phase {state_id}", 
                                fill=True, 
                                ax=ax[state_id]
                                )
            
            ax[state_id].set_xlim(global_xlim)
            ax[state_id].set_ylim(0, 1)
            
            if state_titles:
                ax[state_id].set_ylabel(f"Phase {state_id}")
            
            if not axis_labels:
                ax[state_id].set_ylabel("")
        
            if channel_labels is not None:
                if state_id == 0:
                    ax[state_id].set_title(channel_labels)
                
            if channel_encoder is not None:
            
                label_map = {'START': 'START', 'END': 'END'}
                ticks = ax[state_id].get_xticks()
                current_labels = [t.get_text() for t in ax[state_id].get_xticklabels()]
                new_labels = [label_map.get(lbl, '') for lbl in current_labels]
            
                ax[state_id].set_xticklabels(new_labels)
        
        return None
            
def posterior_disc(params: torch.Tensor, state_distribution: torch.Tensor, num_samples: int = int(1e5)):

    n_states = params.size()[0]
    
    all_hist_data = list()

    for state_id in range(n_states):
        state_probs = params[state_id, :]
        state_num_samples = int(num_samples*state_distribution[state_id])
        
        if state_num_samples > 0:
            hist_data = torch.multinomial(state_probs, num_samples=state_num_samples, replacement=True)
            all_hist_data.append(hist_data)


    posterior_data = pd.Series(torch.concat(all_hist_data))

    return posterior_data        

def gather_data(model: PyroHMM, channel: str, mode: Literal['train', 'va', 'test']):
    assert hasattr(model, 'data_train'), "model needs to have prepared data!"
    
    if mode == 'train':
        channel_data = model.data_train.data[channel]
    elif mode == 'val':
        channel_data = model.data_val.data[channel]
    elif mode == 'test':
        channel_data = model.data_test.data[channel]
    else:
        raise ValueError("mode hast to be one of 'train' (distributional checks for training data) or 'test' (distributional checks for test data)")
        
    return channel_data

def gather_params(model: PyroHMM, channel: str, emission_type: str):
    assert hasattr(model, 'trained_params'), "model needs to be fitted for posterior distributional checks!"
    
    # we expect that guide is AutoDelta and param names look like 'AutoDelta.param_name_y_channel_id
    channel_id = [k for k in model.data_train.traces[0].keys()][1:].index(channel)
    all_param_names = model.trained_params.keys()
    channel_param_names = [n for n in all_param_names if n.endswith(f"y_{channel_id}")]
    
    if len(channel_param_names) > 1: # channel is continuous (normal distribution with loc and scale, gamma with alpha and theta,...)
        if emission_type=='continuous': # gaussian
            channel_params = {'loc': model.trained_params[[p for p in channel_param_names 
                                    if p.startswith('AutoDelta.loc')][0]], 
                            'scale': model.trained_params[[p for p in channel_param_names 
                                        if p.startswith('AutoDelta.scale')][0]]}
        elif emission_type=='gamma': # gamma
            channel_params = {'alpha': model.trained_params[[p for p in channel_param_names 
                                    if p.startswith('AutoDelta.alpha')][0]], 
                            'theta': model.trained_params[[p for p in channel_param_names 
                                        if p.startswith('AutoDelta.theta')][0]]}
        elif emission_type=='zeroinflatedgamma': # gamma
            channel_params = {'zerozero': model.trained_params[[p for p in channel_param_names 
                                    if p.startswith('AutoDelta.zerozero')][0]], 
                            'alphazero': model.trained_params[[p for p in channel_param_names 
                                        if p.startswith('AutoDelta.alphazero')][0]],
                            'thetazero': model.trained_params[[p for p in channel_param_names 
                                        if p.startswith('AutoDelta.thetazero')][0]]}
        else:
            raise ValueError("unknown distributional form for gathering parameters")
    elif len(channel_param_names) == 1: # channel is discrete with emission probs matrix
        channel_params = model.trained_params[channel_param_names[0]]
    else:
        raise KeyError("channel params show unexpected format")
    
    return channel_params

def get_state_distribution(alpha_probs: list[torch.Tensor], n_states: int, argmax: bool = False):
    
    if argmax:
        argmax_states = get_argmax_states(alpha_probs=alpha_probs)
        state_freqs = torch.bincount(argmax_states.squeeze(1), minlength=n_states)
        state_distribution = state_freqs/state_freqs.sum()
    else:
        non_norm_state_distribution = torch.cat(alpha_probs).sum(axis=0)
        state_distribution = non_norm_state_distribution/non_norm_state_distribution.sum()

    return state_distribution

def get_argmax_states(alpha_probs: list[torch.Tensor]):
    alpha_probs_stack = torch.concat(alpha_probs, dim=0)
    argmax_states = alpha_probs_stack.argmax(dim=1, keepdim=True)
    return argmax_states

def calc_ma_state_probs(probs: torch.Tensor, horizon: int, pad: bool, only_past: bool = True):
    """Calculates a moving average of state probabilities given a horizon of adjacent state probabilities.
    E.g. horizon=5 takes the current probability p_t and averages it with probabilities p_{t-2}, p_{t-1}, p_{t+1} and p_{t+2}
    """
    raw_probs = probs.clone()

    if pad and not only_past:
        raw_probs = torch.cat((raw_probs[0,:].repeat(int(horizon/2),1), raw_probs, raw_probs[-1,:].repeat(int(horizon/2),1)))

    if pad and only_past:
        raw_probs = torch.cat((raw_probs[0,:].repeat(horizon-1,1), raw_probs))
        
    ma_probs = torch.zeros(raw_probs.shape)

    if only_past:
        for i in range(horizon-1, ma_probs.shape[0]):
            idx_sum = raw_probs[i-(horizon-1):i+1,:].sum(axis=0)
            ma_probs[i, :] = idx_sum / horizon
            
        return ma_probs[(horizon-1):]
    else:
        
        for i in range(int(horizon/2), ma_probs.shape[0] - int(horizon/2)):
            idx_sum = raw_probs[i-int(horizon/2):i+int(horizon/2) + 1,:].sum(axis=0)
            ma_probs[i, :] = idx_sum / horizon
    
        return ma_probs[int(horizon/2):-int(horizon/2)]