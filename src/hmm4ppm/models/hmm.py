import os
import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.ops.indexing import Vindex

from abc import ABC, abstractmethod

assert pyro.__version__.startswith('1.9.1')

class PyroHMM(ABC):

    def __init__(self, **model_args):

        self.model_args = dict()

        for k, v in model_args.items():
            model_args[k] = v

    @abstractmethod
    def model(self, **model_args):
        pass

    @abstractmethod
    def predict(self, **pred_args):
        pass

    @abstractmethod
    def format_predictions(self, **format_args):
        pass

class MultivariateHMM(PyroHMM):

    def __init__(self, **model_args):
        super().__init__(**model_args)
    
    def model(sequences: torch.Tensor, lengths: torch.Tensor, n_categories_per_variable: list[int], emission_types: list[str], hidden_dim: int, markov_order: int = 1, batch_size: int|None = None, include_prior: bool = True, mask_padded: bool = True):
        """Multivariate HMM model suitable for sequence prediction with pyro

        Args:
            sequences (torch.Tensor): A torch.Tensor of multivariate sequential information in padded format. Shape of sequences tensor is (num_sequences, n_emissions, max_length).
                                    The first emission is the one for the activity sequences, i.e., the "main" emission.
            lengths (torch.Tensor): The observed lengths of the respective sequences (not (!) the padded lengths). Shape of sequences tensor is (num_sequences)
            n_categories_per_variable (list[int]): Specify the number of categories per emission variable added. number of categories for continuous 
                                                emissions (emission_types) are skipped, so either value is accepted.
                                                Needs so be same length as emission_types
            emission_types (list[str]): Specifies the type of emission to be either "categorical" or "continuous". Needs to be same length 
                                        as n_categories_per_variable
            hidden_dim (int): The number of hidden states in the HMM. Needs to be specified according to the data analyzed.
            markov_order (int): Positive integer specifying the order of the markov process. Defaults to 1 (true markov process).
            batch_size (int|None, optional): Batch size for training. Defaults to None.
            include_prior (bool, optional): Boolean for inclusion of the specified prior. Used for training purposes but should be 
                                            deactivated during prediction with pyro.infer.predictive.Predictive(). Defaults to True.
            # deprecated params:
            # pred_length (int|None, optional): Length of sequence elements to predict from the given sequence onwards in case of prediction with 
            #                                 pyro.infer.predictive.Predictive(). Defaults to None.
            # max_pred_length (int|None, optional): Maximum length of predicted sequences in case of prediction with pyro.infer.predictive.Predictive(). Defaults to None.
        """
        assert not torch._C._get_tracing_state()
        assert len(n_categories_per_variable) == len(emission_types), "emission_types list needs to be same length as n_categories_per_variable list"

        start_activity_idx = int(sequences[0][0][0])
        end_activity_idx = int(sequences[0][0][-1])

        num_sequences, n_series, max_length = sequences.shape
        with poutine.mask(mask=include_prior):
            eps = 0.5 / hidden_dim
            initial_distribution_prior = torch.ones(hidden_dim) * eps
            initial_distribution_prior[0] = initial_distribution_prior[0]*100
            
            # informative transition_prior for sequential process
            if hidden_dim >= 3:
                transition_prior = torch.full((hidden_dim, hidden_dim), eps)
                v = 10 - (hidden_dim - 1) * eps
                
                # first row
                transition_prior[0, 1] = v/2
                transition_prior[0, -1] = v/2
                
                # intermediate rows
                for i in range(1, hidden_dim - 2):
                    transition_prior[i, i] = v/3
                    transition_prior[i, i+1] = v/3
                transition_prior[1:-2, -1] = v/3
                
                # next to last row
                transition_prior[hidden_dim-2, -2] = v/2
                transition_prior[hidden_dim-2, -1] = v/2
                
                # last row
                transition_prior[-1, -1] = v
            else:
                transition_prior = torch.full((hidden_dim, hidden_dim), eps)

            alpha0 = 1.0        # alpha0 (row sum) dictates peakiness of the Dirichlet prior (the higher, the more uniform, the lower, the peakier)
            transition_prior = transition_prior * alpha0

            initial_distribution = pyro.sample("probs_initial", dist.Dirichlet(initial_distribution_prior))
            
            with pyro.plate("prob_plate", hidden_dim):
                if markov_order > 1:
                    transition_prob = pyro.sample("probs_x", dist.Dirichlet(transition_prior).expand([hidden_dim for _ in range(markov_order)]).to_event(markov_order - 1))
                else:
                    transition_prob = pyro.sample("probs_x", dist.Dirichlet(transition_prior))

            emission_params = [None] * len(emission_types)
            
            for d, (n_cat, emission_type) in enumerate(zip(n_categories_per_variable, emission_types)):

                if emission_type == 'continuous':
                    loc_y = pyro.sample(
                        f"loc_y_{d}",
                        dist.Normal(0.0, 1.0).expand([hidden_dim]).to_event(1),
                    )
                    scale_y = pyro.sample(
                        f"scale_y_{d}",
                        dist.HalfNormal(1.0).expand([hidden_dim]).to_event(1),
                    )
                    emission_params[d] = (loc_y, scale_y)
                elif emission_type == 'gamma':
                    alpha_y = pyro.sample(
                        f"alpha_y_{d}",
                        dist.LogNormal(0.0, 1.0).expand([hidden_dim]).to_event(1),
                    )
                    theta_y = pyro.sample(
                        f"theta_y_{d}",
                        dist.LogNormal(0.0, 1.0).expand([hidden_dim]).to_event(1),
                    )
                    emission_params[d] = (alpha_y, theta_y)
                elif emission_type == 'zeroinflatedgamma':
                    zero_y = pyro.sample(f"zerozero_y_{d}", 
                                         dist.Beta(1.0, 1.0).expand([hidden_dim]).to_event(1))
                    
                    alpha_y = pyro.sample(
                        f"alphazero_y_{d}",
                        dist.LogNormal(0.0, 1.0).expand([hidden_dim]).to_event(1),
                    )
                    theta_y = pyro.sample(
                        f"thetazero_y_{d}",
                        dist.LogNormal(0.0, 1.0).expand([hidden_dim]).to_event(1),
                    )
                    emission_params[d] = (zero_y, alpha_y, theta_y)
                elif emission_type == 'discrete':
                    if d==0: # activity prior
                        activity_prior = torch.ones((hidden_dim, n_cat))/n_cat
                        activity_prior[0, start_activity_idx] = 1
                        activity_prior[-1, end_activity_idx] = 1
                        
                        probs_y = pyro.sample(
                                f"probs_y_{d}",
                                dist.Dirichlet(activity_prior).to_event(1)
                            )

                    else:
                        probs_y = pyro.sample(
                                f"probs_y_{d}",
                                dist.Dirichlet(torch.ones(n_cat)).expand([hidden_dim]).to_event(1)
                            )
                    emission_params[d] = probs_y
                else:
                    raise ValueError(f"unknown emission type: {emission_type}")

        with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch_idx:
            
            seq = sequences[batch_idx,:,:]
            batch_lens = lengths[batch_idx]
            
            mask = (torch.arange(max_length, device=seq.device)
                        .unsqueeze(0) < batch_lens.unsqueeze(1)).unsqueeze(1)
            
            state_history = [torch.tensor(0) for _ in range(markov_order)]
                
            # sample init state
            state_history[-1] = pyro.sample(f"x_0", dist.Categorical(initial_distribution),
                                            infer={"enumerate": "parallel"})
            
            # condition on first observation
            for d, (n_cat, emission_type) in enumerate(zip(n_categories_per_variable, emission_types)):
                if emission_type == 'discrete':
                    pyro.sample(
                        f"y_{d}_0",
                        dist.Categorical(emission_params[d][state_history[-1]]),
                        obs=seq[:, d, 0].unsqueeze(-1),
                    )
                elif emission_type == 'gamma':
                    pyro.sample(
                        f"y_{d}_0",
                        dist.Gamma(emission_params[d][0][state_history[-1]], emission_params[d][1][state_history[-1]]),
                        obs=seq[:, d, 0].unsqueeze(-1),
                        )
                elif emission_type == 'zeroinflatedgamma':
                    is_zero = (seq[:, d, 0].unsqueeze(-1) == 0)
                    
                    pyro.sample(f"y_zero_{d}_0", dist.Bernoulli(emission_params[d][0][state_history[-1]]), obs=is_zero.float())
                    
                    with pyro.poutine.mask(mask=~is_zero):
                        pyro.sample(
                            f"y_{d}_0",
                            dist.Gamma(emission_params[d][1][state_history[-1]], emission_params[d][2][state_history[-1]]),
                            obs=seq[:, d, 0].unsqueeze(-1).clamp_min(1e-8),
                            )
                elif emission_type == 'continuous':
                    pyro.sample(
                        f"y_{d}_0",
                        dist.Normal(emission_params[d][0][state_history[-1]], emission_params[d][1][state_history[-1]]),
                        obs=seq[:, d, 0].unsqueeze(-1),
                        )

            for t in pyro.markov(range(1, max_length), history=markov_order):
                t_mask = mask[:,:,t]
                if not mask_padded:
                # if True:
                    t_mask[:] = True
                    
                with poutine.mask(mask=t_mask if t_mask is not None else torch.tensor(True)):
                    
                    probs_x_t = Vindex(transition_prob)[tuple(state_history)]

                    # Sample next latent state
                    x_t = pyro.sample(
                        f"x_{t}",
                        dist.Categorical(probs_x_t),
                        infer={"enumerate": "parallel"},
                    )
                    state_history = state_history[1:] + [x_t]
                    
                    for d, (n_cat, emission_type) in enumerate(zip(n_categories_per_variable, emission_types)):
                        
                        if emission_type == 'discrete':
                            pyro.sample(
                                f"y_{d}_{t}",
                                dist.Categorical(emission_params[d][x_t]),
                                obs=seq[:, d, t].unsqueeze(-1),
                            )
                        elif emission_type == 'gamma':
                            pyro.sample(
                                f"y_{d}_{t}",
                                dist.Gamma(emission_params[d][0][x_t], emission_params[d][1][x_t]),
                                obs=seq[:, d, t].unsqueeze(-1),
                                )
                        elif emission_type == 'zeroinflatedgamma':
                            is_zero = (seq[:, d, t].unsqueeze(-1) == 0)
                            
                            pyro.sample(f"y_zero_{d}_{t}", dist.Bernoulli(emission_params[d][0][state_history[-1]]), obs=is_zero.float())
                            
                            with pyro.poutine.mask(mask=(~is_zero) & t_mask):
                                
                                pyro.sample(
                                    f"y_{d}_{t}",
                                    dist.Gamma(emission_params[d][1][state_history[-1]], emission_params[d][2][state_history[-1]]),
                                    obs=seq[:, d, t].unsqueeze(-1).clamp_min(1e-8),
                                    )
                        elif emission_type == 'continuous':
                            pyro.sample(
                                f"y_{d}_{t}",
                                dist.Normal(emission_params[d][0][x_t], emission_params[d][1][x_t]),
                                obs=seq[:, d, t].unsqueeze(-1),
                                )
    
    def predict(self, **pred_args):
        raise NotImplementedError("prediction routine not yet implemented")
    
    def format_predictions(self, **format_args):
        raise NotImplementedError("prediction formatting not yet implemented")

class HMMFactory:
    _model_registry = {
        "MultivariateHMM": MultivariateHMM,
    }

    @classmethod
    def create(cls, model: str):
        try:
            return cls._model_registry[model]
        except KeyError:
            raise ValueError(f"Unknown model '{model}'")