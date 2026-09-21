import torch
import torch.nn as nn
import math
class CamargoLSTMPhase(nn.Module):
    def __init__(
        self, num_activities: int,
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
        
        phase_emb_dim = phase_emb_dim or math.ceil(num_phases ** 0.25)
        self.phase_embedding = nn.Embedding(num_phases, phase_emb_dim)
        
        time_input_dim = num_time_features
        
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

        self.out_activity = nn.Linear(hidden_size, num_activities)  # softmax via CrossEntropyLoss
        self.out_role = nn.Linear(hidden_size, num_roles)           # softmax via CrossEntropyLoss
        self.out_phase = nn.Linear(hidden_size, num_phases)         # softmax via CrossEntropyLoss
        self.out_time = nn.Linear(hidden_size, num_time_features)   # regression (MAE/MSE loss)

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