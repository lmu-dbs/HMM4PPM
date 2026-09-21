import torch
import torch.nn as nn

class CamargoLSTM(nn.Module):
    def __init__(
        self,
        num_activities: int,
        num_roles: int,
        activity_emb_dim: int = None,
        role_emb_dim: int = None,
        hidden_size: int = 100,
        variant: str = "shared_categorical",  # "specialized" | "shared_categorical" | "full_shared"
        dropout: float = 0.2,
    ):
        super().__init__()
        assert variant in ("specialized", "shared_categorical", "full_shared")
        self.variant = variant
        self.hidden_size = hidden_size

        # Embedding dims default to the paper's heuristic: 4th root of #categories
        activity_emb_dim = activity_emb_dim or max(2, int(num_activities ** 0.25))
        role_emb_dim = role_emb_dim or max(2, int(num_roles ** 0.25))

        self.activity_embedding = nn.Embedding(num_activities, activity_emb_dim)
        self.role_embedding = nn.Embedding(num_roles, role_emb_dim)

        time_input_dim = 1

        if variant == "specialized":
            self.lstm1_activity = nn.LSTM(activity_emb_dim, hidden_size, batch_first=True)
            self.lstm1_role = nn.LSTM(role_emb_dim, hidden_size, batch_first=True)
            self.lstm1_time = nn.LSTM(time_input_dim, hidden_size, batch_first=True)

        elif variant == "shared_categorical":
            cat_input_dim = activity_emb_dim + role_emb_dim
            self.lstm1_categorical = nn.LSTM(cat_input_dim, hidden_size, batch_first=True)
            self.lstm1_time = nn.LSTM(time_input_dim, hidden_size, batch_first=True)

        elif variant == "full_shared":
            full_input_dim = activity_emb_dim + role_emb_dim + time_input_dim
            self.lstm1_full = nn.LSTM(full_input_dim, hidden_size, batch_first=True)

        self.dropout1 = nn.Dropout(dropout)

        self.lstm2_activity = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm2_role = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.lstm2_time = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.dropout2 = nn.Dropout(dropout)

        self.out_activity = nn.Linear(hidden_size, num_activities)
        self.out_role = nn.Linear(hidden_size, num_roles)
        self.out_time = nn.Linear(hidden_size, 1)

    def forward(self, activities, roles, times):
        act_emb = self.activity_embedding(activities)
        role_emb = self.role_embedding(roles)
        
        if self.variant == "specialized":
            act_h1, _ = self.lstm1_activity(act_emb)
            role_h1, _ = self.lstm1_role(role_emb)
            time_h1, _ = self.lstm1_time(times)

        elif self.variant == "shared_categorical":
            cat_in = torch.cat([act_emb, role_emb], dim=-1)
            cat_h1, _ = self.lstm1_categorical(cat_in)
            act_h1, role_h1 = cat_h1, cat_h1  # shared representation feeds both branches
            time_h1, _ = self.lstm1_time(times)

        elif self.variant == "full_shared":
            full_in = torch.cat([act_emb, role_emb, times], dim=-1)
            full_h1, _ = self.lstm1_full(full_in)
            act_h1, role_h1, time_h1 = full_h1, full_h1, full_h1

        act_h1 = self.dropout1(act_h1)
        role_h1 = self.dropout1(role_h1)
        time_h1 = self.dropout1(time_h1)

        act_h2, _ = self.lstm2_activity(act_h1)
        role_h2, _ = self.lstm2_role(role_h1)
        time_h2, _ = self.lstm2_time(time_h1)

        act_last = self.dropout2(act_h2[:, -1, :])
        role_last = self.dropout2(role_h2[:, -1, :])
        time_last = self.dropout2(time_h2[:, -1, :])

        activity_logits = self.out_activity(act_last)
        role_logits = self.out_role(role_last)
        time_pred = self.out_time(time_last)

        return activity_logits, role_logits, time_pred