import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import random

from tqdm import tqdm
from ..data.sequencedata import SequenceData

from ..util.logging import init_logging
logger = init_logging(__name__, 'embedding_training.log')

from scipy.stats import pearsonr
import numpy as np
import networkx as nx
from operator import itemgetter
import pandas as pd


def find_index(dictionary, value):
    finish = False
    i = 0
    resp = -1
    while i<len(dictionary) and not finish:
        if dictionary[i]['data']==value:
            resp = dictionary[i]['index']
            finish = True
        i+=1
    return resp

def det_freq_matrix(unique, dictionary, activity_identifier: str, resource_identifier: str):
    freq_matrix = list()
    for u in tqdm(unique, desc='Building role discovery frequency matrix: '):
        freq = 0
        for d in dictionary:
            if u == d:
                freq += 1
        freq_matrix.append({activity_identifier: u[0], resource_identifier: u[1], 'freq': freq})
    return freq_matrix

def build_profile(users, freq_matrix, prof_size, activity_identifier: str, resource_identifier: str):
    profiles=list()
    for user in users:
        exec_tasks = list(filter(lambda x: x[resource_identifier]==user['index'],freq_matrix))
        profile = [0,] * prof_size
        for exec_task in exec_tasks:
            profile[exec_task[activity_identifier]]=exec_task['freq']
        profiles.append({resource_identifier: user['index'], 'profile': profile})
    return profiles

def det_correlation_matrix(profiles, resource_identifier: str):
    correlation_matrix = list()
    for profile_x in profiles:
        for profile_y in profiles:
            x = np.array(profile_x['profile'])
            y = np.array(profile_y['profile'])
            r_row, p_value = pearsonr(x, y)
            correlation_matrix.append(dict(x=profile_x[resource_identifier],y=profile_y[resource_identifier],distance=r_row))
    return correlation_matrix

def role_definition(sub_graphs,users, resource_identifier: str):
    records= list()
    for i in range(0,len(sub_graphs)):
        users_names = list()
        for user in sub_graphs[i]:
            users_names.append(list(filter(lambda x: x['index']==user,users))[0]['data'])
        records.append(dict(role='Role '+ str(i + 1),quantity =len(sub_graphs[i]),members=users_names))
    #Sort roles by number of resources
    records = sorted(records, key=itemgetter('quantity'), reverse=True)
    for i in range(0,len(records)):
        records[i]['role']='Role '+ str(i + 1)
    resource_table = list()
    for record in records:
        for member in record['members']:
            resource_table.append({'role': record['role'], resource_identifier: member})
    return records, resource_table


def role_discovery(data, activity_identifier: str, resource_identifier: str, sim_percentage: float):
    tasks = list(set(list(map(lambda x: x[0], data))))
    tasks = [dict(index=i,data=tasks[i]) for i in range(0,len(tasks))]
    users = list(set(list(map(lambda x: x[1], data))))
    users = [dict(index=i,data=users[i]) for i in range(0,len(users))]
    data_transform = list(map(lambda x: [find_index(tasks, x[0]),find_index(users, x[1])], data ))
    unique = list(set(tuple(i) for i in data_transform))
    unique = [list(i) for i in unique]
    
    # building of a task-size profile of task execution per resource
    profiles = build_profile(users=users, 
                             freq_matrix=det_freq_matrix(unique=unique, 
                                                         dictionary=data_transform, 
                                                         activity_identifier=activity_identifier, 
                                                         resource_identifier=resource_identifier), 
                             prof_size=len(tasks), 
                             activity_identifier=activity_identifier, 
                             resource_identifier=resource_identifier)

    # building of a correlation matrix between resouces profiles
    correlation_matrix = det_correlation_matrix(profiles, resource_identifier=resource_identifier)

    # creation of a relation network between resouces
    g = nx.Graph()
    for user in users:
        g.add_node(user['index'])
    for relation in correlation_matrix:
        # creation of edges between nodes excluding the same element correlation
        # and those below the 0.7 threshold of similarity
        if relation['distance'] > sim_percentage and relation['x']!=relation['y'] :
            g.add_edge(relation['x'],relation['y'],weight=relation['distance'])

    # extraction of fully conected subgraphs as roles
    sub_graphs = [g.subgraph(c) for c in nx.connected_components(g)]

    # role definition from graph
    roles = role_definition(sub_graphs,users, resource_identifier)
    
    logger.info('Role discovery completed!')
    return roles

def read_resource_pool(log: pd.DataFrame, activity_identifier: str, resource_identifier: str, sim_percentage=0.7):
    logger.info('Reading resource pool...')
    act_res_list = log.apply(lambda x: [x[activity_identifier], x[resource_identifier]], axis=1).tolist()
    return role_discovery(data=act_res_list, activity_identifier=activity_identifier, resource_identifier=resource_identifier, sim_percentage=sim_percentage)

class CamargoLSTMRoleEmbedding(nn.Module):
    """Model to embed activities and roles"""
    def __init__(self, ac_index, rl_index, embedding_size):
        super(CamargoLSTMRoleEmbedding, self).__init__()
        # Embedding the activity (output shape will be (None, 1, embedding_size))
        
        self.activity_embedding = nn.Embedding(num_embeddings=len(ac_index),
                                                embedding_dim=embedding_size)
        # Embedding the role (output shape will be (None, 1, embedding_size))
        self.role_embedding = nn.Embedding(num_embeddings=len(rl_index),
                                            embedding_dim=embedding_size)

    def forward(self, activity, role):
        activity_embedding = self.activity_embedding(activity)
        role_embedding = self.role_embedding(role)
        
        # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
        merged = torch.sum(F.normalize(activity_embedding, dim=2) *
                            F.normalize(role_embedding, dim=2),
                            dim=2, keepdim=True)
        # Reshape to be a single number (shape will be (None, 1))
        merged = merged.reshape(-1, 1)
        return merged
    
def train_embedding(log_df, ac_index: dict, rl_index: dict, activity_identifier: str, n_epochs_embedding: int = 100):
    
    logger.info('Preparing activity-role embedding training')
    
    embedding_size = max(1, math.ceil((len(ac_index) * len(rl_index)) ** 0.25))
    
    pairs = list()
    for i in range(0, len(log_df)):
        pairs.append((ac_index[log_df.iloc[i][activity_identifier]], rl_index[log_df.iloc[i]['role']]))
    
    model = CamargoLSTMRoleEmbedding(ac_index, rl_index, embedding_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    n_positive = 1024
    
    logger.info('Generating activity-role pairs')
    gen = generate_batch(pairs, ac_index, rl_index, n_positive, negative_ratio=2)
    
    model.train()
    steps_per_epoch = len(pairs) // n_positive
    for epoch in range(n_epochs_embedding):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            inputs, labels = next(gen)
            activity = torch.tensor(inputs['activity'], dtype=torch.long).unsqueeze(1)
            role = torch.tensor(inputs['role'], dtype=torch.long).unsqueeze(1)
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
            optimizer.zero_grad()
            output = model(activity, role)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{n_epochs_embedding} - loss: {epoch_loss / steps_per_epoch:.4f}")
    
    ac_weights = model.activity_embedding.weight.detach().numpy()
    rl_weights = model.role_embedding.weight.detach().numpy()
    
    return ac_weights, rl_weights

def generate_batch(pairs, ac_index, rl_index, n_positive=50,
                negative_ratio=1.0):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    pairs_set = set(pairs)
    activities = list(ac_index.keys())
    roles = list(rl_index.keys())
    while True:
        # randomly choose positive examples
        idx = 0
        for idx, (activity, role) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (activity, role, 1)
        
        idx += 1

        while idx < batch_size:
            random_ac = random.randrange(len(activities))
            random_rl = random.randrange(len(roles))

            if (random_ac, random_rl) not in pairs_set:
                batch[idx, :] = (random_ac, random_rl, 0)
                idx += 1

        np.random.shuffle(batch)
        yield {'activity': batch[:, 0], 'role': batch[:, 1]}, batch[:, 2]

def create_index(log_df, column):
    """Creates an idx for a categorical attribute.
    Args:
        log_df: dataframe.
        column: column name.
    Returns:
        index of a categorical attribute pairs.
    """
    temp_list = log_df[[column]].values.tolist()
    subsec_set = {(x[0]) for x in temp_list}
    subsec_set = sorted(list(subsec_set))
    alias = dict()
    for i, _ in enumerate(subsec_set):
        alias[subsec_set[i]] = i+2
    
    alias['UNKNOWN'] = 0
    alias['MISSING'] = 1
    
    alias = {k: v for  k, v in sorted(alias.items(), key=lambda x: x[1])}
    return alias

def embedding_training(data_train: SequenceData, data_test: SequenceData, activity_identifier: str, resource_identifier: str, n_epochs_embedding: int = 100):

    log_df_train = data_train.data
    log_df_test = data_test.data
    
    _, resource_table = read_resource_pool(log=log_df_train, activity_identifier=activity_identifier, resource_identifier=resource_identifier)
    
    log_df_resources = pd.DataFrame.from_records(resource_table)
    log_df_combined_train = log_df_train.merge(log_df_resources, on=resource_identifier, how='left')
    log_df_combined_test = log_df_test.merge(log_df_resources, on=resource_identifier, how='left')
    
    ac_index = create_index(log_df_combined_train, activity_identifier)
    index_ac = {v: k for k, v in ac_index.items()}

    rl_index = create_index(log_df_combined_train, 'role')
    index_rl = {v: k for k, v in rl_index.items()}

    ac_weights, rl_weights = train_embedding(log_df_combined_train, ac_index, rl_index, activity_identifier=activity_identifier, n_epochs_embedding=n_epochs_embedding)
    
    return ac_weights, rl_weights, log_df_combined_train, log_df_combined_test