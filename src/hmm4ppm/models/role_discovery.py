# Adapted from Manuel Camargo, Marlon Dumas, Oscar Gonzalez-Rojas
# https://github.com/AdaptiveBProcess/GenerativeLSTM/blob/16eb184093381dd91d0b99493fe52be197cd0de5/support_modules/role_discovery.py
# Not used in the experiments of HMM4PPM

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
    for u in unique:
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
    return roles

def read_resource_pool(log: pd.DataFrame, activity_identifier: str, resource_identifier: str, sim_percentage=0.7):
    act_res_list = log.apply(lambda x: [x[activity_identifier], x[resource_identifier]], axis=1).tolist()
    return role_discovery(data=act_res_list, activity_identifier=activity_identifier, resource_identifier=resource_identifier, sim_percentage=sim_percentage)