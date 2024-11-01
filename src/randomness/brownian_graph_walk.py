import networkx as nx
import numpy as np
from src.randomness.brownian_motion import simulate_single_path
from stochastic_control import apply_stochastic_controls
import json
import time
from dask import delayed, compute
import matplotlib.pyplot as plt
from functools import lru_cache
import psutil

G = nx.DiGraph()
bit_states = ['00', '01', '10', '11']
for state in bit_states:
    G.add_node(state)

transitions = {
    ('00', '01'): 1.0,
    ('00', '10'): 1.0,
    ('01', '11'): 0.5,
    ('10', '11'): 0.5,
    ('11', '00'): 0.2
}
for (start, end), weight in transitions.items():
    G.add_edge(start, end, weight=weight)

def load_user_config(config_path="user_config.json"):
    with open(config_path) as f:
        config = json.load(f)
    return config

user_config = load_user_config()

@lru_cache(maxsize=128)
def get_transition_weight(G, current_state, neighbor, brownian_increment):
    base_weight = G[current_state][neighbor]['weight']
    return base_weight * (1 + brownian_increment)

def hybrid_transition(current_state, G, brownian_increment):
    neighbors = list(G.neighbors(current_state))
    if not neighbors:
        return current_state

    weights = [get_transition_weight(G, current_state, neighbor, brownian_increment) for neighbor in neighbors]
    total_weight = sum(weights)
    probabilities = [weight / total_weight for weight in weights]
    next_state = np.random.choice(neighbors, p=probabilities)
    return next_state

def update_transition_weights(G, state_history):
    visit_counts = {state: state_history.count(state) for state in G.nodes()}
    for start in G.nodes():
        total_visits = sum(visit_counts[neighbor] for neighbor in G.neighbors(start))
        for neighbor in G.neighbors(start):
            if total_visits > 0:
                G[start][neighbor]['weight'] = visit_counts[neighbor] / total_visits

def simulate_graph_walk(initial_state='00', T=1, n_steps=20, directional_bias=0.0, variance_adjustment=None):
    brownian_path = simulate_single_path(T, n_steps, 0, T / n_steps, directional_bias, variance_adjustment, np, apply_stochastic_controls)
    current_state = initial_state
    path = [current_state]

    for i in range(n_steps):
        brownian_increment = brownian_path[i + 1] - brownian_path[i]
        current_state = hybrid_transition(current_state, G, brownian_increment)
        path.append(current_state)

    update_transition_weights(G, path)
    return path

def run_parallel_walks(initial_state='00', T=1, n_steps=20, num_paths=10, parallel_threshold=50):
    if num_paths > parallel_threshold:
        paths = [delayed(simulate_graph_walk)(initial_state, T, n_steps) for _ in range(num_paths)]
        computed_paths = compute(*paths)
    else:
        computed_paths = [simulate_graph_walk(initial_state, T, n_steps) for _ in range(num_paths)]
    return computed_paths

def visualize_graph(G, state_history):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    node_sizes = [state_history.count(node) * 100 for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, font_size=12)
    plt.show()

final_path = simulate_graph_walk()
print("Final path taken through the graph:", final_path)
visualize_graph(G, final_path)
