import unittest
from src.randomness.brownian_graph_walk import (
    load_user_config,
    hybrid_transition,
    simulate_graph_walk,
    run_parallel_walks
)
import networkx as nx

class TestGraphWalk(unittest.TestCase):
    def setUp(self):
        self.G = nx.DiGraph()
        bit_states = ['00', '01', '10', '11']
        for state in bit_states:
            self.G.add_node(state)
        transitions = {
            ('00', '01'): 1.0,
            ('00', '10'): 1.0,
            ('01', '11'): 0.5,
            ('10', '11'): 0.5,
            ('11', '00'): 0.2
        }
        for (start, end), weight in transitions.items():
            self.G.add_edge(start, end, weight=weight)

    def test_hybrid_transition(self):
        next_state = hybrid_transition('00', self.G, 0.1)
        self.assertIn(next_state, self.G.nodes)

    def test_simulate_graph_walk(self):
        path = simulate_graph_walk('00')
        self.assertTrue(len(path) > 0)

    def test_run_parallel_walks(self):
        paths = run_parallel_walks('00', num_paths=10)
        self.assertEqual(len(paths), 10)

if __name__ == "__main__":
    unittest.main()
