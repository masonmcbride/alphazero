import random
import numpy as np
from collections import defaultdict
import torch

class MCTS:
    def __init__(self, node, model):
        self.root = node
        self.model = model

    @property
    def pi(self):
        out = torch.zeros(7, dtype=torch.float64)
        for key, child in self.root.children.items():
            out[key] = child.N/self.root.N
        return out

    def select(self):
        current_node = self.root
        while not current_node.is_terminal_state():
            if not current_node.is_fully_expanded():
                break
            else:
                current_node = self.best_child(current_node)
        return current_node

    def expand(self, node, p):
        action = node.untried_actions.pop()
        next_state = node.state.make_action(action)
        child_node = Node(next_state, parent=node, prior=p[0][action])
        node.children[action] = child_node
        return child_node

    def rollout(self, node):
        game = node.state
        while not game.is_end_state():
            action = random.choice(game.all_legal_actions())
            game = game.make_action(action)
        reward = game.reward
        return reward

    def backprop(self, node, value):
        node._total_sum += value
        node._number_of_visits += 1
        if node.parent:
            self.backprop(node.parent, -value)

    def best_child(self, node):
        return max([child for child in node.children.values()], key=self.PUCT)

    def best_action(self):
        return max([key for key in self.root.children], key=lambda i: self.root.children[i].N)

    def sample_from_pi(self):
        return np.random.choice(range(7), p=self.pi)

    def run(self):
        #SELECTION
        node = self.select()

        #EXPANSION
        p, v = self.model(node.encoded_state)
        if not node.is_terminal_state():
            node = self.expand(node, p)
        else:
            v = node.state.reward

        #BACKPROP
        self.backprop(node, v)

    def search(self, n):
        for _ in range(n):
            self.run()

    def UCB(self, node, c_param=2):
        return node.Q + c_param * np.sqrt(np.log(node.parent.N)/node.N)

    def PUCT(self, node, c_param=1.4):
        U = c_param * node.P * np.sqrt(self.root.N) / (1 + node.N)
        return node.Q + U 


class Node:
    def __init__(self, game, parent=None, prior=None):
        self.state = game
        self.player = self.state.player
        self.parent = parent

        self._prior = prior
        self._number_of_visits = 0
        self._total_sum = 0
        self._untried_actions = None
        self.children = {}

    @property
    def N(self):
        return self._number_of_visits

    @property
    def W(self):
        return self._total_sum
    
    @property
    def Q(self):
        return self.W / self.N #average value of node

    @property
    def P(self):
        return self._prior

    @property
    def encoded_state(self):
        game = self.state
        board = game.state
        encoded = np.zeros((3,6,7))
        encoded_dict = {1: 0, -1: 1}
        for row in range(game.rown):
            for col in range(game.coln):
                val = board[row][col]
                if val != 0:
                    encoded[encoded_dict[val]][row, col] = 1

        if game.player == -1:
            encoded[2,:,:] = 1

        return torch.from_numpy(encoded).float()
    
    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.all_legal_actions()
        return self._untried_actions

    def get_children(self):
        for i, state in self.state.states():
            self.children[i] = Node(state, parent=self)

    def is_terminal_state(self):
        return self.state.is_end_state()

    def is_fully_expanded(self):
        return self.untried_actions == []
