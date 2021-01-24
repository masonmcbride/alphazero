from connect4 import *
from mcts import *
from net import *

numMCTSsims = 50
numEps = 30
numIters = 1

def policyIterSP(game, model):
    dataset = []
    for i in range(numIters):
        print("starting iteration")
        for e in range(numEps):
            dataset += execute_episode(game, model)
            print(f"game {e+1} finished")
        train(model, dataset)

def execute_episode(game, model):
    dataset = []
    game.reset_state()
    node = Node(game)
    tree = MCTS(node, model)

    while True:
        tree.search(numMCTSsims)
        dataset.append([tree.root.encoded_state, tree.pi, None])
        action = tree.sample_from_pi()
        next_state(tree, action)
        game = game.make_action(action)
        if game.is_end_state():
            dataset = assign_rewards(dataset, game.reward)
            return dataset
    
def assign_rewards(dataset, z):
    return [data[:-1] + [z,] for data in dataset]

def next_state(tree, action):
    child = tree.root.children[action]
    child._number_of_visits -= 1
    tree.root = child
    tree.root.parent = None
