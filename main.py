from connect4 import *
from mcts import *
from net import *
from train import *


#initialize board, game, node, and tree        
board = np.zeros((6,7)).astype(float)
game = Connect4State(board)
model = ConnectNet() #this should load the model

policyIterSP(game, model)
#this should save the model

