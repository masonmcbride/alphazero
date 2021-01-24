import numpy as np
import random

class Connect4State:
    def __init__(self, state, player=1):
        self.state = state #holds board
        self.player = player #holds current turn
        
        self.rown = self.state.shape[0] #for indexing
        self.coln = self.state.shape[1] #for indexing

    def __hash__(self):
        return hash(str(self.state))

    def print_state(self):
        print("***current state***")
        print(self.state)

    def make_action(self, action):
        #action will be number 0-6 for the 7 drop locations
        if not self.is_legal_action(action):
            raise ValueError(f"action {action} is not valid")

        new_state = np.copy(self.state) #copy of state because np arrays mutate
        for i, val in enumerate(reversed(self.state[:,action])):
            if val == 0:
                new_state[self.rown-1 - i][action] = self.player #reversed flips indices
                return Connect4State(new_state, -self.player) #returns new object with new state

    def states(self):
        if self.is_end_state():
            return None
        else:
            return [(action, self.make_action(action)) for action in self.all_legal_actions()]

    @property 
    def reward(self):
        result = self.game_result() 
        return result if result is not None else 0

    def is_legal_action(self, action):
        return 0 in self.state[:,action]

    def all_legal_actions(self):
        return [i for i in range(7) if self.is_legal_action(i)]

    def game_result(self):
        def check_frame(frame):
            #input is 4x4 array, output is if it contains a connect 4
            horiz_sum = frame.sum(axis=1)
            vert_sum = frame.sum(axis=0)
            trace = frame.trace()
            rtrace = frame[::-1].trace()
            all_sums = np.concatenate((horiz_sum, vert_sum, [trace], [rtrace]))
            
            if any(s == 4 for s in all_sums):
                return 1

            if any(s == -4 for s in all_sums):
                return -1

            if np.all(self.state):
                return 0.5

        #loop to gather all 4x4 frames
        for i in range(self.rown-3):
            for j in range(self.coln-3):
                result = check_frame(self.state[i:i+4,j:j+4])
                if result is not None:
                    return result
        return None

    def is_end_state(self):
        return self.game_result() is not None

    def reset_state(self):
        self.result = None
        self.player = 1
        self.state.fill(0)


def play_random(game, n):
    game.reset_state()
    for _ in range(2*n):
        action = random.choice(game.all_legal_actions())
        game = game.make_action(action) #make_action returns object, so set game to new object
        result = game.game_result()
        if result is not None:
            game.print_state()
            if result == 0:
                print("RANDOM TIE GAME")
            else:
                print(f"player {result} wins. winning action: {action}")
            return game
    return game

def play_until_tie():
    count = 0
    while play_random(game,50) != 0:
        play_random(game, 50)
        count += 1
    #game.print_state()
    #print(f"number of games it took: {count}")
    return count


