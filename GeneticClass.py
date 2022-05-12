from copy import deepcopy
from random import randint
from typing import OrderedDict
import numpy as np
from numpy.core.arrayprint import dtype_short_repr
from tetris import Tetris

dtypes = [("weights", tuple), ("score", int)]

class GeneticModel:
    def __init__(self, epsilon : float = 0 ) -> None:
        self.t = Tetris(start_level = 29)
        self.EPSILON = epsilon
    
    def crossover(self, p1 : list, p2 : list) -> tuple:
        """Creates a child tuple of weigths from two parents tuples of weights.

        Args:
            p1 (list): the first parent tuple
            p2 (list): the second parent tuple

        Returns:
            tuple: A tuple representing the child of the two parents. 
        """
        ratio = randint(0, len(p1) - 1)
        child = p1[:ratio] + p2[ratio:]
        return tuple(child)

    def mutate(self, weights : tuple) -> tuple:
        """Mutates a tuple of weights randomly. 

        Args:
            weights (tuple): Target tuple to mutate. 

        Returns:
            tuple: Mutated tuple
        """
        w = list(weights)
        if np.random.rand() < self.EPSILON:
            target = randint(0, len(w) - 1)
        w[target] = np.random.rand()
        return tuple(w)
    
    def getFirstWeights(self, n_episodes : int) -> np.ndarray:
        """Returns an array containing tuples woth the weights for the first n_episodes

        Args:
            n_episodes (int): The amount of initial episodes. 

        Returns:
            np.ndarray: An array of tuples containing the weights for an episode. 
        """
        weight_array = np.array(dtype=tuple)
        for i in range(n_episodes):
            weight_array.append(tuple([np.random.rand() for i in range(3)]))
        return weight_array
    
    def collectData(self, weights : np.ndarray) -> np.ndarray:
        """Collects data by playing one game of tetris for every tuple of weights.  

        Args:
            weights (np.ndarray): Array of tuples containing weigths. 

        Returns:
            np.ndarray: Array of tuples containing the tuple of weights as well as the socre the weights managed to get when playing the game. 
        """
        scores_array = np.array(dtype=dtypes)
        for episode_weigths in weights:
            episode_score = self.runEpisode()
            scores_array.append((episode_weigths, episode_score))
            self.t.reset()
        return scores_array
    
    def getBestWeights(self, score_array : np.ndarray, n : int) -> np.ndarray:
        """Returns the n weigths that generated the best scores. 

        Args:
            score_array (np.ndarray): Array containing the weigths as well as the score the weigths generated. 
            n (int): How many weights the finction should return. Eg n=5 returns the 5 best performing weights. 

        Returns:
            np.ndarray: An array of tuples containeng the best performing weights. 
        """
        sorted_arr = np.sort(score_array, order="score")
        reverse_sorted_array = np.flipud(sorted_arr)
        best_array = reverse_sorted_array[:n]
        best_weigths = np.array([x[0] for x in best_array])
        return best_weigths

    def runEpisode(self, weights : tuple) -> int:
        """Runs the game one time making decisions based on the passed weigths and returns the score. 

        Args:
            weights (tuple): A tuple containing the weigths.

        Returns:
            int: The score the weights managed to get. 
        """
        running = True
        while running:
            action = [0,0,0]
            if self.t.new_piece == True:
                board, next_piece, level, cleared_rows, score, running = self.t.gameloop()
                goal_x, rotation = self.getBestMove(weights)
            else:
                if goal_x < self.t.piece.loc[0]:
                    action[0] = -1
                elif goal_x > self.t.piece.loc[0]:
                    action = 1
                if rotation != 0:
                    action[1] = 1
                    rotation -= 1
            board, next_piece, level, cleared_rows, score, running = self.t.gameloop(action)
        return score
    
    def getBestMove(self, weights):
        w1, w2, w3 = weights
        t_ref = deepcopy(self.t)
        p_ref = t_ref.piece
        prev_best_x = np.array()

        for rotation in range(4):
            p_ref.absoluteRotation(rotation)
            

