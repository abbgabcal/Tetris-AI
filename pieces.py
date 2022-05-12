from random import choice
from math import ceil
import numpy as np


class Pieces:
    def __init__(self):
        self.pieces = [
            [
                [0, 0, 0, 0],
                [7, 7, 7, 7],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1],
                [1, 1],
            ],
            [
                [0, 0, 0],
                [2, 2, 2],
                [0, 2, 0],
            ],
            [
                [0, 0, 0],
                [3, 3, 3],
                [0, 0, 3],
            ],
            [
                [0, 0, 0],
                [4, 4, 4],
                [4, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 5, 5],
                [5, 5, 0]
            ],
            [
                [0, 0, 0],
                [6, 6, 0],
                [0, 6, 6]
            ],
        ]


class Piece(Pieces):
    def __init__(self):
        super().__init__()
        self.next_piece = np.array(choice(self.pieces), dtype=np.int32)
        # self.next_piece = np.array(self.pieces[2], dtype=np.int32)
        self.newPiece()

    def newPiece(self):
        self.current_piece = self.next_piece
        self.next_piece = np.array(choice(self.pieces), dtype=np.int32)
        self.loc = [ceil(5 - len(self.current_piece[0])/2), 0]

    def rotatePiece90(self, direction=0):
        if direction == 1:
            self.current_piece = np.rot90(self.current_piece, k=3)
        elif direction == -1:
            self.current_piece = np.rot90(self.current_piece, k=1)

    def rotatePiece(self, amount):
        self.current_piece = np.rot90(self.current_piece, amount)

    
    def getPotRotate(self, direction = 0):
        if direction == 1:
            return np.rot90(self.current_piece, k=-1)
        elif direction == -1:
            return np.rot90(self.current_piece, k=-1)
        else:
            return self.current_piece

    def getPotMove(self, direction = 0):
        return [self.loc[0] + direction, self.loc[1]]

    def movePiece(self, direction=0):
        self.loc[0] += direction

    def getCurrentPiece(self):
        return self.current_piece