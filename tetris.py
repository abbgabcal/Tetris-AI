from board import Board
from pieces import Piece
import numpy as np


class Tetris():
    def __init__(self, start_level = 0, geneticenv = False):
        self.start_level = start_level
        self.board = Board()
        self.piece = Piece()
        self.running = True
        self.rendered_board = np.zeros(shape=(22, 10), dtype=np.int32)
        self.gameframe = 0
        self.cleared_rows = 0
        self.level = self.start_level
        self.points = 0
        self.dropevery = 0
        self.new_piece = True
        self.geneticenv = geneticenv

    def getRender(self) -> np.ndarray:
        rendered_board = np.copy(self.board.landed)
        x = self.piece.loc[0]
        y = self.piece.loc[1]
        for dy in range(len(self.piece.current_piece)):
            for dx in range(len(self.piece.current_piece[dy])):
                if self.piece.current_piece[dy][dx]:
                    rendered_board[y + dy][x + dx] = self.piece.current_piece[dy][dx]
        return rendered_board
    
    def reset(self):
        self.board = Board()
        self.piece = Piece()
        self.running = True
        self.rendered_board = np.zeros(shape=(22, 10), dtype=np.int32)
        self.gameframe = 0
        self.cleared_rows = 0
        self.level = self.start_level
        self.points = 0
        self.dropevery = 0
        self.new_piece = True
        return (self.getRender(), self.piece.next_piece, self.level, self.cleared_rows, self.points, self.running


    def gameloop(self, action=[0, 0, 0]):
        '''
        Run the game one frame.
        Pass in action to manipulate the game, action[0] moves the piece on the x-axis. Pass in -1 to move left, 1 to move right 0 to do nothing.
        Pass in action to manipulate the game, action[1] rotates the piece. Pass in -1 to rotate counterclockwise, 1 to rotate clockwise, 0 to do nothing.
        Pass in action to manipulate the game, action[2] makes the piece drop faster. Pass in 1 to make th epiece drop every two frames. 

        Returns a tuple of the game state. (board, next_piece, level, lines_cleared, score, runnning)
        '''
        landed = False
        self.new_piece = False
        if self.level == 0:
            self.dropevery = 48
        elif self.level == 1: 
            self.dropevery = 43
        elif self.level == 2: 
            self.dropevery = 38
        elif self.level == 3: 
            self.dropevery = 33
        elif self.level == 4: 
            self.dropevery = 28
        elif self.level == 5: 
            self.dropevery = 23
        elif self.level == 6: 
            self.dropevery = 18
        elif self.level == 7: 
            self.dropevery = 13
        elif self.level == 8: 
            self.dropevery = 8
        elif self.level == 9: 
            self.dropevery = 6
        elif 10 <= self.level < 13: 
            self.dropevery = 5
        elif 13 <= self.level < 16: 
            self.dropevery = 4
        elif 16 <= self.level < 19: 
            self.dropevery = 3
        elif 19 <= self.level < 29: 
            self.dropevery = 2
        elif self.level >= 29: 
            self.dropevery = 1


        if not self.board.checkSideCollision(self.piece, action[0]) and action[0] != 0:
            self.piece.movePiece(action[0])

        if action[1] != 0:
            if not self.board.checkRotCollision(self.piece, action[1]):
                self.piece.rotatePiece90(action[1])

        if self.gameframe % self.dropevery == 0 or (action[2] == 1 and self.gameframe % 2 == 0):
            if self.board.checkDownCollision(self.piece):
                for y in range(len(self.piece.current_piece)):
                    for x in range(len(self.piece.current_piece[y])):
                        if self.piece.current_piece[y][x] != 0:
                            self.board.landed[y + self.piece.loc[1]][x +
                                                                     self.piece.loc[0]] = self.piece.current_piece[y][x]
                            landed = True
                            if not np.all((self.board.landed[1] == 0)):
                                self.running = False
            else:
                self.piece.loc[1] += 1
                self.lastCollided = 0

            compLines = 0
            for row in range(len(self.board.landed)):
                if 0 not in self.board.landed[row]:
                    compLines += 1
                    self.board.landed = np.delete(self.board.landed, row, 0)
                    self.board.landed = np.insert(
                        self.board.landed, 0, np.array(0, dtype=np.bool), 0)

            self.cleared_rows += compLines

            if (self.level) * 10 < self.cleared_rows and self.level >= 10:
                self.level += 1

            if compLines == 1:
                self.points += 40 * (self.level + 1)
            elif compLines == 2:
                self.points += 200 * (self.level + 1)

            elif compLines == 3:
                self.points += 600 * (self.level + 1)

            elif compLines == 4:
                self.points += 1200 * (self.level + 1)

            if landed:
                self.piece.newPiece()
                landed = False
                self.new_piece = True



        self.gameframe += 1
        return (self.getRender(), self.piece.next_piece, self.level, self.cleared_rows, self.points, self.running)
