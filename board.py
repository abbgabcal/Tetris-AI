from copy import deepcopy
import numpy as np

class Board:
    def __init__(self):
        self.landed = np.zeros(shape=(22,10), dtype=np.int32)

    def printBoard(self):
        for row in self.landed:
            print(row)

    def checkDownCollision(self, p):
        new_loc = [p.loc[0],p.loc[1] + 1]
        for y in range(len(p.current_piece)):
            for x in range(len(p.current_piece[y])):
                if (p.current_piece[y][x] != 0):
                    if y + new_loc[1] >= len(self.landed):
                        return True
                    elif x + new_loc[0] >= len(self.landed[0]):
                        return True
                    elif self.landed[y + new_loc[1]][x + new_loc[0]] != 0:
                        return True 

        return False 
    
    def checkSideCollision(self,p, direction = 0):
        new_loc_side = p.getPotMove(direction)
        for y in range(len(p.current_piece)):
            for x in range(len(p.current_piece[y])):
                if (p.current_piece[y][x] != 0):
                    if x + new_loc_side[0] < 0:
                        return True
                    elif x + new_loc_side[0] >= len(self.landed[y]):
                        return True
                    elif y + new_loc_side[1] >= len(self.landed):
                        return True
                    elif self.landed[y + new_loc_side[1]][x + new_loc_side[0]] != 0:
                        return True 

        return False 

    def checkRotCollision(self, p, direction):
        new_rot = p.getPotRotate(direction)
        for y in range(len(new_rot)):
            for x in range(len(new_rot[y])):
                if (new_rot[y][x] != 0):
                    if x + p.loc[0] < 0:
                        return True
                    elif x + p.loc[0] >= len(self.landed[y]):
                        return True
                    elif y + p.loc[1] >= len(self.landed):
                        return True
                    elif self.landed[y + p.loc[1]][x + p.loc[0]] != 0:
                        return True 

        return False 

