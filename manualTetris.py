from tetris import Tetris
import numpy as np
import time

import pygame as pg

pg.init()

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 800

screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

t = Tetris(start_level=29)

running = True

while running: 
    action = [0,0,0]
    board, next_piece, level, cleared_rows, score, running = t.gameloop(action=action)
    img = t.getRender().astype(np.uint8)
    # video_frame = np.concatenate((board, np.zeros(shape=(10, 10), dtype=np.uint8)))
    # video_frame = np.concatenate((video_frame, np.zeros(shape=(32, 6), dtype=np.uint8)), axis=1)
    video_frame = np.kron(img, np.ones(shape=(32,32), dtype=np.int32)).astype(np.uint8)*255
    # print(img)
    surf = pg.surfarray.make_surface(video_frame.swapaxes(0, 1))
    screen.blit(surf, (0, 0))
    pg.display.update()
    if running == False:
        break

pg.quit()


# video_frame = np.concatenate((board, np.zeros(shape=(10, 10), dtype=np.uint8)))
# video_frame = np.concatenate((video_frame, np.zeros(shape=(32, 6), dtype=np.uint8)), axis=1)
# video_frame = np.kron(board, np.ones(shape=(220,100), dtype=np.int32)).astype(np.uint8)*255
