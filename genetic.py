from tetris import Tetris
import numpy as np
from random import randint, choice
from copy import deepcopy
import imageio


# weigths: heightmax, heightmin, empty_holes

t = Tetris(start_level=29)
EPSILON = 0.8
DEPSILON = 0.99
INITIAL_COLLECT_EPISODES = 100

NUM_GENERATIONS = 100
GENERATION_SIZE = 100

def getBestMove(weights : tuple):
    w1, w2, w3 = weights
    side = False
    t_test = deepcopy(t)
    current_piece = t_test.piece
    prev_high_score = []
    for rotation in range(4):
        if rotation == 0:
            current_piece.rotatePiece90(0)
        else:
            current_piece.current_piece = orig_piece
            current_piece.rotatePiece90(1)
        piece_ref = current_piece.getCurrentPiece()
        orig_piece = current_piece.getCurrentPiece()
        for dx in range(len(current_piece.current_piece), 1, -1):
            if not np.any(current_piece.current_piece[:,dx -1]):
                piece_ref = np.delete(piece_ref, -1, 1)
            else:
                break
        current_piece.current_piece = piece_ref

        for x in range(len(t.board.landed[0]) + len(current_piece.current_piece)): 
            for y in range(len(t.board.landed)):
                current_piece.loc = [x, y]
                if x == 0 and y == 0:
                    x_offset = 0
                    while not t_test.board.checkSideCollision(current_piece, -1):
                        x_offset -= 1
                        current_piece.loc[0] -= 1

                current_piece.loc = [x+x_offset, y]

                if x + len(current_piece.current_piece[0]) + x_offset > len(t.board.landed[0]):
                    break

                if t_test.board.checkDownCollision(current_piece):
                    n_gaps = 0
                    checked_cols = []
                    for dy in range(len(current_piece.current_piece)):
                        for dx in range(len(current_piece.current_piece[0])):
                            if current_piece.current_piece[dy][dx] and dx not in checked_cols:
                                n_gaps += np.count_nonzero(t_test.getRender()[:,current_piece.loc[0]+dx][current_piece.loc[1]+1+dy:] == 0)
                                checked_cols.append(dx)
                    
                    for dy in range(len(current_piece.current_piece)):
                        if np.count_nonzero(current_piece.current_piece[dy] != 0) != 0:
                            top_height = 22 - current_piece.loc[1] - dy
                            break
                    for dy in range(len(current_piece.current_piece), 0, -1):
                        if np.count_nonzero(current_piece.current_piece[dy - 1] != 0) != 0:
                            bottom_height = 22 - current_piece.loc[1] - dy + 1
                            break
                    
                    current_score = w1*n_gaps + w2*top_height + w3*bottom_height
                    if prev_high_score == [] or w1*n_gaps + w2*top_height + w3*bottom_height < prev_high_score[2]:
                        prev_high_score = [x, rotation, w1*n_gaps + w2*top_height + w3*bottom_height]

                    break

            if side:
                side = False
                break
    return (prev_high_score[0], prev_high_score[1])
        
def runEpisodeVideo(weights, filename):
    running = True
    with imageio.get_writer(filename+ '.mp4', fps=60) as video:
        while running:
            action = [0,0,0]
            if t.new_piece == True:
                board, next_piece, level, cleared_rows, score, running = t.gameloop() 
                goal_x, rotation = getBestMove(weights)
            else:
                if goal_x < t.piece.loc[0]:
                    action[0] = -1
                elif goal_x > t.piece.loc[0]:
                    action[0] = 1

                if rotation != 0:
                    action[1] = 1
                    rotation -= 1
            board, next_piece, level, cleared_rows, score, running = t.gameloop(action)
            video_frame = np.concatenate((board, np.zeros(shape=(10, 10), dtype=np.uint8)))
            video_frame = np.concatenate((video_frame, np.zeros(shape=(32, 6), dtype=np.uint8)), axis=1)
            video_frame = np.kron(board, np.ones(shape=(220,100), dtype=np.int32)).astype(np.uint8)*255
            video.append_data(video_frame)
    return score

def runEpisode(weights):
    running = True
    while running:
        action = [0,0,0]
        if t.new_piece == True:
            board, next_piece, level, cleared_rows, score, running = t.gameloop() 
            goal_x, rotation = getBestMove(weights)
        else:
            if goal_x < t.piece.loc[0]:
                action[0] = -1
            elif goal_x > t.piece.loc[0]:
                action[0] = 1

            if rotation != 0:
                action[1] = 1
                rotation -= 1
        board, next_piece, level, cleared_rows, score, running = t.gameloop(action)
    return score

def getBestWeights(array, n):
    sorted_arr = sorted(array, key=lambda x:x[1], reverse=True)[:n]
    best_weights = [x[0] for x in sorted_arr]
    return best_weights

def crossover(parent1, parent2):
    ratio = randint(0,2)
    child = parent1[:ratio] + parent2[ratio:]
    return tuple(child)

def mutate(child):
    c = list(child)
    if np.random.rand() < EPSILON:
        target = randint(0,2)
        c[target] = np.random.rand()
    return tuple(c)


def initialCollect(n_episodes):
    scores_array = []
    for x in range(n_episodes):
        weights = tuple([np.random.rand() for i in range(3)])
        episode_score = runEpisode(weights)
        scores_array.append((weights, episode_score))
        t.reset()
    return scores_array

def collectData(weigths):
    scores_array = []
    for episode in weigths:
        episode_score = runEpisode(episode)
        scores_array.append((episode, episode_score))
        t.reset()
    return scores_array


saved_models = {}

new_gen = []
for generation in range(NUM_GENERATIONS):
    print(f"Learing generation {generation}")
    if generation == 0:
        old_gen = initialCollect(INITIAL_COLLECT_EPISODES)
    else:
        old_gen = collectData(new_gen)
    best = getBestWeights(old_gen, 10)
    while len(new_gen) < GENERATION_SIZE:
        kid = crossover(choice(best), choice(best))
        new_gen.append(mutate(kid))

    if generation + 1 % 10 == 0 or generation == 0:
        runEpisodeVideo(best[0], f'./Videos/Best_of_gen_{generation}')
        saved_models[generation] = best[0]
        print(saved_models)
    



