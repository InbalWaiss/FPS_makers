from enum import IntEnum
from PIL import Image
import numpy as np
from os import path
import pickle

BASELINES_RUN = True
SAVE_BERLIN_FIXED_STATE = False

ACTION_SPACE_9 = True

RED_PLAYER_MOVES = True
FIXED_START_POINT_RED = False
FIXED_START_POINT_BLUE = False
TAKE_WINNING_STEP_BLUE = False

NONEDETERMINISTIC_TERMINAL_STATE = False
SIMULTANEOUS_STEPS = False
if SIMULTANEOUS_STEPS:
    NONEDETERMINISTIC_TERMINAL_STATE = True

CLOSE_START_POSITION = True # if we want the enemy start points to be <MIN_PATH_DIST_FOR_START_POINTS from agent

STR_FOLDER_NAME = "main_berlin_cnn"
COMMON_PATH = path.dirname(path.realpath(__file__))
MAIN_PATH = path.dirname(COMMON_PATH)
OUTPUT_DIR = path.join(MAIN_PATH, 'Arena')
STATS_RESULTS_RELATIVE_PATH = "statistics"
RELATIVE_PATH_HUMAN_VS_MACHINE_DATA = path.join(MAIN_PATH, 'gym_combat/gym_combat/envsQtable/trained_agents')


DSM_names = {"Baqa", "Baqa_Thicken"}
DSM_name = "Baqa_Thicken" #"Baqa"



if DSM_name=="Baqa":
    SIZE_H = 100
    SIZE_W = 100
    DSM = np.loadtxt(path.join(COMMON_PATH, 'maps', 'BaqaObs.txt'), dtype=np.uint8, usecols=range(SIZE_W))
    if False:
        import matplotlib.pyplot as plt

        plt.matshow(DSM)
        plt.show()

    FIRE_RANGE = 10
    LOS_PENALTY_RANGE = 3 * FIRE_RANGE
    MAX_STEPS_PER_EPISODE =150
    MIN_PATH_DIST_FOR_START_POINTS = 2
    BB_STATE = True
    BB_MARGIN = 16

    BB_EXTENSION = 4 * FIRE_RANGE + BB_MARGIN
    SIZE_W_BB = 2 * BB_EXTENSION + 1
    SIZE_H_BB = 2 * BB_EXTENSION + 1

    all_pairs_distances_path = 'gym_combat/gym_combat/envs/Greedy/all_pairs_distances_' + DSM_name + '___' + '.pkl'
    if path.exists(all_pairs_distances_path):
        with open(all_pairs_distances_path, 'rb') as f:
            all_pairs_distances = pickle.load(f)
            print("all_pairs_distances loaded")

    all_pairs_distances_path_np = 'gym_combat/gym_combat/envs/Greedy/all_pairs_distances_' + DSM_name + 'np' + '.pkl'
    if path.exists(all_pairs_distances_path_np):
        with open(all_pairs_distances_path_np, 'rb') as f:
            all_pairs_distances_np = pickle.load(f)
            print("all_pairs_distances_np loaded")

    all_pairs_shortest_path_path = 'gym_combat/gym_combat/envs/Greedy/all_pairs_shortest_pathBaqa_15.pkl'
    if path.exists(all_pairs_shortest_path_path):
        with open(all_pairs_shortest_path_path, 'rb') as f:
            all_pairs_shortest_path = pickle.load(f)
            print("all_pairs_shortest_path loaded")
    SAVE_BERLIN_FIXED_STATE = False

elif DSM_name=="Baqa_Thicken":
    SIZE_H = 100
    SIZE_W = 100
    DSM = np.loadtxt(path.join(COMMON_PATH, 'maps', 'BaqaObs_thicken.txt'), dtype=np.uint8, usecols=range(SIZE_W))
    if False:
        import matplotlib.pyplot as plt

        plt.matshow(DSM)
        plt.show()

    FIRE_RANGE = 10
    LOS_PENALTY_RANGE = 3 * FIRE_RANGE
    MAX_STEPS_PER_EPISODE = 200# 150
    MIN_PATH_DIST_FOR_START_POINTS = 4+FIRE_RANGE
    BB_STATE = True
    BB_MARGIN = 16

    BB_EXTENSION = 4 * FIRE_RANGE + BB_MARGIN
    SIZE_W_BB = 2 * BB_EXTENSION + 1
    SIZE_H_BB = 2 * BB_EXTENSION + 1

    all_pairs_distances_path = path.join(MAIN_PATH, 'Greedy', 'all_pairs_distances_' + DSM_name + '___' + '.pkl')
    if path.exists(all_pairs_distances_path):
        with open(all_pairs_distances_path, 'rb') as f:
            all_pairs_distances = pickle.load(f)
            print("all_pairs_distances loaded")

    all_pairs_distances_path_np = path.join(MAIN_PATH, 'Greedy', 'all_pairs_distances_' + DSM_name + 'np' + '.pkl')
    if path.exists(all_pairs_distances_path_np):
        with open(all_pairs_distances_path_np, 'rb') as f:
            all_pairs_distances_np = pickle.load(f)
            print("all_pairs_distances_np loaded")

    all_pairs_shortest_path_path = path.join(MAIN_PATH, 'Greedy', 'all_pairs_shortest_pathBaqa_15.pkl')
    if path.exists(all_pairs_shortest_path_path):
        with open(all_pairs_shortest_path_path, 'rb') as f:
            all_pairs_shortest_path = pickle.load(f)
            print("all_pairs_shortest_path loaded")
    SAVE_BERLIN_FIXED_STATE = False

OBSTACLE = 1. #1 is an obstacle


### Load preprocess files ###
try:
    DICT_POS_FIRE_RANGE_path = path.join(COMMON_PATH, 'Preprocessing', 'dictionary_position_los_' + DSM_name + '_' + str(FIRE_RANGE) + '.pkl')
    with open(DICT_POS_FIRE_RANGE_path, 'rb') as f:
        DICT_POS_FIRE_RANGE = pickle.load(f)
except:
    print("did not load DICT_POS_FIRE_RANGE")

try:
    DICT_POS_FIRE_RANGE_tuple_path = path.join(COMMON_PATH, 'Preprocessing', 'dictionary_position_los_'+DSM_name+'_'+str(FIRE_RANGE)+ '_tuple.pkl')
    with open(DICT_POS_FIRE_RANGE_tuple_path, 'rb') as f:
        DICT_POS_FIRE_RANGE_TUPLE = pickle.load(f)
except:
    print("did not load DICT_POS_FIRE_RANGE_tuple")

try:
    #turning DICT_POS_FIRE_RANGE to hold sets
    for k in DICT_POS_FIRE_RANGE:
        DICT_POS_FIRE_RANGE[k] = set(DICT_POS_FIRE_RANGE[k])
except:
    print("error in 'turning DICT_POS_FIRE_RANGE to hold sets'")

try:
    DICT_POS_LOS_path = path.join(COMMON_PATH, 'Preprocessing', 'dictionary_position_los_' + DSM_name + '_' + str(
            LOS_PENALTY_RANGE) + '.pkl')
    with open(DICT_POS_LOS_path, 'rb') as f:
        DICT_POS_LOS = pickle.load(f)
except:
    print("did not load DICT_POS_LOS")

try:
    DICT_POS_LOS_TUPLE_path = path.join(COMMON_PATH, 'Preprocessing', 'dictionary_position_los_' + DSM_name+ '_'+str(LOS_PENALTY_RANGE)+ '_tuple.pkl')
    with open(DICT_POS_LOS_TUPLE_path, 'rb') as f:
        DICT_POS_LOS_TUPLE = pickle.load(f)
except:
    print("did not load DICT_POS_LOS_TUPLE")



MOVE_PENALTY = -0.01
WIN_REWARD = 1
LOST_PENALTY = -1
ENEMY_LOS_PENALTY = MOVE_PENALTY * 2
TIE = 0


class WinEnum(IntEnum):
    Blue = 0
    Red = 1
    Tie = 2
    NoWin = 3

NUMBER_OF_ACTIONS = 9

BLUE_N = 1 #blue player key in dict
DARK_BLUE_N = 2
RED_N = 3 #red player key in dict
DARK_RED_N = 4
DARK_DARK_RED_N = 12
PURPLE_N = 5
YELLOW_N = 6 #to be used for line from blue to red
GREY_N = 7 #obstacle key in dict
GREEN_N = 8
BLACK_N = 9
BRIGHT_RED = 10
BRIGHT_BRIGHT_RED = 11

#### Graphics ####
# for better separation of colors for state
dict_of_colors_for_state = {1: (0, 0, 255),  #blue
                  2: (0, 0, 175), #darker blue
                  3: (255, 0, 0), # red
                  4: (175, 0, 0), #dark red
                  5: (230, 100, 150), #purple
                  6: (60, 255, 255), #yellow
                  7: (100, 100, 100),#grey
                  8: (0, 255, 0),#green
                  9: (0, 0, 0), #black
                  10: (0, 0, 75), #bright red
                  11: (0, 0, 25), #bright bright red
                  12: (50, 0, 0), #dark dark red
                  }

# color for graphics
dict_of_colors_for_graphics = {1: (0, 0, 239),  #blue
                               2: (0, 0, 175),  #darker blue
                               3: (239, 0, 0),  # red
                               4: (175, 20, 0),  #dark red
                               5: (150, 100, 230),  #purple
                               6: (255, 255, 60),  #yellow
                               7: (100, 100, 100),  #grey
                               8: (0, 239, 0),  #green
                               9: (0, 0, 0),  #black
                               10: (100, 0, 0),  #bright red
                               11: (50, 0, 0),  #bright bright red
                               12: (75, 0, 0), #dark dark red
                               }

# color for graphics
dict_of_colors_for_graphics_cv2 = {1: (239, 0, 0),  #blue
                               2: (175, 0, 0),  #darker blue
                               3: (0, 0, 239),  # red
                               4: (0, 20, 175),  #dark red
                               5: (230, 100, 150),  #purple
                               6: (60, 255, 255),  #yellow
                               7: (100, 100, 100),  #grey
                               8: (0, 239, 0),  #green
                               9: (0, 0, 0),  #black
                               10: (0, 0, 100),  #bright red
                               11: (0, 0, 50),  #bright bright red
                               12: (0, 0, 75), #dark dark red
                               }

#dict_of_colors_for_graphics = dict_of_colors_for_graphics_cv2



if ACTION_SPACE_9:
    NUMBER_OF_ACTIONS = 9
    class AgentAction(IntEnum):
        TopRight = 1
        Right = 2
        BottomRight = 3
        Bottom = 4
        Stay = 5
        Top = 6
        BottomLeft = 7
        Left = 8
        TopLeft = 0


class AgentType(IntEnum):
    Q_table = 1
    DQN_basic = 2
    DQN_keras = 3
    Greedy = 4
    Smart = 5

Agent_type_str = {AgentType.Greedy : "Greedy_player",
                  AgentType.Smart : "smart_player"}

class Color(IntEnum):
    Blue = 1
    Red = 2

# params to evaluate trained models
EVALUATE_SHOW_EVERY = 1
EVALUATE_NUM_OF_EPISODES = 100
EVALUATE_SAVE_STATS_EVERY = 1000

EVALUATE_PLAYERS_EVERY = 1000
EVALUATE_BATCH_SIZE=100

#save information
USE_DISPLAY = True # change to False if training in cloud
SHOW_EVERY = 1
NUM_OF_EPISODES = 3_000_000+EVALUATE_BATCH_SIZE
SAVE_STATS_EVERY = 10000+EVALUATE_BATCH_SIZE

# training mode
IS_TRAINING = True

RED_TYPE = AgentType.Smart # AgentType.Greedy