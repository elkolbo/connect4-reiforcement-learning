from tensorflow.keras.optimizers.schedules import ExponentialDecay


class config:
    def __init__(self):
        #################################
        ##### Adjust variables here #####
        #################################

        self.num_episodes = 300000
        self.batch_size = 128
        self.learning_rate = ExponentialDecay(
            initial_learning_rate=0.0005,
            decay_steps=self.num_episodes // 60,
            decay_rate=0.9,
            staircase=True,
        )

        self.train_from_start = False

        self.target_update_frequency = 250
        self.opponent_model_update_frequency = 999

        self.replay_buffer_capacity = 50000

        self.gamma = 0.95  # Discount factor for q-learning

        #### epsilon scheduler ###
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.reach_target_epsilon = (
            0.65  # percentage of episodes when to reach target epsilon
        )
        self.epsilon_mode = (
            "linear"  # alternative: linear; determines model used to calculate epsilon
        )

        self.visualization_frequency = 20000  # Put in a high value to train faster

        ##opponent settins
        self.opponent_switch_interval = 27
        self.min_self_play_prob = 0.1  # Initial probability of self-play
        self.max_self_play_prob = 0.9  # Max probability of self-play

        # Visualization constants
        self.WIDTH, self.HEIGHT = 7, 6
        self.CELL_SIZE = 100
        self.WINDOW_WIDTH, self.WINDOW_HEIGHT = (
            int(
                self.WIDTH * self.CELL_SIZE + 2 * self.CELL_SIZE
            ),  # Width of the window
            int((self.HEIGHT + 2.5) * self.CELL_SIZE),  # Hight of the window
        )

        self.FPS = 30

        self.BACKGROUND_COLOR = (25, 25, 25)  # Dark background color
        self.GRID_COLOR = (100, 100, 100)  # Grid color
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.BLUE = (0, 0, 255)
        self.NUM_ACTIONS = self.WIDTH
        self.STATE_SHAPE = (2, self.HEIGHT, self.WIDTH)

        # Define constants for players
        self.HUMAN_PLAYER = 1
        self.RL_PLAYER = 2
