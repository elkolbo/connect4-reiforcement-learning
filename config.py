from tensorflow.keras.optimizers.schedules import ExponentialDecay


class config:
    def __init__(self):
        #################################
        ##### Adjust variables here #####
        #################################

        self.num_episodes = 200000
        self.batch_size = 128
        self.learning_rate = ExponentialDecay(
            initial_learning_rate=0.001, decay_steps=self.num_episodes, decay_rate=0.999
        )

        self.train_from_start = True

        self.target_update_frequency = 100

        self.replay_buffer_capacity = 25000

        self.gamma = 0.95  # Discount factor for q-learning

        #### epsilon scheduler ###
        self.epsilon_start = 0.95
        self.epsilon_end = 0.05
        self.reach_target_epsilon = (
            0.7  # percentage of episodes when to reach target epsilon
        )
        self.epsilon_mode = "quadratic"  # alternative: linear; determines model used to calculate epsilon

        self.visualization_frequency = 100000  # Put in a high value to train faster

        self.opponent_switch_interval = 501

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
