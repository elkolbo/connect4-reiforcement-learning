class config:
    def __init__(self):
        
        ###########################
        ##### Agent variables #####
        ###########################

        self.num_episodes = 10000
        self.learning_rate = 0.0005

        self.gamma = 0.9  # discount factor for q-learning

        self.train_from_start = False
        
        self.epsilon_start = 0.5
        self.epsilon_end = 0.0
        self.epsilon_decay = 0.9999
        self.target_update_frequency = 10
        self.batch_size = 64

        self.visualization_frequency = 1  # Put in a high value to train faster

        self.opponent_switch_interval = 5

        # visualization constants
        self.WIDTH, self.HEIGHT = 7, 6
        self.CELL_SIZE = 100
        self.WINDOW_WIDTH, self.WINDOW_HEIGHT = (
            self.WIDTH * self.CELL_SIZE,
            (self.HEIGHT + 2.5) * self.CELL_SIZE,
        )
        self.FPS = 30
        self.BACKGROUND_COLOR = (25, 25, 25)  # Dark background color
        self.GRID_COLOR = (100, 100, 100)  # Grid color
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.BLUE = (0, 0, 255)
        self.NUM_ACTIONS = self.WIDTH
        self.STATE_SHAPE = (
            2,
            self.HEIGHT,
            self.WIDTH
        )


        ###########################
        ##### Game variables ######
        ###########################

        # Constants
        WIDTH, HEIGHT = 7, 6
        CELL_SIZE = 100
        # WINDOW_WIDTH, WINDOW_HEIGHT = WIDTH * CELL_SIZE + 2, (HEIGHT + 2.5) * CELL_SIZE +2
        WINDOW_WIDTH, WINDOW_HEIGHT = (
            WIDTH * CELL_SIZE + 2 * CELL_SIZE,
            (HEIGHT + 2.5) * CELL_SIZE + 2,
        )

        FPS = 30

        # # Colors
        # BACKGROUND_COLOR = (200, 200, 200)  # Neutral background color
        # GRID_COLOR = (0, 0, 0)  # Grid color
        # RED = (255, 0, 0)
        # YELLOW = (255, 255, 0)
        # BLUE = (0, 0, 255)

        BACKGROUND_COLOR = (25, 25, 25)  # Dark background color
        GRID_COLOR = (100, 100, 100)  # Grid color
        RED = (255, 0, 0)
        YELLOW = (255, 255, 0)
        BLUE = (0, 0, 255)
        GLOW_GREEN = (0, 255, 0)

        # Define constants for players
        HUMAN_PLAYER = 1
        RL_PLAYER = 2