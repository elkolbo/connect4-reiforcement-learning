import tensorflow as tf
import numpy as np
import random
import pathlib
import pygame

import os
from datetime import datetime
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Concatenate

path = pathlib.Path(__file__).parent / "logs"

# Set up TensorBoard writer
log_parent_dir = path
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(log_parent_dir, current_time)

# Create the log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Rest of your code remains unchanged
summary_writer = tf.summary.create_file_writer(log_dir)

# Create TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, write_graph=True, update_freq="epoch"
)

# Initialize episode_losses list
episode_losses = []
num_episodes = 10000
visualization_frequency = 1

# visualization constants
# Constants
WIDTH, HEIGHT = 7, 6
CELL_SIZE = 100
WINDOW_WIDTH, WINDOW_HEIGHT = WIDTH * CELL_SIZE, (HEIGHT + 2.5) * CELL_SIZE

FPS = 30

BACKGROUND_COLOR = (25, 25, 25)  # Dark background color
GRID_COLOR = (100, 100, 100)  # Grid color
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

# intit pygame to visualize
pygame.init()

screen = pygame.display.set_mode(
    (WINDOW_WIDTH, WINDOW_HEIGHT)
)  # Angepasste Fenstergröße
pygame.display.set_caption("Connect 4")

clock = pygame.time.Clock()


def draw_board(screen, board):
    for col in range(WIDTH):
        for row in range(HEIGHT):
            pygame.draw.rect(
                screen,
                BACKGROUND_COLOR,
                (
                    col * CELL_SIZE,
                    (row + 1.5) * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ),  # Angepasste Y-Koordinate
            )
            pygame.draw.circle(
                screen,
                GRID_COLOR,
                (
                    col * CELL_SIZE + CELL_SIZE // 2,
                    (row + 1.5) * CELL_SIZE + CELL_SIZE // 2,  # Angepasste Y-Koordinate
                ),
                CELL_SIZE // 2,
                5,
            )  # Draw grid circles
            if board[0][row][col] == 1:
                pygame.draw.circle(
                    screen,
                    RED,
                    (
                        col * CELL_SIZE + CELL_SIZE // 2,
                        (row + 1.5) * CELL_SIZE
                        + CELL_SIZE // 2,  # Angepasste Y-Koordinate
                    ),
                    CELL_SIZE // 2 - 5,
                )
            elif board[1][row][col] == 1:
                pygame.draw.circle(
                    screen,
                    BLUE,
                    (
                        col * CELL_SIZE + CELL_SIZE // 2,
                        (row + 1.5) * CELL_SIZE
                        + CELL_SIZE // 2,  # Angepasste Y-Koordinate
                    ),
                    CELL_SIZE // 2 - 5,
                )


def visualize_training(screen, q_values, random_action, action, reward):
    q_values = q_values[0]
    for col, value in enumerate(q_values):
        # Draw the bar for each column
        if value.numpy() == np.max(q_values.numpy()):
            color = (255, 0, 0)
        else:
            color = (10, 10, 255)
        pygame.draw.rect(
            screen,
            color,
            (
                col * CELL_SIZE + 0.25 * CELL_SIZE,
                WINDOW_HEIGHT - int(value.numpy() * 200),
                CELL_SIZE * 0.5,
                int(value.numpy() * 200),
            ),
        )

        # Display Q-values on the bars with 1 digit after the comma
        font_q_values = pygame.font.Font(None, 20)
        q_value_text = font_q_values.render(f"{value:.3f}", True, (255, 255, 255))
        screen.blit(q_value_text, (col * CELL_SIZE, WINDOW_HEIGHT - 20))
    # display reward of state
    font_reward = pygame.font.Font(None, 36)
    reward_text = font_reward.render(f"Reward:{reward:.3f}", True, (255, 255, 255))
    screen.blit(reward_text, (screen.get_width() - reward_text.get_width(), 40))

    # Display information about the chosen action in the top right corner
    font_action = pygame.font.Font(None, 36)
    action_text = font_action.render(
        f"Chosen Action: {action}{' (Random)' if random_action else ''}",
        True,
        (255, 255, 255),
    )
    screen.blit(action_text, (screen.get_width() - action_text.get_width(), 0))


# Constants
NUM_ACTIONS = WIDTH
STATE_SHAPE = (
    2,
    HEIGHT,
    WIDTH,
)  # 2 channels for current player, opponent


# Deep Q Network (DQN) Model
class DQN(tf.keras.Model):
    def __init__(self, num_actions=WIDTH):
        super(DQN, self).__init__()
        # Layers for processing the game field
        self.conv1 = Conv2D(
            32, (3, 3), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv2 = Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv3 = Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", activation="relu"
        )
        self.flatten = Flatten()

        # Layers for processing flags
        self.flag_fc = Dense(64, activation="relu")  # Adjust the size as needed
        self.flag_output = Dense(1, activation="sigmoid")

        # Common dense layers
        self.fc1 = Dense(256, activation="relu")
        self.fc2 = Dense(num_actions)

    def call(self, inputs):
        # Separate game field and flags
        game_field = inputs[0]
        flags = inputs[1]

        # Process game field
        x = self.conv1(game_field)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        # Process flags separately
        flags_output = self.flag_fc(flags)
        flags_output = self.flag_output(flags_output)

        # Concatenate the processed game field and flags
        x = tf.concat([x, flags_output], axis=-1)

        # Common dense layers
        output = self.fc2(x)
        output.set_shape((None, NUM_ACTIONS))  # Adjust NUM_ACTIONS as needed
        return output

    def set_custom_weights(self, weights):
        # Set custom weights for each layer
        self.conv1.set_weights(weights[0:2])
        self.conv2.set_weights(weights[2:4])
        self.conv3.set_weights(weights[4:6])
        self.fc1.set_weights(weights[6:8])
        self.fc2.set_weights(weights[8:10])


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(
        self,
        state,
        action,
        next_state,
        reward,
        game_terminated_flag,
        opponent_won_flag,
        agent_won_flag,
        illegal_agent_move_flag,
        board_full_flag,
    ):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (
            state,
            action,
            next_state,
            reward,
            game_terminated_flag,
            opponent_won_flag,
            agent_won_flag,
            illegal_agent_move_flag,
            board_full_flag,
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


# Epsilon-Greedy Exploration
def epsilon_greedy_action(state, epsilon, model):
    q_values = model([state, np.expand_dims(np.zeros(5), axis=0)])
    if np.random.rand() < epsilon:
        random_move = True
        return np.random.randint(NUM_ACTIONS), q_values, random_move  # Explore
    else:
        random_move = False
        return np.argmax(q_values), q_values, random_move  # Exploit


# Function to check for a winning move


def check_win(board):
    players, rows, cols = board.shape

    # Check for a win in rows
    for player in range(players):
        for row in range(rows):
            for col in range(cols - 3):
                if np.all(board[player, row, col : col + 4] == 1):
                    return True

    # Check for a win in columns
    for player in range(players):
        for col in range(cols):
            for row in range(rows - 3):
                if np.all(board[player, row : row + 4, col] == 1):
                    return True

    # Check for a win in diagonals (from bottom-left to top-right)
    for player in range(players):
        for row in range(3, rows):
            for col in range(cols - 3):
                if np.all(board[player, row - np.arange(4), col + np.arange(4)] == 1):
                    return True

    # Check for a win in diagonals (from top-left to bottom-right)
    for player in range(players):
        for row in range(rows - 3):
            for col in range(cols - 3):
                if np.all(board[player, row + np.arange(4), col + np.arange(4)] == 1):
                    return True

    return False


def next_empty_row(board, action):
    try:
        for row in range(HEIGHT):
            if board[0, row, action] == 0 and board[1, row, action] == 0:
                next = row
                break
            else:
                continue
        return next
    except:
        pass


# Function to calculate the reward
def calculate_reward(board, action, current_player):
    # Default reward
    reward = 0

    # check if board has free spaces (not necessary but doesnt hurt)
    if np.sum(board) < HEIGHT * WIDTH:
        # Check if the column is full
        if np.sum(board[:, :, action]) == HEIGHT:
            reward -= 10  # Give a penalty for placing a disc in a full column
        else:
            # Check if placing a disc prevents the opponent from connecting four
            # if is_blocking_opponent(board, action):
            #     reward += 10  # Give a significant reward for blocking the opponent

            # Check if placing a disc next to many of your own
            adjacent_count = count_adjacent_discs(board, action, empty_row)
            reward += 0.1 * adjacent_count  # Increase reward based on the count

            # Reward for a valid move
            reward += 1  # Give a small reward for a valid move

    else:
        print("BOARD IS FULL!!")

    return reward


def is_blocking_opponent(board, action_column):
    return False


def count_adjacent_discs(board, action_column, action_row):
    # check surrounings and count discs
    count = 0
    # go around disc with catching errors
    for row_offset in [-1, 0, 1]:
        for column_offset in [-1, 0, 1]:
            try:
                if (
                    board[0, action_row + row_offset, action_column + column_offset]
                    == 1
                ):
                    count += 1
            except:
                pass
    return count


# Function to train the opponent
def train_opponent(opponent, opponent_model, epsilon, state):
    if opponent == "rand":
        action = np.random.randint(NUM_ACTIONS)
    elif opponent == "self":
        # flip the current state so the opponent sees his situation on the top layer!!
        # otherwise the rl opponent will predict action based on rl agents position
        state_copy = state.copy()
        state_copy = np.flip(state_copy, axis=1)
        action, _, _ = epsilon_greedy_action(state_copy, epsilon, opponent_model)
    # Add more opponent strategies as needed
    return action


# Function to initialize the models
def model_init(train_from_start):
    learning_rate = 0.0005
    gamma = 0.9
    epsilon_start = 0.0
    epsilon_end = 0.0
    epsilon_decay = 0.9999
    target_update_frequency = 10
    batch_size = 64
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    replay_buffer = ReplayBuffer(capacity=10000)

    if train_from_start:
        model = DQN(num_actions=NUM_ACTIONS)
        model.build([(None, 2, HEIGHT, WIDTH), (None, 5)])
        model.compile(optimizer="adam", loss="mse")
    else:
        model = DQN(num_actions=NUM_ACTIONS)
        model.build([(None, 2, HEIGHT, WIDTH), (None, 5)])
        model.compile(optimizer="adam", loss="mse")

        model.load_weights("./checkpoints/my_checkpoint")

    # Inside model_init() function
    opponent_model = DQN(num_actions=NUM_ACTIONS)
    opponent_model.build([(None, 2, HEIGHT, WIDTH), (None, 5)])

    return (
        model,
        opponent_model,
        replay_buffer,
        optimizer,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        target_update_frequency,
        batch_size,
    )


# Function to get RL action when playing game ->used in other code
def get_rl_action(board, model):
    state = board_to_numpy(board, 2)
    q_values = model(
        [state, np.expand_dims(np.zeros(5), axis=0)]
    )  # all the flags are 0s
    return np.argmax(q_values)


# Convert Connect 4 board to NumPy array and make input indifferent
def board_to_numpy(board, current_player):
    array = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32)
    array[:, :, 0] = board == current_player  # Current player's discs
    array[:, :, 1] = board == 3 - current_player  # Opponent's discs
    return array.transpose((2, 0, 1))[np.newaxis, :]  # Add batch dimension


def numpy_to_board(array, current_player):
    # Transpose back to the original shape
    array = array.transpose((1, 0, 2))

    # Extract current player's discs and opponent's discs
    current_player_discs = np.where(array[:, :, 0] == 1)
    opponent_discs = np.where(array[:, :, 0] == 0)

    # Create an empty board
    board = np.zeros((HEIGHT, WIDTH))

    # Fill in the board with player and opponent discs
    board[current_player_discs[0], current_player_discs[1]] = current_player
    board[opponent_discs[0], opponent_discs[1]] = 3 - current_player

    return board


if __name__ == "__main__":
    train_from_start = False

    (
        model,
        opponent_model,
        replay_buffer,
        optimizer,
        gamma,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        target_update_frequency,
        batch_size,
    ) = model_init(train_from_start)

    max_steps_per_episode = (
        44  # higher than possible steps to enforce full board break to step in
    )

    for episode in range(1, num_episodes + 1):
        print("New episode starting:")
        print("*" * 50)
        state = np.zeros((1, 2, HEIGHT, WIDTH), dtype=np.float32)
        done = False
        step = 0
        epsilon = max(epsilon_end, epsilon_start * epsilon_decay**episode)
        game_ended = False
        pygame.event.pump()
        while not done and step < max_steps_per_episode and not game_ended:
            # move of the RL agent
            action, q_values, random_move = epsilon_greedy_action(state, epsilon, model)
            # check if move is legal
            if (
                state[0, :, :, action].sum() < HEIGHT and not game_ended
            ):  # agent makes legal move and game has not ended
                # calculate next state and reward of this legal move
                # passing on without batch dimension
                empty_row = next_empty_row(state[0], action)
                next_state = state.copy()
                next_state[
                    0, 0, empty_row, action
                ] = 1  # updation state, channel 0 is always for agent
                reward = calculate_reward(next_state[0], action, current_player=1)
                # agent made legal move, now check the outcome of the move:

                if check_win(next_state[0]):  # agent won
                    print("EPISODE ENDED BY WIN OF AGENT")
                    print(next_state[0])
                    print("#" * 30)
                    reward = 1000
                    replay_buffer.push(
                        state,
                        action,
                        next_state,
                        reward,
                        game_terminated_flag=1,
                        opponent_won_flag=0,
                        agent_won_flag=1,
                        illegal_agent_move_flag=0,
                        board_full_flag=0,
                    )

                    game_ended = True

                else:  # move was a "normal" game move, game continues
                    # calculate opponennts move
                    opponent_action = train_opponent(
                        "self", opponent_model, epsilon, next_state
                    )
                    next_state_opponent = next_state.copy()
                    # opponent can be "rand" or "self"
                    if next_state[0, :, :, opponent_action].sum() < HEIGHT:
                        empty_row = next_empty_row(next_state[0], opponent_action)
                        next_state_opponent[0, 1, empty_row, opponent_action] = 1
                    else:
                        # opponent chose an illegal move -> picking free column instead
                        for column in range(WIDTH):
                            if next_state[0, :, :, column].sum() < HEIGHT:
                                opponent_action = column
                                break
                        empty_row = next_empty_row(next_state[0], opponent_action)
                        next_state_opponent[0, 1, empty_row, opponent_action] = 1

                    if check_win(next_state_opponent[0]):  # opponent won
                        print("EPISODE ENDED BY WIN OF OPPONENT")
                        print(next_state_opponent[0])
                        print("#" * 30)
                        reward = -100
                        replay_buffer.push(
                            state,
                            action,
                            next_state,
                            reward,
                            game_terminated_flag=1,
                            opponent_won_flag=1,
                            agent_won_flag=0,
                            illegal_agent_move_flag=0,
                            board_full_flag=0,
                        )
                        game_ended = True

                    else:  # opponent didn't win
                        replay_buffer.push(
                            state,
                            action,
                            next_state,
                            reward,
                            game_terminated_flag=0,
                            opponent_won_flag=0,
                            agent_won_flag=0,
                            illegal_agent_move_flag=0,
                            board_full_flag=0,
                        )
                    next_state = (
                        next_state_opponent.copy()
                    )  # copy for correct continuation in next episode
            elif (
                not np.sum(next_state[0]) < HEIGHT * WIDTH and not game_ended
            ):  # check if board is full --> reason for illegal move
                print("EPISODE ENDED BY FULL BOARD")
                reward = -5
                replay_buffer.push(
                    state,
                    action,
                    next_state,
                    reward,
                    game_terminated_flag=1,
                    opponent_won_flag=0,
                    agent_won_flag=0,
                    illegal_agent_move_flag=0,
                    board_full_flag=1,
                )
                game_ended = True
            elif not game_ended:  # agent makes illegal move
                reward = -50
                next_state = state.copy()
                replay_buffer.push(
                    state,
                    action,
                    next_state,
                    reward,
                    game_terminated_flag=1,
                    opponent_won_flag=0,
                    agent_won_flag=0,
                    illegal_agent_move_flag=1,
                    board_full_flag=0,
                )
                print("Episode ended by agent illegal move")
                game_ended = True

            # set next state as state for next step of episode
            state = next_state.copy()

            if len(replay_buffer.memory) > batch_size:
                batch = replay_buffer.sample(batch_size)
                (
                    states,
                    actions,
                    next_states,
                    rewards,
                    game_terminated_flags,
                    opponent_won_flags,
                    agent_won_flags,
                    illegal_agent_move_flags,
                    board_full_flags,
                ) = zip(*batch)

                states = np.concatenate(states)
                actions = np.array(actions, dtype=np.int32).reshape(-1, 1)
                next_states = np.concatenate(next_states)
                rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)
                game_terminated_flags = np.array(
                    game_terminated_flags, dtype=np.float32
                )
                opponent_won_flags = np.array(opponent_won_flags, dtype=np.float32)
                agent_won_flags = np.array(agent_won_flags, dtype=np.float32)
                illegal_agent_move_flags = np.array(
                    illegal_agent_move_flags, dtype=np.float32
                )
                board_full_flags = np.array(board_full_flags, dtype=np.float32)

                with tf.GradientTape() as tape:
                    flags = np.column_stack(
                        [
                            game_terminated_flags,
                            opponent_won_flags,
                            agent_won_flags,
                            illegal_agent_move_flags,
                            board_full_flags,
                        ],
                    )

                    current_q_values = model([states, flags])
                    current_q_values = tf.reduce_sum(
                        tf.one_hot(actions, NUM_ACTIONS) * current_q_values,
                        axis=1,
                        keepdims=True,
                    )

                    next_q_values = model([next_states, flags])
                    next_q_values = tf.reduce_max(next_q_values, axis=1, keepdims=True)

                    target_q_values = rewards + gamma * next_q_values

                    loss = tf.reduce_mean(tf.square(current_q_values - target_q_values))

                    # Append loss to the list for visualization
                    episode_losses.append(loss.numpy())

                    # Log loss to TensorBoard
                    with summary_writer.as_default():
                        tf.summary.scalar("Loss", loss.numpy(), step=episode)
                        tf.summary.scalar("Epsilon", epsilon, step=episode)

                gradients = tape.gradient(loss, model.trainable_variables)
                # Clip gradients to stabilize training
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 2)
                max_gradient = tf.reduce_max(
                    [tf.reduce_max(grad) for grad in gradients]
                )
                max_clipped_gradient = tf.reduce_max(
                    [tf.reduce_max(grad) for grad in clipped_gradients]
                )

                # print(f"Unclipped garadients max:{max_gradient}")
                # print(f"Clipped garadients max:{max_clipped_gradient}")
                optimizer.apply_gradients(
                    zip(clipped_gradients, model.trainable_variables)
                )

            step += 1

            if episode % visualization_frequency == 0:
                screen.fill(BACKGROUND_COLOR)  # Clear the screen
                draw_board(screen, state[0])
                visualize_training(screen, q_values, random_move, action, reward)
                pygame.display.flip()
                pygame.display.update()  # forced display update
                pygame.time.wait(2000)
                # wait for a bit to make it easier to follow visualization

            # Inside the training loop
        if episode % target_update_frequency == 0:
            model_weights = model.get_weights()
            opponent_model_weights = opponent_model.get_weights()

            # Print the shapes of the weights to identify any mismatches
            for w1, w2 in zip(model_weights, opponent_model_weights):
                pass
                # print(w1.shape, w2.shape)

            opponent_model.set_weights(model_weights)

        if episode % 10 == 0:
            print(f"Episode: {episode}, Epsilon: {epsilon:.3f}")

    print("Training complete.")

    # Save the weights
    model.save_weights("./checkpoints/my_checkpoint")


# Close the writer
summary_writer.close()


# You can launch TensorBoard by running this command in the terminal
# tensorboard --logdir=path/to/log/directory
