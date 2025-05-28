import tensorflow as tf
import numpy as np
import random
import collections
import pathlib
import pygame
import pandas as pd

import os
from datetime import datetime
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Concatenate
from config import config

config_values = config()


# Deep Q Network (DQN) Model
class DQN(tf.keras.Model):
    def __init__(self, num_actions=config_values.WIDTH):
        super(DQN, self).__init__()
        # Layers for processing the game field
        self.conv1 = Conv2D(
            32, (4, 4), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv2 = Conv2D(
            64, (4, 4), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv3 = Conv2D(
            64, (4, 4), strides=(1, 1), padding="same", activation="relu"
        )
        self.flatten = Flatten()  # Flatten the output of the convolutional layers

        # Common dense layers
        self.fc1 = Dense(256, activation="relu")
        self.fc2 = Dense(num_actions)

    def call(self, inputs):
        game_field = inputs
        # Process game field
        x = self.conv1(game_field)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)  # Flatten after convolutions

        # Common dense layers
        x = self.fc1(x)
        output = self.fc2(x)
        output.set_shape(
            (None, self.fc2.units)
        )  # Ensure shape consistency using layer's units
        return output

    def set_custom_weights(self, weights):
        # Set custom weights for each layer
        self.conv1.set_weights(weights[0:2])
        self.conv2.set_weights(weights[2:4])
        self.conv3.set_weights(weights[4:6])
        self.fc1.set_weights(weights[6:8])
        self.fc2.set_weights(weights[8:10])


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(
            maxlen=capacity
        )  # Store priorities separately
        self.epsilon_priority = (
            1e-6  # Small constant to ensure all experiences have a non-zero probability
        )
        self.alpha = 0.6  # Prioritization exponent
        self.beta_start = 0.4  # Initial value for beta (importance sampling exponent)
        self.beta_frames = 100000  # Number of frames over which to anneal beta to 1.0
        self.frame = 1  # Current frame counter for beta annealing

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
        priority,  # Initial priority for the new experience
    ):
        """Saves an experience to memory and assigns an initial priority."""
        experience = (
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
        self.memory.append(experience)

        # Set initial priority: either the max priority in the buffer or the passed 'priority'
        max_priority = np.max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

    def _get_beta(self):
        """Anneals beta from beta_start to 1.0 over beta_frames."""
        fraction = min(self.frame / self.beta_frames, 1.0)
        beta = self.beta_start + fraction * (1.0 - self.beta_start)
        self.frame += 1
        return beta

    def sample(self, batch_size):
        """Samples a batch of experiences from memory based on priorities."""
        if len(self.memory) < batch_size:
            raise ValueError("Not enough samples in the replay buffer")

        # Calculate probabilities: P(i) = p_i^alpha / sum(p_j^alpha)
        priorities_array = np.array(self.priorities)
        scaled_priorities = np.power(priorities_array, self.alpha)
        probabilities = scaled_priorities / np.sum(scaled_priorities)

        # Sample indices based on probabilities
        indices = np.random.choice(
            len(self.memory), batch_size, replace=False, p=probabilities
        )

        samples = [self.memory[i] for i in indices]

        # Calculate Importance Sampling (IS) weights: w_i = (N * P(i))^-beta
        beta = self._get_beta()
        weights = np.power(len(self.memory) * probabilities[indices], -beta)
        weights /= np.max(weights)  # Normalize weights for stability

        # Unzip the samples for RL_agent.py
        # Each sample is (state, action, next_state, reward, game_terminated_flag, ...)
        # RL_agent.py expects: (indices, states, actions, next_states, rewards, game_terminated_flags, ..., IS_weights)
        # The 'indices' here are the indices into the deque for updating priorities.

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
        ) = zip(*samples)

        batch_for_zipping = []
        for i in range(batch_size):
            (
                state,
                action,
                next_state,
                reward,
                game_terminated_flag,
                opponent_won_flag,
                agent_won_flag,
                illegal_agent_move_flag,
                board_full_flag,
            ) = samples[i]
            batch_for_zipping.append(
                (
                    indices[i],
                    state,
                    action,
                    next_state,
                    reward,
                    game_terminated_flag,
                    opponent_won_flag,
                    agent_won_flag,
                    illegal_agent_move_flag,
                    board_full_flag,
                    weights[i],
                )
            )

        return batch_for_zipping

    def update_priorities(self, indices, td_errors):
        """Updates the priorities of sampled experiences."""
        priorities = np.abs(td_errors) + self.epsilon_priority
        for idx, priority in zip(indices, priorities):
            if hasattr(priority, "numpy"):
                self.priorities[idx] = priority.numpy().item()
            else:
                self.priorities[idx] = float(
                    priority
                )  # Stelle sicher, dass es ein Float ist


def draw_board(screen, board):
    for col in range(config_values.WIDTH):
        for row in range(config_values.HEIGHT):
            pygame.draw.rect(
                screen,
                config_values.BACKGROUND_COLOR,
                (
                    col * config_values.CELL_SIZE,
                    (row + 1.5) * config_values.CELL_SIZE,
                    config_values.CELL_SIZE,
                    config_values.CELL_SIZE,
                ),  # Angepasste Y-Koordinate
            )
            pygame.draw.circle(
                screen,
                config_values.GRID_COLOR,
                (
                    col * config_values.CELL_SIZE + config_values.CELL_SIZE // 2,
                    (row + 1.5) * config_values.CELL_SIZE
                    + config_values.CELL_SIZE // 2,  # Angepasste Y-Koordinate
                ),
                config_values.CELL_SIZE // 2,
                5,
            )  # Draw grid circles
            if board[0][row][col] == 1:
                pygame.draw.circle(
                    screen,
                    config_values.RED,
                    (
                        col * config_values.CELL_SIZE + config_values.CELL_SIZE // 2,
                        (row + 1.5) * config_values.CELL_SIZE
                        + config_values.CELL_SIZE // 2,  # Angepasste Y-Koordinate
                    ),
                    config_values.CELL_SIZE // 2 - 5,
                )
            elif board[1][row][col] == 1:
                pygame.draw.circle(
                    screen,
                    config_values.BLUE,
                    (
                        col * config_values.CELL_SIZE + config_values.CELL_SIZE // 2,
                        (row + 1.5) * config_values.CELL_SIZE
                        + config_values.CELL_SIZE // 2,  # Angepasste Y-Koordinate
                    ),
                    config_values.CELL_SIZE // 2 - 5,
                )


def visualize_training(screen, q_values, random_action, action, reward, opponent):
    q_values = q_values[0]
    for col, value in enumerate(q_values):
        # Draw the bar for each column
        if value.numpy() == np.max(q_values.numpy()):
            color = (255, 0, 0)
        else:
            color = (10, 10, 255)
        value_norm = (
            (value.numpy() / np.max(q_values.numpy())) * 1.5 * config_values.CELL_SIZE
        )
        pygame.draw.rect(
            screen,
            color,
            (
                col * config_values.CELL_SIZE + 0.25 * config_values.CELL_SIZE,  # x
                1.3 * config_values.CELL_SIZE - int(value_norm),  # y (move up)
                config_values.CELL_SIZE * 0.5,  # width
                int(value_norm),
            ),
        )

        # Display Q-values on the bars with 1 digit after the comma
        font_q_values = pygame.font.Font(None, 20)
        q_value_text = font_q_values.render(f"{value:.3f}", True, (255, 255, 255))
        screen.blit(
            q_value_text,
            (col * config_values.CELL_SIZE, 1.3 * config_values.CELL_SIZE),
        )
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

    # Display which opponent
    font_action = pygame.font.Font(None, 36)
    opponent_text = font_action.render(
        f"Opponent: " + opponent,
        True,
        (255, 255, 255),
    )
    screen.blit(opponent_text, (screen.get_width() - opponent_text.get_width(), 80))


# Constants
NUM_ACTIONS = config_values.WIDTH
STATE_SHAPE = (
    2,
    config_values.HEIGHT,
    config_values.WIDTH,
)  # 2 channels for current player, opponent


# Epsilon-Greedy Exploration
def epsilon_greedy_action(state, epsilon, model):
    q_values = model(state)
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


def is_blocking_opponent(board, action_column):
    # FIXME: This logic is incorrect. It checks if the opponent would win by playing in action_column.
    # Copy the board to simulate the effect of placing a disc in the specified column
    temp_board = np.copy(board)

    # Find the empty row in the specified column
    empty_row = next_empty_row(temp_board, action_column)

    if empty_row is None:
        # The column is full, and placing a disc is not possible
        return False

    # Place a disc in the specified column
    temp_board[1, empty_row, action_column] = 1

    # Check if this move blocks the opponent from connecting four discs
    return check_win(temp_board)


def next_empty_row(board, action):
    # Iterates from bottom to top, which is not how Connect4 drops work.
    # It should find the lowest available row.
    # The board state representation is (channel, row, col).
    # Channel 0: agent, Channel 1: opponent.
    # A cell is empty if board[0, row, action] == 0 AND board[1, row, action] == 0.
    # This function is called with state[0] from RL_agent.py, so board has shape (2, H, W)
    for r in range(config_values.HEIGHT - 1, -1, -1):  # Start from bottom row
        if board[0, r, action] == 0 and board[1, r, action] == 0:
            return r
    return None  # Column is full or error


# Function to calculate the reward
def calculate_reward(agent_board_plane_after_move, action_column, action_row):
    """
    Calculates intermediate rewards for the agent's move.
    Assumes agent_board_plane_after_move is the agent's plane AFTER the move (H, W).
    action_column and action_row are where the agent just placed its piece.
    """
    reward = 0

    # Small base reward for making a legal, non-terminal move
    reward += 0.5  # Tunable (e.g., 0.1 to 1.0)

    # Reward for connecting to existing friendly pieces
    adjacent_friendly_discs = count_adjacent_discs(
        agent_board_plane_after_move, action_column, action_row
    )
    reward += (
        0.2 * adjacent_friendly_discs
    )  # Tunable (e.g., 0.1 to 0.5 per adjacent disc)

    return reward


# def is_blocking_opponent(board, action_column):
#     return False


def count_adjacent_discs(agent_board_plane, action_column, action_row):
    """
    Counts friendly discs adjacent to the newly placed piece on the agent's board plane.
    - agent_board_plane: A (H, W) numpy array representing the agent's pieces (1s and 0s).
    - action_column: The column where the new piece was placed.
    - action_row: The row where the new piece was placed.
    """
    count = 0
    # Iterate over the 8 neighbors
    for r_offset in [-1, 0, 1]:
        for c_offset in [-1, 0, 1]:
            if r_offset == 0 and c_offset == 0:
                continue  # Don't count the piece itself

            check_row, check_col = action_row + r_offset, action_column + c_offset

            if (
                0 <= check_row < config_values.HEIGHT
                and 0 <= check_col < config_values.WIDTH
            ):
                if agent_board_plane[check_row, check_col] == 1:
                    count += 1
    return count


# Function to train the opponent
def train_opponent(opponent, opponent_model, epsilon, state, step):
    if opponent == "rand":
        action = np.random.randint(NUM_ACTIONS)
    elif opponent == "self":
        # flip the current state so the opponent sees his situation on the top layer!!
        # otherwise the rl opponent will predict action based on rl agents position
        state_copy = state.copy()
        state_copy = np.flip(state_copy, axis=1)
        action, _, _ = epsilon_greedy_action(state_copy, epsilon, opponent_model)
    elif opponent == "ascending_columns":
        # Opponent places discs in columns in ascending order
        action = step % NUM_ACTIONS
    # Add more opponent strategies as needed
    return action


# Function to initialize the models
def model_init(train_from_start):
    optimizer = tf.keras.optimizers.Adam(config_values.learning_rate)
    replay_buffer = ReplayBuffer(
        capacity=config_values.replay_buffer_capacity
    )  # Use config for capacity

    if train_from_start:
        model = DQN(num_actions=NUM_ACTIONS)
        model.build((None, 2, config_values.HEIGHT, config_values.WIDTH))
    else:
        model = DQN(num_actions=NUM_ACTIONS)
        model.build((None, 2, config_values.HEIGHT, config_values.WIDTH))

        model.load_weights("./checkpoints/my_checkpoint.h5")
    target_model = DQN(num_actions=NUM_ACTIONS)
    target_model.build((None, 2, config_values.HEIGHT, config_values.WIDTH))
    target_model.set_weights(model.get_weights())

    opponent_model = DQN(num_actions=NUM_ACTIONS)
    opponent_model.build((None, 2, config_values.HEIGHT, config_values.WIDTH))

    return (
        model,
        opponent_model,
        target_model,
        replay_buffer,
        optimizer,
    )


# Function to get RL action when playing game ->used in other code
def get_rl_action(board, model):
    state = board_to_numpy(board, 2)
    q_values = model(state)
    return np.argmax(q_values), q_values


# Convert Connect 4 board to NumPy array and make input indifferent
def board_to_numpy(board, current_player):
    array = np.zeros((config_values.HEIGHT, config_values.WIDTH, 2), dtype=np.float32)
    array[:, :, 0] = (np.array(board) == current_player) * 1  # Current player's discs
    array[:, :, 1] = (np.array(board) == 3 - current_player) * 1  # Opponent's discs
    return array.transpose((2, 0, 1))[np.newaxis, :]  # Add batch dimension


def numpy_to_board(array, current_player):
    # Transpose back to the original shape
    array = array.transpose((1, 0, 2))

    # Extract current player's discs and opponent's discs
    current_player_discs = np.where(array[:, :, 0] == 1)
    opponent_discs = np.where(array[:, :, 0] == 0)

    # Create an empty board
    board = np.zeros((config_values.HEIGHT, config_values.WIDTH))

    # Fill in the board with player and opponent discs
    board[current_player_discs[0], current_player_discs[1]] = current_player
    board[opponent_discs[0], opponent_discs[1]] = 3 - current_player

    return board


def choose_opponent(episode, opponent_switch_interval):
    if episode % opponent_switch_interval == 0:
        if np.random.rand() < 0.5:
            current_opponent = "rand"
        else:
            current_opponent = "ascending_columns"
    else:
        current_opponent = "self"
    print(f"Opponent: {current_opponent}")
    return current_opponent


class EpsilonScheduler:
    def __init__(
        self,
        epsilon_start: float,
        epsilon_end: float,
        num_episodes: int,
        reach_target_epsilon: float,
        mode: str = "linear",
    ):
        self.num_episodes = num_episodes
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.reach_target_epsilon = reach_target_epsilon

        if mode == "linear":
            self.epsilon_calculation = self.linear_model
        elif mode == "quadratic":
            self.epsilon_calculation = self.quadratic_model
        else:
            raise ValueError(f"Unknown epsilon mode: {mode}")

    def linear_model(self, episode: int):
        progress = min(
            (1.0 / self.reach_target_epsilon) * (episode / self.num_episodes), 1
        )
        epsilon = (1 - progress) * (
            self.epsilon_start - self.epsilon_end
        ) + self.epsilon_end
        return epsilon

    def quadratic_model(self, episode: int):
        progress = min(
            (1.0 / self.reach_target_epsilon) * (episode / self.num_episodes), 1
        )
        epsilon = (1 - progress**2) * (
            self.epsilon_start - self.epsilon_end
        ) + self.epsilon_end
        return epsilon

    def calculate_epsilon(self, episode: int):
        return self.epsilon_calculation(episode)
