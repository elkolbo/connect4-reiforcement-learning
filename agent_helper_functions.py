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
from tensorflow.keras import layers
from config import config

config_values = config()


# Deep Q Network (DQN) Model
class DQN(tf.keras.Model):
    def __init__(self, num_actions=config_values.WIDTH):  # Original DQN
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
        game_field = tf.expand_dims(inputs, -1)
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


# Dueling Deep Q Network (Dueling DQN) Model
class DuelingDQN(tf.keras.Model):
    def __init__(self, num_actions=config_values.WIDTH):
        super(DuelingDQN, self).__init__()
        # Common convolutional layers
        self.conv1 = Conv2D(
            32, (4, 4), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv2 = Conv2D(
            64, (4, 4), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv3 = Conv2D(
            64, (4, 4), strides=(1, 1), padding="same", activation="relu"
        )
        self.flatten = Flatten()

        # Value stream
        self.value_fc1 = Dense(128, activation="relu")
        self.value_fc2 = Dense(1)

        # Advantage stream
        self.advantage_fc1 = Dense(128, activation="relu")
        self.advantage_fc2 = Dense(num_actions)

    def call(self, inputs):
        # Inputs shape: (batch_size, HEIGHT, WIDTH)
        game_field = tf.expand_dims(
            inputs, -1
        )  # Add channel dimension: (batch_size, HEIGHT, WIDTH, 1)

        # Common layers
        x = self.conv1(game_field)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)  # Output shape: (batch_size, H*W*filters)

        # Value stream
        value = self.value_fc1(x)
        value = self.value_fc2(value)  # Output shape: (batch_size, 1)

        # Advantage stream
        advantage = self.advantage_fc1(x)
        advantage = self.advantage_fc2(
            advantage
        )  # Output shape: (batch_size, num_actions)

        # Combine streams to get Q-values
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        mean_advantage = tf.reduce_mean(
            advantage, axis=1, keepdims=True
        )  # Output shape: (batch_size, 1)
        q_values = value + (
            advantage - mean_advantage
        )  # Output shape: (batch_size, num_actions)

        return q_values


class SmallDuelingDQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(SmallDuelingDQN, self).__init__()

        self.conv = layers.Conv2D(
            filters=16, kernel_size=(4, 4), activation="relu", padding="same"
        )
        self.flatten = layers.Flatten()

        # Shared dense layer
        self.shared_dense = layers.Dense(64, activation="relu")

        # Value stream
        self.value_dense = layers.Dense(32, activation="relu")
        self.value_output = layers.Dense(1)

        # Advantage stream
        self.adv_dense = layers.Dense(32, activation="relu")
        self.adv_output = layers.Dense(num_actions)

    def call(self, inputs):
        game_field = tf.expand_dims(inputs, -1)
        x = self.conv(game_field)
        x = self.flatten(x)
        x = self.shared_dense(x)

        # Value stream
        v = self.value_dense(x)
        v = self.value_output(v)

        # Advantage stream
        a = self.adv_dense(x)
        a = self.adv_output(a)

        # Combine streams
        q = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
        return q


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
        self.alpha = 0.5  # Prioritization exponent
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

        # also add mirror state to the replay buffer
        experience_mirrored = (
            np.fliplr(state.copy()),
            (config_values.WIDTH - 1) - action,
            np.fliplr(next_state.copy()),
            reward,
            game_terminated_flag,
            opponent_won_flag,
            agent_won_flag,
            illegal_agent_move_flag,
            board_full_flag,
        )
        self.memory.append(experience_mirrored)

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
            if board[row][col] == 1:
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
            elif board[row][col] == -1:
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
    q_values_np = q_values.numpy().flatten()
    for col, value in enumerate(q_values):
        # Draw the bar for each column
        if q_values_np[col] == np.max(q_values_np):
            color = (255, 0, 0)
        else:
            color = (10, 10, 255)
        value_norm = (
            q_values_np[col]
            / max(np.max(q_values_np), 1e-6)
            * 1.5
            * config_values.CELL_SIZE
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
    1,
    config_values.HEIGHT,
    config_values.WIDTH,
)  # 1 channel handels both players


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

import time


def check_win(board, row, col):

    player = board[row, col]
    if player == 0:
        return False

    def count_dir(dr, dc):
        count = 0
        r, c = row + dr, col + dc
        while (
            0 <= r < board.shape[0]
            and 0 <= c < board.shape[1]
            and board[r, c] == player
        ):
            count += 1
            r += dr
            c += dc
        return count

    # directions
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        total = 1 + count_dir(dr, dc) + count_dir(-dr, -dc)
        if total >= 4:
            return True
    return False


def is_blocking_opponent(board, action_column, action_row):

    temp_board = np.copy(board)

    # Place a opponent disc in the specified column nad row
    temp_board[action_row, action_column] = -1

    # Check if this move blocks the opponent from connecting four discs
    return check_win(temp_board, action_row, action_column)


def next_empty_row(board, action):

    for r in range(config_values.HEIGHT - 1, -1, -1):  # Start from bottom row
        if board[r, action] == 0:  # Check if the cell is empty (0)
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
    reward += 0.001

    if is_blocking_opponent(agent_board_plane_after_move, action_column, action_row):
        reward += 0.1

    if placed_three_in_a_row(agent_board_plane_after_move, action_column, action_row):
        reward += 0.05

    return reward


# Function to train the opponent
def train_opponent(opponent, opponent_model, epsilon, state, step, episode):
    if opponent == "rand":
        action = np.random.randint(NUM_ACTIONS)
    elif opponent == "self":
        # flip the current state so the opponent sees himself as 1 and not -1
        # otherwise the rl opponent will predict action based on rl agents position
        state_copy = state.copy()
        state_copy = state_copy * -1
        action, _, _ = epsilon_greedy_action(state_copy, epsilon, opponent_model)
    elif opponent == "ascending_columns":
        # Opponent places discs in columns in ascending order
        action = step % NUM_ACTIONS
    elif opponent == "descending_columns":
        # Opponent places discs in columns in descending order
        action = (NUM_ACTIONS - 1) - (step % NUM_ACTIONS)
    elif opponent == "stacker":
        action = episode % NUM_ACTIONS

    if np.any(state[0, :, action] == 0):
        return action
    else:
        # opponent chose an illegal move -> picking random free column instead
        empty = np.any(state[0, :, :] == 0, axis=0)
        empty = np.where(empty == True)
        action = np.random.choice(empty[0])
        return action


# Function to initialize the models
def model_init(train_from_start):
    optimizer = tf.keras.optimizers.Adam(config_values.learning_rate)
    replay_buffer = ReplayBuffer(capacity=config_values.replay_buffer_capacity)

    input_shape = (1, config_values.HEIGHT, config_values.WIDTH)
    dummy_input = tf.zeros(input_shape, dtype=tf.float32)

    if train_from_start:
        model = SmallDuelingDQN(num_actions=NUM_ACTIONS)
        _ = model(dummy_input)
    else:
        model = SmallDuelingDQN(num_actions=NUM_ACTIONS)
        _ = model(dummy_input)

        model.load_weights("./checkpoints/my_checkpoint.weights.h5")
    target_model = SmallDuelingDQN(num_actions=NUM_ACTIONS)
    _ = target_model(dummy_input)
    target_model.set_weights(model.get_weights())

    opponent_model = SmallDuelingDQN(num_actions=NUM_ACTIONS)
    _ = opponent_model(dummy_input)

    return (
        model,
        opponent_model,
        target_model,
        replay_buffer,
        optimizer,
    )


# Function to get RL action when playing game ->used in other code
def get_rl_action(board, model):
    state = board_to_numpy(board)
    q_values = model(state)
    return np.argmax(q_values), q_values


# Convert Connect 4 board to NumPy array and make input indifferent
def board_to_numpy(board):
    array = np.zeros((config_values.HEIGHT, config_values.WIDTH), dtype=np.float32)
    array[np.array(board) == 1] = -1  # human player discs
    array[np.array(board) == 2] = 1  # RL agent discs
    return np.expand_dims(array, 0)  # Add batch dimension


def numpy_to_board(array):
    # Transpose back to the original shape
    array = array.transpose((1, 0, 2))

    # Extract agent and human discs
    agent_discs = np.where(array[:, :, 0] == 1)
    human_discs = np.where(array[:, :, 0] == -1)

    # Create an empty board
    board = np.zeros((config_values.HEIGHT, config_values.WIDTH))

    # Fill in the board with player and opponent discs
    board[agent_discs] = 2
    board[human_discs] = 1

    return board


def choose_opponent(episode, opponent_switch_interval):
    """
    Chooses an opponent based on the current episode and a schedule.
    Gradually increases the probability of self-play as training progresses.
    Also includes fixed opponents at certain intervals.
    """
    progress = episode / config_values.num_episodes

    # At specific intervals, force a non-self opponent to ensure diversity
    if episode > 0 and episode % opponent_switch_interval == 0:
        current_opponent = np.random.choice(
            ["rand", "ascending_columns", "descending_columns", "stacker"]
        )
    else:
        # Gradually increase chance of self-play
        # Starts with a lower chance of self-play, e.g., 30%
        # Ends with a higher chance of self-play, e.g., 90%
        self_play_probability = (
            config_values.min_self_play_prob
            + (config_values.max_self_play_prob - config_values.min_self_play_prob)
            * progress
        )

        if np.random.rand() < self_play_probability:
            current_opponent = "self"
        else:  # Fallback to a fixed opponent if not self-play
            current_opponent = np.random.choice(
                ["rand", "ascending_columns", "descending_columns", "stacker"]
            )

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


def placed_three_in_a_row(
    board: np.ndarray,
    col: int,
    row: int,
) -> bool:
    player = board[row, col]

    def count_dir(dr, dc):
        count = 0
        r, c = row + dr, col + dc
        while (
            0 <= r < board.shape[0]
            and 0 <= c < board.shape[1]
            and board[r, c] == player
        ):
            count += 1
            r += dr
            c += dc
        return count

    # Nur 3er-Reihe zählt
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        total = 1 + count_dir(dr, dc) + count_dir(-dr, -dc)
        if total == 3:
            return True

    return False
