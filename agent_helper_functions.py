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
            32, (3, 3), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv2 = Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", activation="relu"
        )
        self.conv3 = Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", activation="relu"
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


# class ReplayBuffer: # Old Pandas-based buffer
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = pd.DataFrame(
#             columns=[
#                 "index",
#                 "state",
#                 "action",
#                 "next_state",
#                 "reward",
#                 "game_terminated_flag",
#                 "opponent_won_flag",
#                 "agent_won_flag",
#                 "illegal_agent_move_flag",
#                 "board_full_flag",
#                 "loss",
#             ]
#         )
#         self.position = 0
#         self.next_index = 0

#     def push(
#         self,
#         state,
#         action,
#         next_state,
#         reward,
#         game_terminated_flag,
#         opponent_won_flag,
#         agent_won_flag,
#         illegal_agent_move_flag,
#         board_full_flag,
#         loss, # This 'loss' is used as initial priority
#     ):
#         row = pd.DataFrame(
#             [
#                 (
#                     self.next_index,
#                     state,
#                     action,
#                     next_state,
#                     reward,
#                     game_terminated_flag,
#                     opponent_won_flag,
#                     agent_won_flag,
#                     illegal_agent_move_flag,
#                     board_full_flag,
#                     loss,
#                 )
#             ],
#             columns=self.memory.columns,
#         )

#         if len(self.memory) < self.capacity:
#             self.memory = pd.concat([self.memory, row], ignore_index=True)
#         else:
#             self.memory.loc[self.position] = row.iloc[0]

#         self.position = (self.position + 1) % self.capacity
#         self.next_index += 1

#     def update_loss(self, indices, new_loss): # new_loss here are the TD errors
#         for i, index in enumerate(indices): # 'index' here is the unique ID from self.next_index
#             self.memory.loc[self.memory["index"] == index, "loss"] = new_loss[i].numpy().item() # Store new priority

#     def sample(self, batch_size):
#         if len(self.memory) < batch_size:
#             raise ValueError("Not enough samples in the replay buffer")

#         sorted_memory = self.memory.sort_values(by="loss", ascending=False)
#         selected_samples_df = sorted_memory.head(batch_size)

#         # We need to return the unique 'index' for loss updates, and the actual data
#         # The original code expected a list of tuples, where the first element was the unique ID.
#         # selected_samples_list = [tuple(row) for row in selected_samples_df.values]
#         # For PER, we need to return (indices_for_update, samples, IS_weights)
#         # The current RL_agent.py unpacks: indices, states, actions, ..., losses (which are priorities)
#         # Let's adapt to return what RL_agent.py expects for now, but this isn't full PER.
#         # The 'indices' it expects are the unique IDs.

#         # To make it compatible with the current RL_agent.py unpacking,
#         # we need to return the unique ID as the first element of each sample tuple.
#         # The last element is the 'loss' (priority).
#         # We also need to return a dummy weight if we are not doing full PER yet.

#         # This is still not proper PER, just greedy sampling.
#         # For proper PER, we need probabilistic sampling and IS weights.
#         # The previous diff for PER was more accurate. Let's re-implement that.
#         # The current RL_agent.py expects the 'index' from the DataFrame as the first element.

#         # For now, let's keep the structure simple and address PER more thoroughly if requested.
#         # The current sample method is problematic for true PER.
#         # The user's RL_agent.py expects:
#         # (indices, states, actions, ..., losses) = zip(*batch)
#         # where 'indices' are the unique IDs for updating losses.
#         # 'losses' are the priorities.

#         # The DataFrame stores: index, state, action, ..., loss (priority)
#         # So each row in selected_samples_df.values is (unique_id, state, ..., priority)
#         selected_samples_list = [tuple(row) for row in selected_samples_df.values]
#         return selected_samples_list


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
        pygame.draw.rect(
            screen,
            color,
            (
                col * config_values.CELL_SIZE + 0.25 * config_values.CELL_SIZE,
                config_values.WINDOW_HEIGHT - int(value.numpy() * 200),
                config_values.CELL_SIZE * 0.5,
                int(value.numpy() * 200),
            ),
        )

        # Display Q-values on the bars with 1 digit after the comma
        font_q_values = pygame.font.Font(None, 20)
        q_value_text = font_q_values.render(f"{value:.3f}", True, (255, 255, 255))
        screen.blit(
            q_value_text,
            (col * config_values.CELL_SIZE, config_values.WINDOW_HEIGHT - 20),
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
def calculate_reward(board, action, current_player):
    # Default reward
    reward = 0

    # check if board has free spaces (not necessary but doesn't hurt)
    if np.sum(board) < config_values.HEIGHT * config_values.WIDTH:
        # Check if the column is full
        adjacent_count = count_adjacent_discs(board, action)
        reward += 0.1 * adjacent_count  # Increase reward based on the count

        # Reward for a valid move
        reward += 1  # Give a small reward for a valid move

    else:
        print("BOARD IS FULL!!")

    return reward


# def is_blocking_opponent(board, action_column):
#     return False


def count_adjacent_discs(board, action_column):
    # check surrounings and count discs
    count = 0
    action_row = 0
    for row in range(config_values.HEIGHT):  # find row in which ction was taken
        if board[0, row, action_column] == 1:
            action_row = row

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
        model.build([(None, 2, config_values.HEIGHT, config_values.WIDTH), (None, 5)])
        model.build((None, 2, config_values.HEIGHT, config_values.WIDTH))
    else:
        model = DQN(num_actions=NUM_ACTIONS)
        model.build((None, 2, config_values.HEIGHT, config_values.WIDTH))

        model.load_weights("./checkpoints/my_checkpoint.h5")
    target_model = DQN(num_actions=NUM_ACTIONS)
    target_model.build(
        [(None, 2, config_values.HEIGHT, config_values.WIDTH), (None, 5)]
    )
    target_model.set_weights(model.get_weights())

    opponent_model = DQN(num_actions=NUM_ACTIONS)
    opponent_model.build(
        [(None, 2, config_values.HEIGHT, config_values.WIDTH), (None, 5)]
    )

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

    return current_opponent
