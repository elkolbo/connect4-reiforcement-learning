import tensorflow as tf
import numpy as np
import random

# Constants
WIDTH, HEIGHT = 7, 6
NUM_ACTIONS = WIDTH
STATE_SHAPE = (3, HEIGHT, WIDTH)  # 3 channels for current player, opponent, and empty spaces

# Deep Q Network (DQN) Model
class DQN(tf.keras.Model):
    def __init__(self, num_actions=WIDTH):
        super(DQN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_actions)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# Epsilon-Greedy Exploration
def epsilon_greedy_action(state, epsilon, model):
    if np.random.rand() < epsilon:
        return np.random.randint(NUM_ACTIONS)  # Explore
    else:
        q_values = model.predict(state)
        return np.argmax(q_values)  # Exploit

# Convert Connect 4 board to NumPy array and make input indifferent -> doesnt matter which player number the agent has
def board_to_numpy(board, current_player):
    array = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    array[:, :, 0] = (board == current_player) # Current player's discs
    array[:, :, 1] = (board == 3 - current_player)  # Opponent's discs
    array[:, :, 2] = (board == 0) # Empty spaces
    return array.transpose((2, 0, 1))[np.newaxis, :]  # Add batch dimension

def model_init():
    # Initialize DQN model and optimizer
    model = DQN(num_actions=NUM_ACTIONS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=10000)

    # Training parameters
    gamma = 0.99  # Discount factor
    epsilon_start = 1.0  # Initial exploration rate
    epsilon_end = 0.01  # Final exploration rate
    epsilon_decay = 0.999  # Exploration rate decay
    target_update_frequency = 10  # Update target network every N episodes
    batch_size = 64

    model.compile(optimizer='adam', loss='mse')


    # Training loop
    num_episodes = 2
    max_steps_per_episode = 5  # You can adjust this value
    print("starting training")
    for episode in range(1, num_episodes + 1):
        state = np.zeros((1, 3, HEIGHT, WIDTH), dtype=np.float32)  # Initial state
        done = False
        step = 0  # Counter for steps in the episode
        print(episode)
        while not done and step < max_steps_per_episode:
            epsilon = max(epsilon_end, epsilon_start * epsilon_decay ** episode)
            action = epsilon_greedy_action(state, epsilon, model)

            # Simulate environment (in this case, play a random opponent)
            opponent_action = np.random.randint(NUM_ACTIONS)
            next_state = board_to_numpy(np.zeros((HEIGHT, WIDTH), dtype=np.int64), 3 - (episode % 2) + 1)  # Opponent's turn
            reward = 0  # Reward is 0 during gameplay

            # Check if the action is valid and update the board
            if np.sum(state[0, 0]) < HEIGHT * WIDTH and state[0, 0, :, action].sum() == 0:
                reward = 1  # The player gets a reward for making a valid move
                state[0, 0, :, action] = 1  # Update the board

            next_state[0, 1] = state[0, 0].copy()  # Copy the current player's discs
            state = next_state.copy()

            # Store the experience in the replay buffer
            replay_buffer.push(state, action, next_state, reward)

            # Sample a random batch from the replay buffer and perform a Q-learning update
            if len(replay_buffer.memory) > batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, next_states, rewards = zip(*batch)

                states = np.concatenate(states)
                actions = np.array(actions, dtype=np.int32).reshape(-1, 1)
                next_states = np.concatenate(next_states)
                rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)

                # Create a new GradientTape for each iteration
                with tf.GradientTape() as tape:
                    current_q_values = model(states, training=True)
                    current_q_values = tf.reduce_sum(tf.one_hot(actions, NUM_ACTIONS) * current_q_values, axis=1, keepdims=True)

                    next_q_values = model(next_states, training=True)
                    next_q_values = tf.reduce_max(next_q_values, axis=1, keepdims=True)

                    target_q_values = rewards + gamma * next_q_values

                    loss = tf.reduce_mean(tf.square(current_q_values - target_q_values))

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Increment the step counter
            step += 1

        # Print statistics
        if episode % 10 == 0:
            print(f"Episode: {episode}, Epsilon: {epsilon:.3f}")

    print("Training complete.")
    dummy_input = np.zeros((1, 3, HEIGHT, WIDTH), dtype=np.float32)
    model(dummy_input)
    model.save(r"C:\Users\loren\Documents\AI_BME\FinalProject\connect4-reiforcement-learning\saved_model.tf")

    return model


def get_rl_action(board,model):
    state = board_to_numpy(board, 2)  # Assuming RL agent is player 2
    q_values = model.predict(state)
    return np.argmax(q_values)

if __name__=="__main__":
    model=model_init()


