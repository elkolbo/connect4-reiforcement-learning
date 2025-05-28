import tensorflow as tf
import numpy as np
import random
import pathlib
import pygame

import os
from datetime import datetime
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Concatenate
from config import config
from agent_helper_functions import *

# load the config to access all hyperparameters
path = pathlib.Path(__file__).parent / "logs"

config_values = config()
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


# intit pygame to visualize
pygame.init()

screen = pygame.display.set_mode(
    (config_values.WINDOW_WIDTH, config_values.WINDOW_HEIGHT)
)  # Angepasste Fenstergröße
pygame.display.set_caption("Connect 4")

clock = pygame.time.Clock()

if __name__ == "__main__":
    (model, opponent_model, target_model, replay_buffer, optimizer) = model_init(
        config_values.train_from_start
    )

    gamma = config_values.gamma

    epsilon_scheduler = EpsilonScheduler(
        config_values.epsilon_start,
        config_values.epsilon_end,
        config_values.num_episodes,
        config_values.reach_target_epsilon,
        "linear",
    )

    target_update_frequency = config_values.target_update_frequency
    batch_size = config_values.batch_size

    max_steps_per_episode = (
        44  # higher than possible steps to enforce full board break to step in
    )

    for episode in range(1, config_values.num_episodes + 1):
        print(f"New episode starting :{episode}/{config_values.num_episodes}")
        print("*" * 50)
        state = np.zeros(
            (1, 2, config_values.HEIGHT, config_values.WIDTH), dtype=np.float32
        )
        done = False
        step = 0
        epsilon = epsilon_scheduler.calculate_epsilon(episode)
        game_ended = False
        current_episode_batch_losses = (
            []
        )  # List to store batch losses for the current episode

        current_opponent = choose_opponent(
            episode, config_values.opponent_switch_interval
        )  # Self, Random or Ascending_Columns

        pygame.event.pump()
        rewards_episode_log = []

        while not done and step < max_steps_per_episode and not game_ended:
            # move of the RL agent
            action, q_values, random_move = epsilon_greedy_action(state, epsilon, model)
            # check if move is legal
            if (
                state[0, :, :, action].sum() < config_values.HEIGHT and not game_ended
            ):  # agent makes legal move and game has not ended
                # calculate next state and reward of this legal move
                # passing on without batch dimension
                empty_row = next_empty_row(state[0], action)
                next_state = state.copy()
                next_state[0, 0, empty_row, action] = (
                    1  # updation state, channel 0 is always for agent
                )
                next_state_opponent = next_state.copy()
                # Pass the agent's plane (H,W) AFTER the move, the action column, and the action row (empty_row)
                reward = calculate_reward(next_state[0, 0, :, :], action, empty_row)
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
                        priority=1.0,  # Initial high priority for new experiences
                    )

                    game_ended = True

                else:  # move was a "normal" game move, game continues
                    # FIXME: The is_blocking_opponent logic in agent_helper_functions.py is problematic.
                    # if is_blocking_opponent(
                    #     state[0], action
                    # ):
                    #     # Reward for blocking the opponent
                    #     reward += 20
                    # calculate opponennts move

                    opponent_action = train_opponent(
                        current_opponent, opponent_model, epsilon, next_state, step
                    )

                    next_state_opponent = next_state.copy()
                    # opponent can be "rand" or "self"
                    if (
                        next_state[0, :, :, opponent_action].sum()
                        < config_values.HEIGHT
                    ):
                        empty_row = next_empty_row(next_state[0], opponent_action)
                        next_state_opponent[0, 1, empty_row, opponent_action] = 1
                    else:
                        # opponent chose an illegal move -> picking free column instead
                        for column in range(config_values.WIDTH):
                            if next_state[0, :, :, column].sum() < config_values.HEIGHT:
                                opponent_action = column
                                break
                        empty_row = next_empty_row(next_state[0], opponent_action)
                        next_state_opponent[0, 1, empty_row, opponent_action] = 1

                    if check_win(next_state_opponent[0]):  # opponent won
                        print("EPISODE ENDED BY WIN OF OPPONENT")
                        print(next_state_opponent[0])
                        print("#" * 30)
                        reward = -1000
                        replay_buffer.push(
                            state,
                            action,
                            next_state_opponent,
                            reward,
                            game_terminated_flag=1,
                            opponent_won_flag=1,
                            agent_won_flag=0,
                            illegal_agent_move_flag=0,
                            board_full_flag=0,
                            priority=1.0,
                        )
                        game_ended = True

                    else:  # opponent didn't win
                        replay_buffer.push(
                            state,
                            action,
                            next_state_opponent,
                            reward,
                            game_terminated_flag=0,
                            opponent_won_flag=0,
                            agent_won_flag=0,
                            illegal_agent_move_flag=0,
                            board_full_flag=0,
                            priority=1.0,
                        )
                    next_state = (
                        next_state_opponent.copy()
                    )  # copy for correct continuation in next episode
            elif (
                not np.sum(next_state[0]) < config_values.HEIGHT * config_values.WIDTH
                and not game_ended
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
                    priority=1.0,
                )
                game_ended = True
            elif not game_ended:
                if not random_move:  # report if illegal move was actively chosen
                    print("--- AGENT CHOSE ILLEGAL MOVE (TRAINING) ---")
                    print(f"Current State (Agent's view channel 0):\n{state[0,0,:,:]}")
                    print(
                        f"Current State (Opponent's view channel 1):\n{state[0,1,:,:]}"
                    )
                    print(f"Chosen illegal action column: {action}")
                    # Re-evaluate Q-values for this state without exploration to see greedy choice
                    _, q_values_for_illegal_log, _ = epsilon_greedy_action(
                        state, 0, model
                    )  # Epsilon = 0
                    print(
                        f"Q-values for this state: {q_values_for_illegal_log.numpy()}"
                    )
                    print(
                        "-----------------------------------------"
                    )  # agent makes illegal move
                reward = -1001
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
                    priority=1.0,
                )
                print("Episode ended by agent illegal move")
                game_ended = True

            # set next state as state for next step of episode
            state = next_state.copy()

            if len(replay_buffer.memory) > batch_size:
                batch = replay_buffer.sample(batch_size)
                (
                    indices,
                    states,
                    actions,
                    next_states,
                    rewards,
                    game_terminated_flags,
                    opponent_won_flags,
                    agent_won_flags,
                    illegal_agent_move_flags,
                    board_full_flags,
                    is_weights,  # Importance Sampling Weights
                ) = zip(*batch)
                indices = np.array(indices, dtype=np.int32)
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
                is_weights = np.array(is_weights, dtype=np.float32).reshape(-1, 1)

                with tf.GradientTape() as tape:
                    current_q_values_all_actions = model(states)
                    one_hot = tf.squeeze(tf.one_hot(actions, NUM_ACTIONS), axis=1)
                    # Select Q-value for the action taken
                    current_q_values = one_hot * current_q_values_all_actions
                    current_q_values = tf.reduce_sum(
                        current_q_values,
                        axis=-1,
                        keepdims=True,
                    )

                    # Use Double DQN: select max action from online model, get Q-value from target model
                    # 1. Get the best actions for next_states from the online model
                    next_actions_from_online_model = tf.argmax(
                        model(next_states), axis=1
                    )
                    # 2. Get all Q-values for next_states from the target model
                    next_q_values_from_target_model_all = target_model(next_states)
                    # 3. Select the Q-values from the target model corresponding to the best actions chosen by the online model
                    # We need to create indices for tf.gather_nd
                    batch_indices = tf.range(
                        tf.shape(next_actions_from_online_model)[0], dtype=tf.int64
                    )
                    action_indices = tf.stack(
                        [batch_indices, next_actions_from_online_model], axis=1
                    )
                    next_q_values = tf.gather_nd(
                        next_q_values_from_target_model_all, action_indices
                    )
                    next_q_values = tf.expand_dims(
                        next_q_values, axis=1
                    )  # Ensure it's (batch_size, 1)

                    # Ensure future reward is 0 if the state was terminal
                    target_q_values = rewards + gamma * next_q_values * (
                        1 - game_terminated_flags.reshape(-1, 1)
                    )

                    td_error = current_q_values - target_q_values

                    # Update priorities in replay buffer using absolute TD error
                    abs_td_error = tf.abs(td_error)
                    replay_buffer.update_priorities(indices, abs_td_error)

                    # Calculate weighted loss for gradient update
                    # Loss is (TD_error)^2 * IS_weight
                    weighted_loss = tf.square(td_error) * is_weights
                    batch_loss = tf.reduce_sum(weighted_loss)
                    current_episode_batch_losses.append(batch_loss.numpy())

                gradients = tape.gradient(batch_loss, model.trainable_variables)
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

            if episode % config_values.visualization_frequency == 0:
                screen.fill(config_values.BACKGROUND_COLOR)  # Clear the screen
                draw_board(screen, state[0])
                visualize_training(
                    screen, q_values, random_move, action, reward, current_opponent
                )
                pygame.display.flip()
                pygame.display.update()  # forced display update
                pygame.time.wait(200)
                # Wait for a bit to make it easier to follow visualization.
                # Consider reducing wait time or frequency for faster overall training.

            rewards_episode_log.append(reward)
            # Inside the training loop
        # At the end of the episode, log the average batch loss and other episode-level metrics
        with summary_writer.as_default():
            if current_episode_batch_losses:  # Ensure there were training steps
                average_episode_loss = np.mean(current_episode_batch_losses)
                tf.summary.scalar(
                    "Average Batch Loss per Episode", average_episode_loss, step=episode
                )
            tf.summary.scalar("Epsilon", epsilon, step=episode)

            if rewards_episode_log:
                avg_reward = np.array(rewards_episode_log).mean()
                tf.summary.scalar(
                    "Average reward during episode", avg_reward, step=episode
                )

            # You can add other episode-level summary statistics here, e.g., total reward for the episode

        if episode % target_update_frequency == 0:
            model_weights = model.get_weights()
            opponent_model_weights = opponent_model.get_weights()

            opponent_model.set_weights(model_weights)

            target_model.set_weights(model.get_weights())

        if episode % 10 == 0:
            print(f"Episode: {episode}, Epsilon: {epsilon:.3f}")

        if episode % 1000 == 0:
            model.save_weights(f"./checkpoints/my_checkpoint_epochs{episode}.h5")

    print("Training complete.")

    # Save the weights
    model.save_weights("./checkpoints/my_checkpoint.h5")


# Close the writer
summary_writer.close()
