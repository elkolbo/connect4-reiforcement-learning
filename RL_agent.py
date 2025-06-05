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
        config_values.epsilon_mode,
    )

    batch_size = config_values.batch_size

    max_steps_per_episode = (
        44  # higher than possible steps to enforce full board break to step in
    )

    for episode in range(1, config_values.num_episodes + 1):
        print(f"New episode starting :{episode}/{config_values.num_episodes}")
        print("*" * 50)
        state = np.zeros(
            (1, config_values.HEIGHT, config_values.WIDTH), dtype=np.float32
        )
        next_state = state.copy()
        step = 0
        reward = 0
        epsilon = epsilon_scheduler.calculate_epsilon(episode)
        game_ended = False
        current_episode_batch_losses = (
            []
        )  # List to store batch losses for the current episode

        current_opponent = choose_opponent(
            episode, config_values.opponent_switch_interval
        )  # Self, Random or Ascending/descending Columns

        pygame.event.pump()
        rewards_episode_log = []
        agent_starts = random.choice([True, False])

        while step < max_steps_per_episode and not game_ended:

            # move of the RL agent
            if (step % 2 == 0 and agent_starts) or (step % 2 == 1 and not agent_starts):

                if current_opponent == "self":

                    action, q_values, random_move = epsilon_greedy_action(
                        state, epsilon, model
                    )
                else:
                    action, q_values, random_move = epsilon_greedy_action(
                        state, epsilon * 0.15, model
                    )
                if (
                    np.any(state[0, :, action] == 0) and not game_ended
                ):  # agent makes legal move and game has not ended
                    # calculate next state and reward of this legal move
                    # passing on without batch dimension
                    empty_row = next_empty_row(state[0], action)
                    next_state = state.copy()
                    next_state[0, empty_row, action] = (
                        1  # update state, agent discs are always 1
                    )
                    next_state_opponent = next_state.copy()

                    # agent made legal move, now check the outcome of the move:

                    if check_win(next_state[0, ...], empty_row, action):  # agent won
                        print("EPISODE ENDED BY WIN OF AGENT")
                        print(next_state[0])
                        print("#" * 30)
                        reward = 10
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
                        # reward for normal move
                        reward = calculate_reward(
                            next_state[0, :, :], action, empty_row
                        )

                elif (
                    not np.any(state[0, :, action] == 0) and not game_ended
                ):  # illegal move
                    if not random_move:  # report if illegal move was actively chosen
                        print("--- AGENT CHOSE ILLEGAL MOVE (TRAINING) ---")
                        print(f"Current State :\n{state[0,:,:]}")
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
                    reward = -11
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
            else:  # opponents turn
                opponent_action = train_opponent(
                    current_opponent, opponent_model, epsilon, next_state, step, episode
                )

                next_state_opponent = next_state.copy()
                empty_row = next_empty_row(next_state[0], opponent_action)
                next_state_opponent[0, empty_row, opponent_action] = -1

                if check_win(
                    next_state_opponent[0], empty_row, opponent_action
                ):  # opponent won
                    print("EPISODE ENDED BY WIN OF OPPONENT")
                    print(next_state_opponent[0])
                    print("#" * 30)
                    reward = -10
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

                elif step != 0:  # opponent didn't win
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

            # check if board is full after move
            if np.all(next_state[0, :, :] != 0):
                print("EPISODE ENDED BY FULL BOARD")
                reward = 0
                replay_buffer.push(
                    state,
                    action if action is not None else 0,
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
                break
            # set next state as state for next step of episode
            state = next_state.copy()

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
            if step > 1 and not game_ended:
                rewards_episode_log.append(reward)

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
            game_terminated_flags = np.array(game_terminated_flags, dtype=np.float32)
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

                # For Q-value stability monitoring
                avg_q_value_in_batch = tf.reduce_mean(current_q_values_all_actions)
                max_q_value_in_batch = tf.reduce_max(current_q_values_all_actions)
                min_q_value_in_batch = tf.reduce_min(current_q_values_all_actions)

                # Use Double DQN: select max action from online model, get Q-value from target model
                # 1. Get the best actions for next_states from the online model
                next_actions_from_online_model = tf.argmax(model(next_states), axis=1)
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
                # Ensure abs_td_error is detached from the graph for priority updates
                abs_td_error = tf.abs(td_error)
                replay_buffer.update_priorities(indices, abs_td_error)

                # Calculate weighted loss for gradient update
                # Loss is (TD_error)^2 * IS_weight
                weighted_loss = tf.square(td_error) * is_weights  # Element-wise
                batch_loss = tf.reduce_mean(weighted_loss)  # Mean over the batch
                current_episode_batch_losses.append(batch_loss.numpy())

            gradients = tape.gradient(batch_loss, model.trainable_variables)
            # Clip gradients to stabilize training
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 2)
            max_gradient = tf.reduce_max([tf.reduce_max(grad) for grad in gradients])
            max_clipped_gradient = tf.reduce_max(
                [tf.reduce_max(grad) for grad in clipped_gradients]
            )

            tf.summary.scalar(f"Unclipped garadients max", max_gradient, step=episode)
            tf.summary.scalar(
                f"Clipped garadients max", max_clipped_gradient, step=episode
            )
            optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

        with summary_writer.as_default():
            if current_episode_batch_losses:
                average_episode_loss = np.sum(current_episode_batch_losses)
                tf.summary.scalar(
                    "Average Batch Loss per Episode", average_episode_loss, step=episode
                )
            tf.summary.scalar("Epsilon", epsilon, step=episode)

            current_lr = optimizer.learning_rate.numpy()
            tf.summary.scalar("Learning Rate", current_lr, step=episode)

            if rewards_episode_log and len(replay_buffer.memory) > batch_size:
                summed_reward = np.array(rewards_episode_log).sum()
                tf.summary.scalar(
                    "Summed reward during episode (without terminal reward)",
                    summed_reward,
                    step=episode,
                )
                # Log Q-value statistics
                tf.summary.scalar(
                    "Q-Values/Average Batch Q-Value", avg_q_value_in_batch, step=episode
                )
                tf.summary.scalar(
                    "Q-Values/Max Batch Q-Value", max_q_value_in_batch, step=episode
                )
                tf.summary.scalar(
                    "Q-Values/Min Batch Q-Value", min_q_value_in_batch, step=episode
                )
                tf.summary.scalar(
                    "TD Error/Average Absolute TD Error",
                    tf.reduce_mean(abs_td_error),
                    step=episode,
                )

        if episode % config_values.target_update_frequency == 0:
            target_model.set_weights(model.get_weights())
            print(f"Target model updated at episode {episode}")

        if episode % config_values.opponent_model_update_frequency == 0:
            opponent_model.set_weights(model.get_weights())
            print(f"Opponent model updated at episode {episode}")

        if episode % 1000 == 0:
            model.save_weights(
                f"./checkpoints/my_checkpoint_epochs{episode}.weights.h5"
            )

    print("Training complete.")

    # Save the weights
    model.save_weights("./checkpoints/my_checkpoint.weights.h5")


# Close the writer
summary_writer.close()
