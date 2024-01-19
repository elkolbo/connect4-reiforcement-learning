import pygame
import sys
import tensorflow as tf
import random
from RL_agent import get_rl_action, DQN

# Constants
WIDTH, HEIGHT = 7, 6
CELL_SIZE = 100
WINDOW_WIDTH, WINDOW_HEIGHT = WIDTH * CELL_SIZE, (HEIGHT + 2) * CELL_SIZE
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

# Define constants for players
HUMAN_PLAYER = 1
RL_PLAYER = 2

# Initialize pygame
pygame.init()


# Function to draw the Connect 4 board
def draw_board(screen, board):
    for col in range(WIDTH):
        for row in range(HEIGHT):
            pygame.draw.rect(
                screen,
                BACKGROUND_COLOR,
                (col * CELL_SIZE, (row + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE),
            )
            pygame.draw.circle(
                screen,
                GRID_COLOR,
                (
                    col * CELL_SIZE + CELL_SIZE // 2,
                    (row + 1) * CELL_SIZE + CELL_SIZE // 2,
                ),
                CELL_SIZE // 2,
                5,
            )  # Draw grid circles
            if board[row][col] == 1:
                pygame.draw.circle(
                    screen,
                    RED,
                    (
                        col * CELL_SIZE + CELL_SIZE // 2,
                        (row + 1) * CELL_SIZE + CELL_SIZE // 2,
                    ),
                    CELL_SIZE // 2 - 5,
                )
            elif board[row][col] == 2:
                pygame.draw.circle(
                    screen,
                    BLUE,
                    (
                        col * CELL_SIZE + CELL_SIZE // 2,
                        (row + 1) * CELL_SIZE + CELL_SIZE // 2,
                    ),
                    CELL_SIZE // 2 - 5,
                )


# Function to drop a disc in a column
def drop_disc(board, col, player):
    for row in range(HEIGHT - 1, -1, -1):
        if board[row][col] == 0:
            board[row][col] = player
            return True
    return False


# Function to check for a win
def check_win(board, player):
    # Check horizontally, vertically, and diagonally
    for row in range(HEIGHT):
        for col in range(WIDTH - 3):
            if all(board[row][col + i] == player for i in range(4)):
                return True

    for col in range(WIDTH):
        for row in range(HEIGHT - 3):
            if all(board[row + i][col] == player for i in range(4)):
                return True

    for row in range(3, HEIGHT):
        for col in range(WIDTH - 3):
            if all(board[row - i][col + i] == player for i in range(4)):
                return True

    for row in range(HEIGHT - 3):
        for col in range(WIDTH - 3):
            if all(board[row + i][col + i] == player for i in range(4)):
                return True

    return False


# Function to check for a draw
def check_draw(board):
    return all(board[0][col] != 0 for col in range(WIDTH))


# Function to reset the game
def reset_game():
    return [[0] * WIDTH for _ in range(HEIGHT)]


# ...

# Main game loop
def main():
    model = DQN()
    model = DQN(num_actions=WIDTH)
    model.build([(None, 2, HEIGHT, WIDTH), (None, 5)])
    model.compile(optimizer="adam", loss="mse")

    model.load_weights('./checkpoints/my_checkpoint')
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))  # Adjusted window size
    pygame.display.set_caption("Connect 4")

    clock = pygame.time.Clock()

    board = reset_game()
    current_player = 1

    # Initialize game statistics
    your_wins = 0
    rl_agent_wins = 0
    draws = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and current_player == HUMAN_PLAYER:
                column = event.pos[0] // CELL_SIZE
                if 0 <= column < WIDTH and board[0][column] == 0:
                    if drop_disc(board, column, current_player):
                        if check_win(board, current_player):
                            your_wins += 1
                            end_screen("You Win!", screen, your_wins, rl_agent_wins, draws)
                            board = reset_game()
                        elif check_draw(board):
                            draws += 1
                            end_screen("It's a Draw!", screen, your_wins, rl_agent_wins, draws)
                            board = reset_game()
                        else:
                            current_player = 3 - current_player  # Switch players
            # Add logic for RL agent's move
            if current_player == RL_PLAYER:
                # Get the RL agent's action
                rl_action = get_rl_action(board, model)
                print(rl_action)
                # Update the board based on the RL agent's move
                if drop_disc(board, rl_action, current_player):
                    if check_win(board, current_player):
                        rl_agent_wins += 1
                        end_screen("RL-Agent Wins!", screen, your_wins, rl_agent_wins, draws)
                        board = reset_game()
                    elif check_draw(board):
                        draws += 1
                        end_screen("It's a Draw!", screen, your_wins, rl_agent_wins, draws)
                        board = reset_game()
                    else:
                        current_player = 3 - current_player  # Switch players
                elif drop_disc(
                    board, random.randint(0, WIDTH - 1), current_player
                ):  ##make random action if agent finds no action
                    if check_win(board, current_player):
                        rl_agent_wins += 1
                        end_screen("RL-Agent Wins!", screen, your_wins, rl_agent_wins, draws)
                        board = reset_game()
                    elif check_draw(board):
                        draws += 1
                        end_screen("It's a Draw!", screen, your_wins, rl_agent_wins, draws)
                        board = reset_game()
                    else:
                        current_player = 3 - current_player  # Switch players

        screen.fill(BACKGROUND_COLOR)
        draw_board(screen, board)

        # Display current player
        pygame.draw.circle(
            screen,
            RED if current_player == 1 else BLUE,
            (WIDTH * CELL_SIZE // 2, CELL_SIZE // 2 + CELL_SIZE // 4),  # Adjusted Y position
            CELL_SIZE // 2 - 5,
        )

        # Display game statistics
        font_stats = pygame.font.Font(None, 36)
        your_wins_text = font_stats.render(f"Your Wins: {your_wins}", True, (255, 255, 255))  # White text
        rl_agent_wins_text = font_stats.render(f"RL-Agent Wins: {rl_agent_wins}", True, (255, 255, 255))  # White text
        draws_text = font_stats.render(f"Draws: {draws}", True, (255, 255, 255))  # White text

        rect_your_wins = your_wins_text.get_rect(center=(WIDTH * CELL_SIZE // 4, CELL_SIZE // 8))  # Adjusted Y position
        rect_rl_agent_wins = rl_agent_wins_text.get_rect(center=(3 * WIDTH * CELL_SIZE // 4, CELL_SIZE // 8))  # Adjusted Y position
        rect_draws = draws_text.get_rect(center=(WIDTH * CELL_SIZE // 2, CELL_SIZE // 8))  # Adjusted Y position

        screen.blit(your_wins_text, rect_your_wins)
        screen.blit(rl_agent_wins_text, rect_rl_agent_wins)
        screen.blit(draws_text, rect_draws)

        pygame.display.flip()
        clock.tick(FPS)


# Function to display the end screen
def end_screen(message, screen, player1_wins, player2_wins, draws):
    font_large = pygame.font.Font(None, 74)
    text_large = font_large.render(message, True, (255, 0, 0))
    rect_large = text_large.get_rect(center=(WIDTH * CELL_SIZE // 2, HEIGHT * CELL_SIZE // 2))

    # Create a surface with a lighter background
    background_surface = pygame.Surface((WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE))
    background_surface.fill((255, 255, 255))  # Lighter background color
    background_rect = background_surface.get_rect(topleft=(0, CELL_SIZE))
    screen.blit(background_surface, background_rect)

    # Blit the large text on the background surface
    screen.blit(text_large, rect_large)

    # Display game statistics with a smaller font
    font_small = pygame.font.Font(None, 36)
    stats_text = font_small.render(
        f"Player 1 Wins: {player1_wins} | RL-Agent Wins: {player2_wins} | Draws: {draws}",
        True,
        (0, 0, 0),
    )
    rect_stats = stats_text.get_rect(center=(WIDTH * CELL_SIZE // 2, HEIGHT * CELL_SIZE // 1.5))
    screen.blit(stats_text, rect_stats)

    # Display instructions with a smaller font
    instructions = font_small.render("Press any key to continue", True, (0, 0, 0))
    rect_small = instructions.get_rect(center=(WIDTH * CELL_SIZE // 2, HEIGHT * CELL_SIZE // 1.3))
    screen.blit(instructions, rect_small)

    pygame.display.flip()

    pygame.time.wait(2000)  # Pause for 2 seconds before starting a new game

    waiting_for_key = True
    while waiting_for_key:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                waiting_for_key = False


if __name__ == "__main__":
    main()
