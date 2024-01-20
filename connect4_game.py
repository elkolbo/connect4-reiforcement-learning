import pygame
import sys
import tensorflow as tf
import random
from RL_agent import get_rl_action, DQN

# Constants
WIDTH, HEIGHT = 7, 6
CELL_SIZE = 100
WINDOW_WIDTH, WINDOW_HEIGHT = WIDTH * CELL_SIZE, (HEIGHT + 2.5) * CELL_SIZE

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
                (col * CELL_SIZE, (row + 1.5) * CELL_SIZE, CELL_SIZE, CELL_SIZE),  # Angepasste Y-Koordinate
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
            if board[row][col] == 1:
                pygame.draw.circle(
                    screen,
                    RED,
                    (
                        col * CELL_SIZE + CELL_SIZE // 2,
                        (row + 1.5) * CELL_SIZE + CELL_SIZE // 2,  # Angepasste Y-Koordinate
                    ),
                    CELL_SIZE // 2 - 5,
                )
            elif board[row][col] == 2:
                pygame.draw.circle(
                    screen,
                    BLUE,
                    (
                        col * CELL_SIZE + CELL_SIZE // 2,
                        (row + 1.5) * CELL_SIZE + CELL_SIZE // 2,  # Angepasste Y-Koordinate
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


# Main game loop
def main():
    model = DQN()
    model = DQN(num_actions=WIDTH)
    model.build([(None, 2, HEIGHT, WIDTH), (None, 5)])
    model.compile(optimizer="adam", loss="mse")

    model.load_weights('./checkpoints/my_checkpoint')
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))  # Angepasste Fenstergröße
    pygame.display.set_caption("Connect 4")

    clock = pygame.time.Clock()

    board = reset_game()
    current_player = 1

    # Initialize game statistics
    your_wins = 0
    rl_agent_wins = 0
    draws = 0

    # Height of the space below the game board for displaying stats
    stats_height = 80

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
            (WIDTH * CELL_SIZE // 2, CELL_SIZE // 2 + 25), 
            CELL_SIZE // 2 - 5,
        )

        # Draw stats background
        stats_background_rect = pygame.Rect(0, WINDOW_HEIGHT - stats_height, WINDOW_WIDTH, stats_height)
        pygame.draw.rect(screen, BACKGROUND_COLOR, stats_background_rect)

        # Display game statistics
        font_stats = pygame.font.Font(None, 36)
        your_wins_text = font_stats.render(f"Your Wins: {your_wins}", True, (255, 255, 255))  # White text
        rl_agent_wins_text = font_stats.render(f"Agent Wins: {rl_agent_wins}", True, (255, 255, 255))  # White text
        draws_text = font_stats.render(f"Draws: {draws}", True, (255, 255, 255))  # White text

        rect_your_wins = your_wins_text.get_rect(center=(WINDOW_WIDTH // 4, WINDOW_HEIGHT - stats_height // 2))
        rect_rl_agent_wins = rl_agent_wins_text.get_rect(center=(3 * WINDOW_WIDTH // 4, WINDOW_HEIGHT - stats_height // 2))
        rect_draws = draws_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - stats_height // 2))

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

    # Display game statistics with a smaller font
    font_small = pygame.font.Font(None, 36)
    stats_text = font_small.render(
        f"Your Wins: {player1_wins} | Agent Wins: {player2_wins} | Draws: {draws}",
        True,
        (255, 255, 255),
    )

    # Display instructions with a smaller font
    instructions = font_small.render("Press any key to continue", True, (255, 255, 255))

    # Calculate the size of the background surface
    padding = 20
    background_width = max(rect_large.width, stats_text.get_width(), instructions.get_width()) + 2 * padding
    background_height = rect_large.height + stats_text.get_height() + instructions.get_height() + 4 * padding

    background_surface = pygame.Surface((background_width, background_height), pygame.SRCALPHA)
    pygame.draw.rect(background_surface, (255, 255, 255, 128), (0, 0, background_width, background_height), border_radius=20)
    pygame.draw.rect(background_surface, (255, 255, 255), (0, 0, background_width, background_height), 5, border_radius=20)

    # Blit the large text on the background surface
    background_surface.blit(text_large, (background_width // 2 - rect_large.width // 2, padding))

    # Blit the game statistics on the background surface
    background_surface.blit(stats_text, (padding, rect_large.height + 2 * padding))

    # Blit the instructions on the background surface, centered horizontally
    instructions_x = (background_width - instructions.get_width()) // 2
    background_surface.blit(instructions, (instructions_x, rect_large.height + stats_text.get_height() + 3 * padding))

    background_rect = background_surface.get_rect(center=(WIDTH * CELL_SIZE // 2, HEIGHT * CELL_SIZE // 2))
    screen.blit(background_surface, background_rect)

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
