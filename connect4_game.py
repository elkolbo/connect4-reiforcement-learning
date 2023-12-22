import pygame
import sys
from RL_agent import get_rl_action, DQN

# Constants
WIDTH, HEIGHT = 7, 6
CELL_SIZE = 100
WINDOW_SIZE = (WIDTH * CELL_SIZE, (HEIGHT + 1) * CELL_SIZE)  # Extra row for displaying the current player
FPS = 30

# Colors
BACKGROUND_COLOR = (200, 200, 200)  # Neutral background color
GRID_COLOR = (0, 0, 0)  # Grid color
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
            pygame.draw.rect(screen, BACKGROUND_COLOR, (col * CELL_SIZE, (row + 1) * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.circle(screen, GRID_COLOR, (col * CELL_SIZE + CELL_SIZE // 2, (row + 1) * CELL_SIZE + CELL_SIZE // 2),
                               CELL_SIZE // 2, 5)  # Draw grid circles
            if board[row][col] == 1:
                pygame.draw.circle(screen, RED, (col * CELL_SIZE + CELL_SIZE // 2, (row + 1) * CELL_SIZE + CELL_SIZE // 2),
                                   CELL_SIZE // 2 - 5)
            elif board[row][col] == 2:
                pygame.draw.circle(screen, BLUE, (col * CELL_SIZE + CELL_SIZE // 2, (row + 1) * CELL_SIZE + CELL_SIZE // 2),
                                   CELL_SIZE // 2 - 5)

# Function to drop a disc in a column
def drop_disc(board, col, player):
    for row in range(HEIGHT - 1, -1, -1):
        if board[row][col] == 0:
            board[row][col] = player
            return True
    return False

# Function to check for a win
def check_win(board, player):
    # Check horizontally, vertically, and diagonally (similar to the previous code)
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
    # If you want to play the game with a pre-trained model, uncomment the lines below
    #model = DQN()
    #model.load_weights("path/to/your/saved/model/weights")

    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('Connect 4')

    clock = pygame.time.Clock()

    board = reset_game()
    current_player = 1

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
                            print(f"Player {current_player} wins!")
                            board = reset_game()
                        elif check_draw(board):
                            print("It's a draw!")
                            board = reset_game()
                        else:
                            current_player = 3 - current_player  # Switch players
            # Add logic for RL agent's move
            if current_player == RL_PLAYER:
                # Get the RL agent's action
                #rl_action = get_rl_action(board)
                # Update the board based on the RL agent's move
                if drop_disc(board, 1, current_player):
                    if check_win(board, current_player):
                        print(f"Player {current_player} wins!")
                        board = reset_game()
                    elif check_draw(board):
                        print("It's a draw!")
                        board = reset_game()
                    else:
                        current_player = 3 - current_player  # Switch players

        screen.fill(BACKGROUND_COLOR)
        draw_board(screen, board)

        # Display current player
        pygame.draw.circle(screen, RED if current_player == 1 else BLUE, (WIDTH * CELL_SIZE // 2, CELL_SIZE // 2),
                           CELL_SIZE // 2 - 5)

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
