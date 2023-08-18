import numpy as np
from tensorflow import keras
import time

# Game constants
BOARD_SIZE = 9
EMPTY = 0
AI_PLAYER = 1
OPPONENT = -1

# Neural network architecture
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(BOARD_SIZE,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(BOARD_SIZE, activation='softmax')
])

# Function to get user's move
def get_user_move(board):
    while True:
        user_input = input("Enter your move (1-9) or 101 to end the game: ")
        if user_input == "101":
            return 100  # Special value to end the game
        elif user_input.isdigit():
            move = int(user_input) - 1  # Subtract 1 to convert to 0-based index
            if move in range(BOARD_SIZE) and board[move] == EMPTY:
                return move
            print("Invalid move. Try again.")
        else:
            print("Invalid input. Try again.")

# Function to train the model
def train_model(training_data, epochs):
    game_boards, moves = zip(*training_data)
    game_boards = np.array(game_boards)
    moves = np.array(moves)

    # Convert moves to one-hot encoded vectors
    moves_one_hot = keras.utils.to_categorical(moves, BOARD_SIZE)

    # Compile and train the model on the training data
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(game_boards, moves_one_hot, epochs=epochs)

def get_valid_moves(board):
    return [i for i in range(BOARD_SIZE) if board[i] == EMPTY]

def play_game():
    board = np.zeros(BOARD_SIZE, dtype=int)
    game_states = []
    moves = []
    player_turn = AI_PLAYER

    while True:
        valid_moves = get_valid_moves(board)

        # Check if the game is over
        if len(valid_moves) == 0 or check_win(board, AI_PLAYER) or check_win(board, OPPONENT):
            break

        # Store the current game state before the move is made
        game_states.append(board.copy())

        # AI player's turn
        if player_turn == AI_PLAYER:
            move = select_move(board)
        # Opponent's turn (AI move)
        else:
            move = select_move(board)

        # Make the move
        board[move] = player_turn

        # Store the move
        moves.append(move)

        # Switch players
        player_turn = OPPONENT if player_turn == AI_PLAYER else AI_PLAYER

    return game_states, moves

# Function to select a move using the neural network
def select_move(board):
    valid_moves = [i for i in range(BOARD_SIZE) if board[i] == EMPTY]

    if not valid_moves:
        return None

    best_move = None
    best_score = float('-inf')

    for move in valid_moves:
        board[move] = AI_PLAYER

        if check_win(board, AI_PLAYER):
            board[move] = EMPTY
            return move

        score = calculate_score(board)
        board[move] = EMPTY

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

# Function to calculate the score of a board configuration
def calculate_score(board):
    if check_win(board, AI_PLAYER):
        return 1  # AI wins, so assign a score of 1
    elif check_win(board, OPPONENT):
        return -1  # Opponent wins, so assign a score of -1

    # Count the number of empty cells
    num_empty_cells = len(get_valid_moves(board))

    if num_empty_cells == 0:
        return 0  # It's a draw, so assign a score of 0

    ai_score = evaluate_position(board, AI_PLAYER)
    opponent_score = evaluate_position(board, OPPONENT)

    return (ai_score - opponent_score) / num_empty_cells

def evaluate_position(board, player):
    winning_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]  # Diagonals
    ]
    score = 0

    for positions in winning_positions:
        markers = [board[pos] for pos in positions]
        if markers.count(player) == 2 and markers.count(-player) == 0:
            score += 0.5  # Player has 2 out of 3 markers in a winning position
        elif markers.count(player) == 1 and markers.count(-player) == 0:
            score += 0.1  # Player has 1 out of 3 markers in a winning position

    return score


def opponent_score(board):
    # Check if the opponent has won
    if not check_win(board, OPPONENT):
        return -1  # Opponent wins, so assign a score of -1

    # Count the number of empty cells
    num_empty_cells = len([cell for cell in board if cell == EMPTY])

    # Assign scores based on the number of opponent's markers in winning positions
    opponent_winning_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]  # Diagonals
    ]
    opponent_score = 0

    for positions in opponent_winning_positions:
        markers = [board[pos] for pos in positions]
        if markers.count(OPPONENT) == 2 and markers.count(AI_PLAYER) == 0:
            opponent_score += 0.5  # Opponent has 2 out of 3 markers in a winning position
        elif markers.count(OPPONENT) == 1 and markers.count(AI_PLAYER) == 0:
            opponent_score += 0.1  # Opponent has 1 out of 3 markers in a winning position

    # Normalize the opponent's score by the number of empty cells
    return opponent_score / num_empty_cells

# Function to check if a player wins
def check_win(board, player):
    # Check rows
    for i in range(0, BOARD_SIZE, 3):
        if board[i] == board[i+1] == board[i+2] == player:
            return True
    
    # Check columns
    for i in range(3):
        if board[i] == board[i+3] == board[i+6] == player:
            return True
    
    # Check diagonals
    if board[0] == board[4] == board[8] == player:
        return True
    if board[2] == board[4] == board[6] == player:
        return True
    
    return False

def self_play(num_games, batch_size, epochs):
    training_data = []

    for game in range(num_games):
        game_states, moves = play_game()

        if len(game_states) != len(moves):
            print(f"Skipping game {game + 1}/{num_games}. Invalid game state.")
            continue

        training_data.extend(list(zip(game_states, moves)))
        print(f"Game {game + 1}/{num_games} completed.")

        if (game + 1) % batch_size == 0:
            train_model(training_data, epochs)
            training_data = []

    if training_data:
        train_model(training_data, epochs)

    return training_data

# Function to check if the board is full
def is_board_full(board):
    return all(cell != EMPTY for cell in board)

# Generate training data
num_games = 1000
batch_size = 256
epochs = 10
training_data = self_play(num_games, batch_size, epochs)

# Separate game states and moves
game_states, moves = zip(*training_data)
game_states = np.array(game_states)
moves = np.array(moves)

# Convert moves to one-hot encoded vectors
move_vectors = keras.utils.to_categorical(moves, num_classes=BOARD_SIZE)

# Train the neural network with the generated training data
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(game_states, move_vectors, epochs=100)

# Test the trained AI against a human user
def play_against_ai():
    while True:
        board = np.zeros(BOARD_SIZE, dtype=int)
        game_over = False
        training_completed = False

        while not game_over:
            if not training_completed:
                move = select_move(board)
                board[move] = AI_PLAYER
            else:
                user_move = get_user_move(board)
                if user_move == 100:
                    print("Game ended.")
                    game_over = True
                    break
                board[user_move] = OPPONENT

            print("AI Player Move:")
            # Display the board
            for i in range(0, BOARD_SIZE, 3):
                print("|".join(["X" if cell == AI_PLAYER else " " if cell == EMPTY else "O" for cell in board[i:i + 3]]))

            if check_win(board, AI_PLAYER):
                print("AI Player wins!")
                game_over = True
                break

            if is_board_full(board):
                print("It's a tie!")
                game_over = True
                break

            if not training_completed:
                if check_win(board, AI_PLAYER):
                    print("AI Player wins!")
                    game_over = True
                    break

                if is_board_full(board):
                    print("It's a tie!")
                    game_over = True
                    break

                user_move = get_user_move(board)
                if user_move == 100:
                    print("Game ended.")
                    game_over = True
                    break
                board[user_move] = OPPONENT

                if check_win(board, OPPONENT):
                    game_over = True
                    # Display the board
                    for i in range(0, BOARD_SIZE, 3):
                        print("|".join(["X" if cell == AI_PLAYER else " " if cell == EMPTY else "O" for cell in board[i:i + 3]]))
                    print("Opponent wins!")
                    break

                if is_board_full(board):
                    game_over = True
                    # Display the board
                    for i in range(0, BOARD_SIZE, 3):
                        print("|".join(["X" if cell == AI_PLAYER else " " if cell == EMPTY else "O" for cell in board[i:i + 3]]))
                    print("It's a tie!")
                    break
            else:
                if check_win(board, OPPONENT):
                    print("Opponent wins!")
                    game_over = True
                    break

                if is_board_full(board):
                    print("It's a tie!")
                    game_over = True
                    break

                move = select_move(board)
                board[move] = AI_PLAYER
                time.sleep(1)  # Add a delay of one second after AI's move

        if not training_completed:
            print("Training completed.")
            training_completed = True

# Play against the trained AI
play_against_ai()
