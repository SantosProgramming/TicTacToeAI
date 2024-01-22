# Tic-Tac-Toe AI with Neural Network

## Overview

This repository contains a Python implementation of a Tic-Tac-Toe game with an AI opponent powered by a neural network. The AI has been trained through self-play to make strategic moves and provides a challenging opponent for human players.

![Tic-Tac-Toe AI](/image.png)

## Requirements

- Python 3
- TensorFlow (for the neural network)
- numpy

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/SantosProgramming/TicTacToeAI
cd TicTacToeAI
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the game:

```bash
python tictactoe.py
```

## How to Play

- Enter your move by specifying a number from 1 to 9 corresponding to the positions on the board.
- You can end the game at any time by entering '101'.
- The AI will display its move, and the game board will be updated accordingly.

## Neural Network Architecture

The AI utilizes a neural network with the following architecture:

```python
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(BOARD_SIZE,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(BOARD_SIZE, activation='softmax')
])
```

## Training the Model

The model is trained through self-play using the following functions:

- `self_play(num_games, batch_size, epochs)`: Generates training data by simulating games between the AI and itself.
- `train_model(training_data, epochs)`: Trains the neural network with the generated training data.

Adjust the `num_games`, `batch_size`, and `epochs` parameters to customize the training process.

## Playing Against the AI

Run the `play_against_ai()` function to engage in a game against the trained AI. The AI will make strategic moves based on its training.

Enjoy playing Tic-Tac-Toe against the AI!
