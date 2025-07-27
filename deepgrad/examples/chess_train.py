import random
import chess
import chess.pgn
import dill
import io
from deepgrad.batchnorm import BatchNorm2D
from deepgrad.tensor import Tensor
from deepgrad.optim import Adam
from tqdm import tqdm
import math
from collections import Counter


def board_to_tensor(board):
    """
    Convert a chess board to a tensor of shape (12, 8, 8) representing piece positions.
    12 channels: 6 piece types (P, N, B, R, Q, K) x 2 colors (white, black).
    """
    data = [0.0] * (12 * 8 * 8)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        rank = 7 - (square // 8)  # Convert to 0-7 (flipped for white's perspective)
        file = square % 8
        color_idx = 0 if piece.color == chess.WHITE else 6
        piece_idx = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}[piece.symbol().upper()]
        channel = color_idx + piece_idx
        idx = channel * 8 * 8 + rank * 8 + file
        data[idx] = 1.0
    
    tensor = Tensor(data, shape=(12, 8, 8), requires_grad=False)
    return tensor

class ChessNet:
    def __init__(self, num_legal_moves):
        self.training = True
        
        # Conv1: 12 → 64, 3x3
        self.w1 = Tensor.randn((64, 12, 3, 3), std=math.sqrt(2 / (12 * 3 * 3)), requires_grad=True)
        self.b1 = Tensor.zeros((64,), requires_grad=True)
        self.bn1 = BatchNorm2D(64)

        # Conv2: 64 → 128, 3x3
        self.w2 = Tensor.randn((128, 64, 3, 3), std=math.sqrt(2 / (64 * 3 * 3)), requires_grad=True)
        self.b2 = Tensor.zeros((128,), requires_grad=True)
        self.bn2 = BatchNorm2D(128)

        # FC1: 128 * 2 * 2 → 512
        self.w3 = Tensor.randn((128 * 2 * 2, 512), std=math.sqrt(2 / (128 * 2 * 2)), requires_grad=True)
        self.b3 = Tensor.zeros((1, 512), requires_grad=True)

        # FC2: 512 → num_legal_moves
        self.w4 = Tensor.randn((512, num_legal_moves), std=math.sqrt(2 / 512), requires_grad=True)
        self.b4 = Tensor.zeros((1, num_legal_moves), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(1, 1))
        x = self.bn1(x).relu().maxpool2d(kernel_size=2, stride=2)
        x = x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(1, 1))
        x = self.bn2(x).relu().maxpool2d(kernel_size=2, stride=2)
        x = x.flatten(start_dim=1)
        x = (x.matmul(self.w3) + self.b3).relu()
        return x.matmul(self.w4) + self.b4

    def parameters(self):
        return [
            self.w1, self.b1,
            self.w2, self.b2,
            self.w3, self.b3,
            self.w4, self.b4,
            *self.bn1.parameters(),
            *self.bn2.parameters(),
        ]

    def train(self):
        self.training = True
        self.bn1.training = True
        self.bn2.training = True

    def eval(self):
        self.training = False
        self.bn1.training = False
        self.bn2.training = False

class Model:
    def __init__(self, num_legal_moves):
        self.model = ChessNet(num_legal_moves)
    
    def __call__(self, x):
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

class ChessTrainer:
    def __init__(self, games_as_tensors, legal_moves_list, batch_size=32, learning_rate=0.001):
        #print(f"Initializing ChessTrainer with {len(games_as_tensors)} games, {len(legal_moves_list)} moves")
        self.games_as_tensors = games_as_tensors
        self.legal_moves_list = legal_moves_list
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = Model(len(legal_moves_list))
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def encode_move(self, move, legal_moves):
        try:
            return legal_moves.index(move)
        except ValueError:
            raise ValueError(f"Move {move} not found in legal moves list.")

    def train_epoch(self, epoch):
        if not self.games_as_tensors:
            raise ValueError("No games to train on.")
        
        self.model.train()
        random.shuffle(self.games_as_tensors)

        for i in tqdm(range(0, len(self.games_as_tensors), self.batch_size), desc=f"Training epoch {epoch}"):
            batch = self.games_as_tensors[i:i + self.batch_size]
            inputs = [game['fen_tensor'] for game in batch]
            targets = [self.encode_move(game['move'], self.legal_moves_list) for game in batch]

            # Manually concatenate input tensors
            batch_size = len(inputs)
            flat_data = []
            for tensor in inputs:
                if tensor.shape != (12, 8, 8):
                    raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
                flat_data.extend(tensor.data)  # Assuming tensor.data is a flat list
            inputs_tensor = Tensor(flat_data, shape=(batch_size, 12, 8, 8), requires_grad=False)
            targets_tensor = Tensor(targets, shape=(batch_size,), requires_grad=False)

            outputs = self.model(inputs_tensor)
            loss = outputs.cross_entropy(targets_tensor)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad_c()

    def evaluate(self, max_games=500):
        if not self.games_as_tensors:
            raise ValueError("No valid games to evaluate.")
        
        self.model.eval()
        correct = 0
        total = 0

        for game in tqdm(self.games_as_tensors[:max_games], desc="Evaluating"):
            inputs = game['fen_tensor']
            target = self.encode_move(game['move'], self.legal_moves_list)

            inputs_tensor = Tensor(inputs.data, shape=(1, 12, 8, 8), requires_grad=False)
            outputs = self.model(inputs_tensor)
            
            # Manually compute argmax on outputs.data
            output_data = list(outputs.data)  # Convert c_float_Array to Python list
            pred_idx = max(range(len(output_data)), key=lambda i: output_data[i])

            if pred_idx == target:
                correct += 1
            total += 1

        if total > 0:
            accuracy = correct / total
            tqdm.write(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
        else:
            tqdm.write("No games were processed during evaluation.")

    def save_model(self, epoch):
        model_filename = f"chess_model_epoch_{epoch}.pkl"
        with open(model_filename, "wb") as f:
            dill.dump(self.model, f)
        tqdm.write(f"Model saved to {model_filename}")

    def train(self, epochs=10):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            self.evaluate()
            self.save_model(epoch)

def parse_pgn(pgn_file_path, max_games=500, max_legal_moves=1000):
    """
    Parse PGN file and extract board states and moves, limiting to max_games.
    Limit legal moves to the most frequent max_legal_moves to reduce memory usage.
    """
    games_as_tensors = []
    legal_moves_counter = Counter()
    with open(pgn_file_path, 'r') as pgn_file:
        pgn_data = pgn_file.read()
    pgn = io.StringIO(pgn_data)

    game_count = 0
    with tqdm(desc="Parsing PGN games", total=max_games) as pbar:
        while game_count < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            result = game.headers.get("Result", "*")
            if result not in ["1-0", "0-1", "1/2-1/2"]:
                continue
            board = game.board()
            for move in game.mainline_moves():
                fen_tensor = board_to_tensor(board)
                move_str = str(move)
                games_as_tensors.append({'fen_tensor': fen_tensor, 'move': move_str})
                legal_moves_counter[move_str] += 1
                board.push(move)
            game_count += 1
            pbar.update(1)

    # Limit to the most frequent legal moves to reduce output layer size
    legal_moves_list = [move for move, _ in legal_moves_counter.most_common(max_legal_moves)]
    #print(f"Total games parsed: {len(games_as_tensors)}")
    #print(f"Total unique moves: {len(legal_moves_counter)}")
    #print(f"Limited to top {len(legal_moves_list)} legal moves")

    # Filter games to only include moves in legal_moves_list
    filtered_games = [game for game in games_as_tensors if game['move'] in legal_moves_list]
    #print(f"Filtered games with allowed moves: {len(filtered_games)}")
    return filtered_games, legal_moves_list

def main():
    """Main function to run the training."""
    pgn_file_path = 'deepgrad/examples/datasets/lichess.pgn'
    games_as_tensors, legal_moves_list = parse_pgn(pgn_file_path, max_games=500, max_legal_moves=1000)
    trainer = ChessTrainer(games_as_tensors, legal_moves_list, batch_size=32)
    trainer.train(epochs=10)

if __name__ == "__main__":
    main()