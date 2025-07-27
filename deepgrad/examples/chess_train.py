import random
import chess
import chess.pgn
import io
import math
from deepgrad.tensor import Tensor
from deepgrad.optim import Adam
from deepgrad.model import ChessNet
from tqdm import tqdm

# --- Model Wrapper ---
class Model:
    def __init__(self):
        self.model = ChessNet()
    
    def __call__(self, x):
        return self.model(x)
    
    def parameters(self):
        return self.model.parameters()
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()

class ChessTrainer:
    def __init__(self, games_as_tensors, batch_size=32, learning_rate=0.001):
        self.games_as_tensors = games_as_tensors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = Model()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def encode_result(self, result):
        """
        Encode the result of the game into a numeric value for classification:
        - 0: Draw
        - 1: White Win
        - 2: Black Win
        """
        if result == '1-0':
            return 1  # White win
        elif result == '0-1':
            return 2  # Black win
        return 0  # Draw

    def fen_to_tensor(self, fen):
        """
        Convert a FEN string to a 12x8x8 tensor for 12 piece channels (6 piece types x 2 colors).
        Returns a flattened list of shape (12*8*8) with floats.
        """
        board = chess.Board(fen)
        # Initialize 12 channels: 6 for white (pawn, knight, bishop, rook, queen, king), 6 for black
        tensor = [[[0.0 for _ in range(8)] for _ in range(8)] for _ in range(12)]
        piece_map = {
            (chess.PAWN, chess.WHITE): 0,   # White pawn channel
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,   # Black pawn channel
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece_map[(piece.piece_type, piece.color)]
                rank = 7 - (square // 8)  # Convert square to rank (0-7, flipped for white's perspective)
                file = square % 8         # Convert square to file (0-7)
                tensor[channel][rank][file] = 1.0  # Mark piece presence
        # Flatten to a 1D list for Tensor compatibility
        flat_tensor = [item for channel in tensor for row in channel for item in row]
        return flat_tensor

    def accuracy(self, predictions, targets):
        """
        Computes the accuracy of the predictions compared to the targets.
        
        Args:
            predictions (Tensor): Predicted outputs, shape (batch_size, output_size).
            targets (Tensor): True labels, shape (batch_size,).
            
        Returns:
            Tensor: The accuracy (percentage of correct predictions).
        """
        correct = 0
        for i in range(predictions.shape[0]):
            # Find the index of the maximum value in the prediction row
            pred_row = predictions.data[i * 3:(i + 1) * 3]  # Assuming output_size=3
            max_idx = 0
            max_val = pred_row[0]
            for j in range(1, 3):
                if pred_row[j] > max_val:
                    max_val = pred_row[j]
                    max_idx = j
            # Compare with target
            if max_idx == int(targets.data[i]):
                correct += 1
        accuracy = correct / predictions.shape[0]
        return Tensor([accuracy])

    def train_epoch(self):
        """
        Train for one epoch using the dataset.
        """
        if len(self.games_as_tensors) == 0:
            raise ValueError("No games to train on.")
        
        self.model.train()
        random.shuffle(self.games_as_tensors)
        
        for i in tqdm(range(0, len(self.games_as_tensors), self.batch_size), desc="Training epoch"):
            batch = self.games_as_tensors[i:i + self.batch_size]
            inputs = [game['fen_tensor'] for game in batch]  # List of flattened (12*8*8) lists
            targets = [self.encode_result(game['result']) for game in batch]
            
            # Flatten the list of lists into a single list for the batch
            flat_inputs = []
            for input_tensor in inputs:
                flat_inputs.extend(input_tensor)  # Combine all 768-element lists
            batch_size = len(batch)
            # Create 4D tensor: (batch_size, 12, 8, 8)
            inputs_tensor = Tensor(flat_inputs, shape=(batch_size, 12, 8, 8), requires_grad=False)
            targets_tensor = Tensor(targets, shape=(batch_size,), requires_grad=False)
            
            outputs = self.model(inputs_tensor)
            loss = outputs.cross_entropy(targets_tensor)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad_c()
            
            accuracy = self.accuracy(outputs, targets_tensor)

    def evaluate(self, max_games=500):
        """
        Evaluate the model on the dataset with a limit on the number of evaluation games.
        """
        if len(self.games_as_tensors) == 0:
            raise ValueError("No valid games to evaluate.")
        
        self.model.eval()
        correct = 0
        total = 0
        
        for game in tqdm(self.games_as_tensors[:max_games], desc="Evaluating"):
            inputs = game['fen_tensor']  # Single flattened (12*8*8) list
            target = [self.encode_result(game['result'])]
            
            # Create 4D tensor: (1, 12, 8, 8)
            inputs_tensor = Tensor(inputs, shape=(1, 12, 8, 8), requires_grad=False)
            target_tensor = Tensor(target, shape=(1,), requires_grad=False)
            outputs = self.model(inputs_tensor)
            # Find the index of the maximum value in the output
            pred_row = outputs.data  # Shape (1, 3) -> flat list of 3 elements
            max_idx = 0
            max_val = pred_row[0]
            for j in range(1, 3):
                if pred_row[j] > max_val:
                    max_val = pred_row[j]
                    max_idx = j
            # Compare with target
            if max_idx == int(target_tensor.data[0]):
                correct += 1
            total += 1
        
        if total > 0:
            accuracy = correct / total
            tqdm.write(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
        else:
            tqdm.write("No games were processed during evaluation.")

    def train(self, epochs=10, max_train_games=5000, max_eval_games=500):
        """
        Train the model for a given number of epochs and evaluate after each epoch.
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            self.train_epoch()
            self.evaluate(max_games=max_eval_games)

def parse_pgn(pgn_file_path, max_games=5000):
    """
    Parse PGN file and extract final board state and result, limiting to max_games.
    """
    games_as_tensors = []
    with open(pgn_file_path, 'r') as pgn_file:
        pgn_data = pgn_file.read()
    pgn = io.StringIO(pgn_data)
    
    trainer = ChessTrainer([], batch_size=32)  # Temporary instance for fen_to_tensor
    game_count = 0
    
    with tqdm(desc="Parsing PGN games", total=max_games) as pbar:
        while game_count < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            result = game.headers.get("Result", "*")
            if result not in ["1-0", "0-1", "1/2-1/2"]:
                continue  # Skip invalid games
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            fen = board.fen()
            games_as_tensors.append({
                'fen_tensor': trainer.fen_to_tensor(fen),
                'result': result
            })
            game_count += 1
            pbar.update(1)
    
    return games_as_tensors

def main():
    """
    Main function to run the training.
    """
    pgn_file_path = 'deepgrad/examples/datasets/lichess.pgn'
    games_as_tensors = parse_pgn(pgn_file_path, max_games=5000)
    print(f"Total valid games: {len(games_as_tensors)}")
    
    trainer = ChessTrainer(games_as_tensors, batch_size=32)
    trainer.train(epochs=10)

if __name__ == "__main__":
    main()
