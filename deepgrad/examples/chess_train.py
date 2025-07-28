import random
import chess
import math
from tqdm import tqdm
from deepgrad.tensor import Tensor
from deepgrad.batchnorm import BatchNorm2D
from deepgrad.optim import Adam
import dill

# Board Representation
def board_to_tensor(board):
    """
    Convert a chess board to a tensor of shape (12, 8, 8) from the perspective of the player to move.
    12 channels: 6 piece types (P, N, B, R, Q, K) x 2 colors (white, black).
    Flips the board if Black is to move.
    """
    data = [0.0] * (12 * 8 * 8)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        rank = square // 8
        file = square % 8
        rank = 7 - rank if board.turn == chess.BLACK else rank
        color_idx = 0 if piece.color == chess.WHITE else 6
        piece_idx = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}[piece.symbol().upper()]
        if board.turn == chess.BLACK:
            color_idx = 6 - color_idx if color_idx < 6 else color_idx - 6
        channel = color_idx + piece_idx
        idx = channel * 64 + rank * 8 + file
        data[idx] = 1.0
    return Tensor(data, shape=(12, 8, 8), requires_grad=False)

# Move Encoding (AlphaZero-style 73 planes)
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # N, NE, E, SE, S, SW, W, NW
KNIGHT_MOVES = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]

def move_to_plane_row_col(board, move):
    """
    Convert a chess move to a (plane, row, col) tuple for AlphaZero's 73-plane policy encoding.
    Handles piece movements, knight moves, and promotions. Flips ranks for Black's perspective.

    Args:
        board: chess.Board object representing the current board state.
        move: chess.Move object representing the move to encode.

    Returns:
        Tuple (plane, row, col) where plane is the policy plane index (0-72),
        row and col are the destination square coordinates (0-7).
    """
    from_sq = move.from_square
    to_sq = move.to_square
    piece = board.piece_at(from_sq)
    from_row, from_col = divmod(from_sq, 8)
    to_row, to_col = divmod(to_sq, 8)
    if not board.turn:  # Black to move, flip ranks
        from_row, to_row = 7 - from_row, 7 - to_row
    delta_row, delta_col = to_row - from_row, to_col - from_col
    distance = max(abs(delta_row), abs(delta_col))

    if piece.piece_type == chess.KNIGHT:
        for i, (dr, dc) in enumerate(KNIGHT_MOVES):
            if (dr, dc) == (delta_row, delta_col):
                return 56 + i, to_row, to_col
    elif move.promotion and piece.piece_type == chess.PAWN:
        promotion_piece = move.promotion
        direction = 0 if delta_col == 0 else (-1 if delta_col < 0 else 1)
        base_plane = {'N': 64, 'B': 67, 'R': 70}.get(chess.piece_symbol(promotion_piece).upper(), 0)
        if base_plane == 0:  # Queen promotion uses sliding planes
            for i, (dr, dc) in enumerate(DIRECTIONS):
                if distance > 0 and (dr, dc) == (delta_row // distance, delta_col // distance):
                    return i * 7 + (distance - 1), to_row, to_col
        else:
            plane = base_plane + (1 if direction < 0 else 2 if direction > 0 else 0)
            return plane, to_row, to_col
    else:
        for i, (dr, dc) in enumerate(DIRECTIONS):
            if distance > 0 and (dr, dc) == (delta_row // distance, delta_col // distance):
                return i * 7 + (distance - 1), to_row, to_col
    return 0, to_row, to_col  # Default for castling or edge cases

def get_policy_index(board, move):
    plane, row, col = move_to_plane_row_col(board, move)
    return plane * 64 + row * 8 + col

def get_legal_indices(board):
    indices = [get_policy_index(board, move) for move in board.legal_moves]
    return indices

# Neural Network
class ChessNet:
    def __init__(self):
        self.training = True
        self.num_planes = 73  # AlphaZero's 73-plane encoding
        self.num_moves = self.num_planes * 64  # 73 * 8 * 8
        self.w1 = Tensor.randn((64, 12, 3, 3), std=math.sqrt(2 / (12 * 3 * 3)), requires_grad=True)
        self.b1 = Tensor.zeros((64,), requires_grad=True)
        self.bn1 = BatchNorm2D(64)
        self.w2 = Tensor.randn((128, 64, 3, 3), std=math.sqrt(2 / (64 * 3 * 3)), requires_grad=True)
        self.b2 = Tensor.zeros((128,), requires_grad=True)
        self.bn2 = BatchNorm2D(128)
        self.policy_w = Tensor.randn((128 * 2 * 2, self.num_moves), std=math.sqrt(2 / (128 * 2 * 2)), requires_grad=True)
        self.policy_b = Tensor.zeros((1, self.num_moves), requires_grad=True)
        self.value_w1 = Tensor.randn((128 * 2 * 2, 512), std=math.sqrt(2 / (128 * 2 * 2)), requires_grad=True)
        self.value_b1 = Tensor.zeros((1, 512), requires_grad=True)
        self.value_w2 = Tensor.randn((512, 1), std=math.sqrt(2 / 512), requires_grad=True)
        self.value_b2 = Tensor.zeros((1, 1), requires_grad=True)

    def __call__(self, x: Tensor):
        x = x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(1, 1))
        x = self.bn1(x).relu().maxpool2d(kernel_size=2, stride=2)
        x = x.conv2d(self.w2, self.b2, stride=(1, 1), padding=(1, 1))
        x = self.bn2(x).relu().maxpool2d(kernel_size=2, stride=2)
        x = x.flatten(start_dim=1)
        policy = x.matmul(self.policy_w) + self.policy_b
        value = (x.matmul(self.value_w1) + self.value_b1).relu()
        value = (value.matmul(self.value_w2) + self.value_b2).tanh()
        return policy, value

    def parameters(self):
        return [
            self.w1, self.b1, self.w2, self.b2,
            self.policy_w, self.policy_b,
            self.value_w1, self.value_b1, self.value_w2, self.value_b2,
            *self.bn1.parameters(), *self.bn2.parameters()
        ]

    def train(self):
        self.training = True
        self.bn1.training = True
        self.bn2.training = True

    def eval(self):
        self.training = False
        self.bn1.training = False
        self.bn2.training = False

# Model Wrapper
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

# Monte Carlo Tree Search
class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.N = 0  # Visit count
        self.W = 0.0  # Total value
        self.P = 0.0  # Prior probability

class MCTS:
    def __init__(self, model, num_simulations=50, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, board):
        root = MCTSNode(board.copy())
        for sim in tqdm(range(self.num_simulations), desc="MCTS simulations", leave=False):
            node = self.select(root)
            value = self.simulate(node)
            self.backpropagate(node, value)
        move_probs = {}
        for move, child in root.children.items():
            move_probs[move] = child.N / root.N if root.N > 0 else 0.0
        return move_probs

    def select(self, node):
        while node.children and all(move in node.children for move in node.board.legal_moves):
            node = max(node.children.values(), key=lambda n: self.ucb(n))
        if not node.board.is_game_over():
            legal_moves = list(node.board.legal_moves)
            if not legal_moves:
                return node
            if not node.children:
                policy, value = self.evaluate(node.board)
                for move, p in zip(legal_moves, policy):
                    child = MCTSNode(node.board.copy(), node, move)
                    child.P = p
                    node.children[move] = child
                return node
            move = random.choice([m for m in legal_moves if m not in node.children])
            child = MCTSNode(node.board.copy(), node, move)
            node.children[move] = child
            node = child
            node.board.push(move)
        return node

    def ucb(self, node):
        Q = node.W / node.N if node.N > 0 else 0
        return Q + self.c_puct * node.P * math.sqrt(node.parent.N) / (1 + node.N)

    def evaluate(self, board):
        tensor = board_to_tensor(board)
        inputs = Tensor(tensor.data, shape=(1, 12, 8, 8), requires_grad=False)
        self.model.eval()
        policy_logits, value = self.model(inputs)
        legal_indices = get_legal_indices(board)
        policy = [policy_logits.data[idx] for idx in legal_indices]
        if not policy:
            logsumexp_legal = float('-inf')
        else:
            max_val = max(policy)
            exp_sum = sum(math.exp(x - max_val) for x in policy)
            logsumexp_legal = max_val + math.log(exp_sum) if exp_sum > 0 else float('-inf')
        policy = [math.exp(p - logsumexp_legal) for p in policy]
        return policy, value.data[0]

    def simulate(self, node):
        return self.evaluate(node.board)[1]
    
    def backpropagate(self, node, value):
        sign = 1
        current = node
        while current:
            current.N += 1
            current.W += sign * value
            sign = -sign
            current = current.parent

# AlphaZero Trainer
class ChessTrainer:
    def __init__(self, batch_size=32, num_games=5, num_simulations=50, learning_rate=0.001):
        self.model = Model()
        self.batch_size = batch_size
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.mcts = MCTS(self.model, num_simulations)
        self.memory = []
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

    def self_play(self):
        for game_idx in tqdm(range(self.num_games), desc="Self-play games"):
            board = chess.Board()
            game_data = []
            move_count = 0
            max_moves = 100
            while not board.is_game_over() and move_count < max_moves:
                move_probs = self.mcts.search(board)
                moves = list(move_probs.keys())
                probs = list(move_probs.values())
                if not moves:
                    break
                move = random.choices(moves, weights=probs, k=1)[0]
                game_data.append((board_to_tensor(board), move_probs, board.copy()))
                board.push(move)
                move_count += 1
            outcome = 1 if board.result() == "1-0" else -1 if board.result() == "0-1" else 0
            for tensor, probs, board in game_data:
                self.memory.append((tensor, probs, outcome, board))
            if len(self.memory) > 10000:
                self.memory = self.memory[-10000:]

    def train(self, num_iterations=10):
        for iteration in tqdm(range(num_iterations), desc="Training iterations"):
            self.self_play()
            if not self.memory:
                continue
            random.shuffle(self.memory)
            for i in tqdm(range(0, len(self.memory), self.batch_size), desc="Training batches"):
                batch = self.memory[i:i + self.batch_size]
                batch_size = len(batch)
                input_data = [d[0].data for d in batch]
                flat_data = []
                for tensor_data in input_data:
                    flat_data.extend(tensor_data)
                inputs = Tensor(flat_data, shape=(batch_size, 12, 8, 8))
                
                target_values = []
                for d in batch:
                    value = d[2] if d[3].turn == chess.WHITE else -d[2]
                    target_values.append(value)
                target_values = Tensor(target_values, shape=(batch_size, 1))
                
                target_indices = []
                for _, probs, _, board in batch:
                    best_move_index = 0
                    max_prob = -float('inf')
                    for move, p in probs.items():
                        idx = get_policy_index(board, move)
                        if p > max_prob:
                            max_prob = p
                            best_move_index = idx
                    target_indices.append(best_move_index)
                target_indices = Tensor(target_indices, shape=(batch_size,), size=batch_size)

                self.model.train()
                policy_logits, value = self.model(inputs)
                policy_loss = policy_logits.cross_entropy(target_indices)
                value_loss = ((value - target_values) ** 2).mean()
                loss = policy_loss + value_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad_c()

            self.save_model(iteration)

            # Policy Accuracy Evaluation
            correct = 0
            total = 0
            for tensor, probs, _, board in self.memory:
                input_tensor = Tensor(tensor.data, shape=(1, 12, 8, 8), requires_grad=False)
                self.model.eval()
                policy_logits, _ = self.model(input_tensor)
                predicted_index = 0
                max_value = policy_logits.data[0]
                for i in range(1, len(policy_logits.data)):
                    if policy_logits.data[i] > max_value:
                        max_value = policy_logits.data[i]
                        predicted_index = i
                best_move_index = 0
                max_prob = -float('inf')
                for move, p in probs.items():
                    idx = get_policy_index(board, move)
                    if p > max_prob:
                        max_prob = p
                        best_move_index = idx
                if predicted_index == best_move_index:
                    correct += 1
                total += 1

            if total > 0:
                acc = correct / total
                tqdm.write(f"[Iteration {iteration + 1}] Policy Accuracy: {acc:.4f}")

    def select_move(self, board):
        self.model.eval()
        move_probs = self.mcts.search(board)
        return max(move_probs.items(), key=lambda x: x[1])[0]

    def save_model(self, iteration):
        with open(f"chess_model_iter_{iteration}.pkl", "wb") as f:
            dill.dump(self.model, f)

def play_game(trainer):
    board = chess.Board()
    move_count = 0
    max_moves = 100
    while not board.is_game_over() and move_count < max_moves:
        move = trainer.select_move(board)
        board.push(move)
        move_count += 1

def main():
    trainer = ChessTrainer(batch_size=32, num_games=10, num_simulations=25, learning_rate=0.001)
    trainer.train(num_iterations=10)
    play_game(trainer)

if __name__ == "__main__":
    main()