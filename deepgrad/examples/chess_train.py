import random, math, os, dill, copy
import chess
from tqdm import tqdm
from deepgrad.tensor import Tensor
from deepgrad.batchnorm import BatchNorm2D
from deepgrad.optim import Adam
import numpy as np

# --- Board Encoding and Symmetry Augmentation ---
def board_to_tensor(board, augment=False):
    data = [0.0] * (12 * 8 * 8)
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        rank = 7 - rank if board.turn == chess.BLACK else rank
        color_offset = 0 if piece.color == chess.WHITE else 6
        piece_type = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}[piece.symbol().upper()]
        channel = color_offset + piece_type
        data[channel * 64 + rank * 8 + file] = 1.0
    tensor = Tensor(data, shape=(12, 8, 8), requires_grad=False)
    
    if augment and random.random() < 0.5:
        # Horizontal flip
        data = np.array(data).reshape(12, 8, 8)
        data = np.flip(data, axis=2).flatten()
        tensor = Tensor(data.tolist(), shape=(12, 8, 8), requires_grad=False)
    return tensor

# --- Policy Indexing (AlphaZero style) ---
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KNIGHT_MOVES = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]

def move_to_plane_row_col(board, move, flipped=False):
    from_sq, to_sq = move.from_square, move.to_square
    from_row, from_col = divmod(from_sq, 8)
    to_row, to_col = divmod(to_sq, 8)
    if not board.turn:
        from_row, to_row = 7 - from_row, 7 - to_row
    if flipped:
        from_col, to_col = 7 - from_col, 7 - to_col
    delta_row, delta_col = to_row - from_row, to_col - from_col
    distance = max(abs(delta_row), abs(delta_col))

    if board.piece_at(from_sq).piece_type == chess.KNIGHT:
        for i, (dr, dc) in enumerate(KNIGHT_MOVES):
            if (dr, dc) == (delta_row, delta_col):
                return 56 + i, to_row, to_col
    else:
        for i, (dr, dc) in enumerate(DIRECTIONS):
            if distance > 0 and (dr, dc) == (delta_row // distance, delta_col // distance):
                return i * 7 + (distance - 1), to_row, to_col
    return 0, to_row, to_col

def get_policy_index(board, move, flipped=False):
    plane, row, col = move_to_plane_row_col(board, move, flipped)
    if flipped:
        col = 7 - col
    return plane * 64 + row * 8 + col

# --- Model (ResNet-style) ---
class ChessNet:
    def __init__(self, num_blocks=10, num_filters=128):
        self.num_filters = num_filters
        # Initial conv layer
        self.w1 = Tensor.randn((num_filters, 12, 3, 3), std=0.1, requires_grad=True)
        self.b1 = Tensor.zeros((num_filters,), requires_grad=True)
        self.bn1 = BatchNorm2D(num_filters)
        
        # Residual blocks
        self.res_blocks = []
        for _ in range(num_blocks):
            w1 = Tensor.randn((num_filters, num_filters, 3, 3), std=0.1, requires_grad=True)
            b1 = Tensor.zeros((num_filters,), requires_grad=True)
            bn1 = BatchNorm2D(num_filters)
            w2 = Tensor.randn((num_filters, num_filters, 3, 3), std=0.1, requires_grad=True)
            b2 = Tensor.zeros((num_filters,), requires_grad=True)
            bn2 = BatchNorm2D(num_filters)
            self.res_blocks.append((w1, b1, bn1, w2, b2, bn2))
        
        # Policy head (adjusted to match feature map size after pooling)
        self.policy_conv = Tensor.randn((num_filters, num_filters, 1, 1), std=0.1, requires_grad=True)
        self.policy_b = Tensor.zeros((num_filters,), requires_grad=True)
        self.policy_bn = BatchNorm2D(num_filters)
        self.policy_w = Tensor.randn((num_filters * 8 * 8, 4672), std=0.1, requires_grad=True)
        self.policy_b2 = Tensor.zeros((4672,), requires_grad=True)
        
        # Value head
        self.value_conv = Tensor.randn((1, num_filters, 1, 1), std=0.1, requires_grad=True)
        self.value_b = Tensor.zeros((1,), requires_grad=True)
        self.value_bn = BatchNorm2D(1)
        self.value_w1 = Tensor.randn((1 * 8 * 8, 512), std=0.1, requires_grad=True)
        self.value_b1 = Tensor.zeros((512,), requires_grad=True)
        self.value_w2 = Tensor.randn((512, 1), std=0.1, requires_grad=True)
        self.value_b2 = Tensor.zeros((1,), requires_grad=True)

    def __call__(self, x):
        # Initial conv
        x = x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(1, 1))
        x = self.bn1(x).relu()
        
        # Residual blocks
        for w1, b1, bn1, w2, b2, bn2 in self.res_blocks:
            residual = x
            x = x.conv2d(w1, b1, stride=(1, 1), padding=(1, 1))
            x = bn1(x).relu()
            x = x.conv2d(w2, b2, stride=(1, 1), padding=(1, 1))
            x = bn2(x)
            x = x + residual
            x = x.relu()
        
        # Policy head
        policy = x.conv2d(self.policy_conv, self.policy_b, stride=(1, 1), padding=(0, 0))
        policy = self.policy_bn(policy).relu()
        policy = policy.flatten(start_dim=1)
        policy = policy.matmul(self.policy_w) + self.policy_b2
        
        # Value head
        value = x.conv2d(self.value_conv, self.value_b, stride=(1, 1), padding=(0, 0))
        value = self.value_bn(value).relu()
        value = value.flatten(start_dim=1)
        value = (value.matmul(self.value_w1) + self.value_b1).relu()
        value = (value.matmul(self.value_w2) + self.value_b2).tanh()
        
        return policy, value

    def parameters(self):
        params = [self.w1, self.b1, self.policy_conv, self.policy_b, self.policy_w, self.policy_b2,
                 self.value_conv, self.value_b, self.value_w1, self.value_b1, self.value_w2, self.value_b2,
                 *self.bn1.parameters(), *self.policy_bn.parameters(), *self.value_bn.parameters()]
        for w1, b1, bn1, w2, b2, bn2 in self.res_blocks:
            params.extend([w1, b1, w2, b2, *bn1.parameters(), *bn2.parameters()])
        return params

    def copy(self):
        return copy.deepcopy(self)

# --- MCTS with Dirichlet Noise ---
class Node:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.Q = 0.0

    def is_expanded(self):
        return bool(self.children)

    def expand(self, model, is_root=False, dirichlet_alpha=0.3):
        tensor = board_to_tensor(self.board)
        policy_logits, value = model(Tensor(tensor.data, shape=(1, 12, 8, 8)))
        policy_logits = policy_logits.data
        legal_moves = list(self.board.legal_moves)
        total_logits = [math.exp(policy_logits[get_policy_index(self.board, m)]) for m in legal_moves]
        total = sum(total_logits) + 1e-8
        priors = [p / total for p in total_logits]

        if is_root:
            dirichlet_noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
            priors = [0.75 * p + 0.25 * n for p, n in zip(priors, dirichlet_noise)]

        for move, prior in zip(legal_moves, priors):
            new_board = self.board.copy()
            new_board.push(move)
            self.children[move] = Node(new_board, parent=self, prior=prior)
        return float(value.data[0])

    def select_child(self, c_puct=1.0):
        if not self.children:
            return None, None  # Handle case with no children
        def ucb_score(child):
            return child.Q + c_puct * child.P * math.sqrt(self.N + 1e-8) / (1 + child.N)
        return max(self.children.items(), key=lambda item: ucb_score(item[1]))

    def backprop(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backprop(-value)

class MCTS:
    def __init__(self, model, sims=100):
        self.model = model
        self.sims = sims

    def run(self, board):
        root = Node(board)
        root.expand(self.model, is_root=True)

        for _ in range(self.sims):
            node = root
            search_path = [node]
            while node.is_expanded():
                move, node = node.select_child()
                search_path.append(node)

            value = node.expand(self.model)
            for n in reversed(search_path):
                n.backprop(value)

        move_visits = {m: child.N for m, child in root.children.items()}
        total_visits = sum(move_visits.values())
        policy = {m: n / total_visits for m, n in move_visits.items()}
        return policy, root

# --- Dataset Loader with Augmentation ---
class ChessDataset:
    def __init__(self, games):
        self.games = games

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        fen, state, policy, value = self.games[idx]
        return state, policy, value

    def get_batch(self, batch_size):
        indices = np.random.choice(len(self.games), batch_size, replace=False)
        states, policies, values = [], [], []
        for idx in indices:
            fen, state, policy, value = self.games[idx]
            flipped = random.random() < 0.5
            states.append(board_to_tensor(chess.Board(fen), augment=flipped).data)
            policies.append(self.encode_policy(chess.Board(fen), policy, flipped))
            values.append(value)
        return (Tensor(np.array(states), shape=(batch_size, 12, 8, 8)),
                Tensor(np.array(policies), shape=(batch_size, 4672)),
                Tensor(np.array(values), shape=(batch_size, 1)))

    def encode_policy(self, board, move_probs, flipped=False):
        arr = [0.0] * 4672
        for move, prob in move_probs.items():
            idx = get_policy_index(board, move, flipped)
            arr[idx] = prob
        return arr

# --- Model Evaluation ---
def evaluate_models(model_new, model_old, num_games=10):
    wins, losses, draws = 0, 0, 0
    for _ in range(num_games):
        board = chess.Board()
        mcts_new = MCTS(model_new, sims=100)
        mcts_old = MCTS(model_old, sims=100)
        while not board.is_game_over():
            policy = (mcts_new if board.turn == chess.WHITE else mcts_old).run(board)[0]
            move = random.choices(list(policy.keys()), weights=list(policy.values()))[0]
            board.push(move)
        result = board.result()
        if result == '1-0':
            wins += 1 if board.turn == chess.BLACK else 0
            losses += 1 if board.turn == chess.WHITE else 0
        elif result == '0-1':
            wins += 1 if board.turn == chess.WHITE else 0
            losses += 1 if board.turn == chess.BLACK else 0
        else:
            draws += 1
    return wins, losses, draws

# --- Training ---
class Trainer:
    def __init__(self):
        self.model = ChessNet(num_blocks=10, num_filters=128)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.games = []
        self.memory_size = 10000
        self.best_model = self.model.copy()
        self.best_model_filename = "deepgrad/examples/models/best_model.pkl"

    def self_play(self, num_games=10, model=None):
        model = model or self.best_model  # Use fixed snapshot
        mcts = MCTS(model, sims=100)
        for _ in tqdm(range(num_games), desc="Self-Play"):
            board = chess.Board()
            game = []
            move_count = 0
            while not board.is_game_over():
                temperature = 1.0 if move_count < 30 else 0.1  # Temperature annealing
                policy, root = mcts.run(board)
                moves, probs = list(policy.keys()), list(policy.values())
                probs = [p ** (1 / temperature) for p in probs]
                total = sum(probs) + 1e-8
                probs = [p / total for p in probs]
                move = random.choices(moves, weights=probs)[0]
                game.append((board.fen(), board_to_tensor(board).data, policy.copy(), None))
                board.push(move)
                move_count += 1
            outcome = 1 if board.result() == '1-0' else -1 if board.result() == '0-1' else 0
            for i, (fen, tensor_data, move_probs, _) in enumerate(game):
                outcome_signed = outcome if chess.Board(fen).turn == chess.WHITE else -outcome
                game[i] = (fen, tensor_data, move_probs, outcome_signed)
            self.games.extend(game)
        if len(self.games) > self.memory_size:
            self.games = self.games[-self.memory_size:]

    def loss_function(self, policy_logits, target_policy, value, target_value):
        batch_size, policy_dim = policy_logits.shape
        exp_logits = Tensor([math.exp(policy_logits.data[i]) for i in range(policy_logits.size)], shape=policy_logits.shape)
        exp_sum = Tensor.zeros((batch_size, 1))
        for b in range(batch_size):
            sum_val = 0.0
            for i in range(policy_dim):
                sum_val += exp_logits.data[b * policy_dim + i]
            exp_sum.data[b] = sum_val
        log_softmax = Tensor([math.log(exp_logits.data[i] / exp_sum.data[i // policy_dim]) for i in range(policy_logits.size)], shape=policy_logits.shape)
        policy_loss = -Tensor.sum(target_policy * log_softmax) / batch_size
        value_loss = ((value - target_value) ** 2).mean()
        return policy_loss + value_loss

    def train(self, epochs=5, batch_size=32, games_per_epoch=10):
        dataset = ChessDataset(self.games)
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            self.self_play(num_games=games_per_epoch, model=self.best_model)
            dataset.games = self.games
            num_batches = len(dataset) // batch_size
            total_loss = 0.0

            for _ in tqdm(range(num_batches), desc=f"Training Epoch {epoch+1}"):
                inputs, target_policy, target_value = dataset.get_batch(batch_size)
                policy_logits, value = self.model(inputs)
                loss = self.loss_function(policy_logits, target_policy, value, target_value)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += float(loss.data[0])

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

            # Evaluate new model against best model
            wins, losses, draws = evaluate_models(self.model, self.best_model, num_games=10)
            print(f"Evaluation: Wins={wins}, Losses={losses}, Draws={draws}")
            if wins > losses:
                self.best_model = self.model.copy()
                self.save_model(self.best_model_filename)
                print("Updated best model")

            self.save_model(f"deepgrad/examples/models/model_epoch_{epoch+1}.pkl")

    def save_model(self, filename):
        def to_list(ctypes_array):
            return list(ctypes_array)
        parameters = {
            'num_blocks': len(self.model.res_blocks),
            'num_filters': self.model.num_filters,
            'w1': to_list(self.model.w1.data),
            'b1': to_list(self.model.b1.data),
            'policy_w': to_list(self.model.policy_w.data),
            'policy_b': to_list(self.model.policy_b.data),
            'value_w1': to_list(self.model.value_w1.data),
            'value_b1': to_list(self.model.value_b1.data),
            'value_w2': to_list(self.model.value_w2.data),
            'value_b2': to_list(self.model.value_b2.data),
            'bn1_gamma': to_list(self.model.bn1.gamma.data),
            'bn1_beta': to_list(self.model.bn1.beta.data),
            'bn1_running_mean': to_list(self.model.bn1.running_mean) if hasattr(self.model.bn1, 'running_mean') else [0.0] * self.model.num_filters,
            'bn1_running_var': to_list(self.model.bn1.running_var) if hasattr(self.model.bn1, 'running_var') else [1.0] * self.model.num_filters,
            'res_blocks': [
                {
                    'w1': to_list(w1.data),
                    'b1': to_list(b1.data),
                    'w2': to_list(w2.data),
                    'b2': to_list(b2.data),
                    'bn1_gamma': to_list(bn1.gamma.data),
                    'bn1_beta': to_list(bn1.beta.data),
                    'bn1_running_mean': to_list(bn1.running_mean) if hasattr(bn1, 'running_mean') else [0.0] * self.model.num_filters,
                    'bn1_running_var': to_list(bn1.running_var) if hasattr(bn1, 'running_var') else [1.0] * self.model.num_filters,
                    'bn2_gamma': to_list(bn2.gamma.data),
                    'bn2_beta': to_list(bn2.beta.data),
                    'bn2_running_mean': to_list(bn2.running_mean) if hasattr(bn2, 'running_mean') else [0.0] * self.model.num_filters,
                    'bn2_running_var': to_list(bn2.running_var) if hasattr(bn2, 'running_var') else [1.0] * self.model.num_filters
                }
                for w1, b1, bn1, w2, b2, bn2 in self.model.res_blocks
            ]
        }
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            dill.dump(parameters, f)

# --- Run ---
if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(epochs=5, batch_size=32, games_per_epoch=10)