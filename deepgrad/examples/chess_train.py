import random
import math
import os
import dill
import copy
import chess
from ctypes import c_float
from tqdm import tqdm
from deepgrad.tensor import Tensor
from deepgrad.batchnorm import BatchNorm2D
from deepgrad.optim import Adam

# --- Constants ---
POLICY_SIZE = 4672  # 73 planes * 8 * 8 (56 queen + 8 knight + 9 underpromotion)
MODEL_DIR = "deepgrad/examples/models"

# --- Board Encoding and Symmetry Augmentation ---
def board_to_tensor(board, augment=False):
    """
    Converts a chess board to a tensor with 12 planes (6 piece types x 2 colors).

    Args:
        board (chess.Board): The chess board.
        augment (bool): If True, applies random horizontal flip augmentation.

    Returns:
        Tensor: Shape (12, 8, 8) representing the board.
    """
    data = (c_float * (12 * 8 * 8))()
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        rank = 7 - rank if board.turn == chess.BLACK else rank
        color_offset = 0 if piece.color == chess.WHITE else 6
        piece_type = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}[piece.symbol().upper()]
        channel = color_offset + piece_type
        data[channel * 64 + rank * 8 + file] = 1.0
    
    tensor = Tensor(data, shape=(12, 8, 8), requires_grad=False)
    
    if augment and random.random() < 0.5:
        data_flipped = (c_float * (12 * 8 * 8))()
        for c in range(12):
            for r in range(8):
                for f in range(8):
                    data_flipped[c * 64 + r * 8 + (7 - f)] = data[c * 64 + r * 8 + f]
        tensor = Tensor(data_flipped, shape=(12, 8, 8), requires_grad=False)
    
    return tensor

# --- Policy Indexing (AlphaZero style) ---
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KNIGHT_MOVES = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]
UNDER_PROMOTIONS = [(56, chess.QUEEN), (64, chess.ROOK), (65, chess.BISHOP), (66, chess.KNIGHT)]

def move_to_plane_row_col(board, move, flipped=False):
    """
    Maps a chess move to a policy plane index, row, and column.

    Args:
        board (chess.Board): The chess board.
        move (chess.Move): The move to encode.
        flipped (bool): If True, applies horizontal flip.

    Returns:
        Tuple[int, int, int]: (plane, row, col) for the policy tensor.
    """
    from_sq, to_sq = move.from_square, move.to_square
    from_row, from_col = divmod(from_sq, 8)
    to_row, to_col = divmod(to_sq, 8)
    
    if not board.turn:
        from_row, to_row = 7 - from_row, 7 - to_row
    if flipped:
        from_col, to_col = 7 - from_col, 7 - to_col
    
    delta_row, delta_col = to_row - from_row, to_col - from_col
    distance = max(abs(delta_row), abs(delta_col), 1)
    
    piece = board.piece_at(from_sq)
    if piece is None:
        return 0, to_row, to_col
    
    if piece.piece_type == chess.KNIGHT:
        for i, (dr, dc) in enumerate(KNIGHT_MOVES):
            if (dr, dc) == (delta_row, delta_col):
                return 56 + i, to_row, to_col
    elif move.promotion and piece.piece_type == chess.PAWN:
        for plane, prom_piece in UNDER_PROMOTIONS:
            if move.promotion == prom_piece:
                return plane, to_row, to_col
    else:
        for i, (dr, dc) in enumerate(DIRECTIONS):
            if (delta_row, delta_col) != (0, 0) and (dr, dc) == (delta_row // distance, delta_col // distance):
                return i * 7 + (distance - 1), to_row, to_col
    
    return 0, to_row, to_col

def get_policy_index(board, move, flipped=False):
    """
    Converts a move to a policy index in the 4672-dimensional policy vector.

    Args:
        board (chess.Board): The chess board.
        move (chess.Move): The move to encode.
        flipped (bool): If True, applies horizontal flip.

    Returns:
        int: Index in the policy vector.
    """
    plane, row, col = move_to_plane_row_col(board, move, flipped)
    if flipped:
        col = 7 - col
    return plane * 64 + row * 8 + col

# --- Model (ResNet-style) ---
class ChessNet:
    def __init__(self):
        """
        Initializes a ResNet-style chess neural network with policy and value heads.
        """
        self.num_filters = 128
        self.num_blocks = 10
        
        # Initial conv layer
        self.w1 = Tensor.randn((self.num_filters, 12, 3, 3), std=0.1, requires_grad=True)
        self.b1 = Tensor.zeros((self.num_filters,), requires_grad=True)
        self.bn1 = BatchNorm2D(self.num_filters)
        
        # Residual blocks
        self.res_blocks = []
        for _ in range(self.num_blocks):
            w1 = Tensor.randn((self.num_filters, self.num_filters, 3, 3), std=0.1, requires_grad=True)
            b1 = Tensor.zeros((self.num_filters,), requires_grad=True)
            bn1 = BatchNorm2D(self.num_filters)
            w2 = Tensor.randn((self.num_filters, self.num_filters, 3, 3), std=0.1, requires_grad=True)
            b2 = Tensor.zeros((self.num_filters,), requires_grad=True)
            bn2 = BatchNorm2D(self.num_filters)
            self.res_blocks.append((w1, b1, bn1, w2, b2, bn2))
        
        # Policy head
        self.policy_conv = Tensor.randn((self.num_filters, self.num_filters, 1, 1), std=0.1, requires_grad=True)
        self.policy_b = Tensor.zeros((self.num_filters,), requires_grad=True)
        self.policy_bn = BatchNorm2D(self.num_filters)
        self.policy_w = Tensor.randn((self.num_filters * 64, POLICY_SIZE), std=0.1, requires_grad=True)
        self.policy_b2 = Tensor.zeros((POLICY_SIZE,), requires_grad=True)
        
        # Value head
        self.value_conv = Tensor.randn((1, self.num_filters, 1, 1), std=0.1, requires_grad=True)
        self.value_b = Tensor.zeros((1,), requires_grad=True)
        self.value_bn = BatchNorm2D(1)
        self.value_w1 = Tensor.randn((64, 512), std=0.1, requires_grad=True)
        self.value_b1 = Tensor.zeros((512,), requires_grad=True)
        self.value_w2 = Tensor.randn((512, 1), std=0.1, requires_grad=True)
        self.value_b2 = Tensor.zeros((1,), requires_grad=True)

    def __call__(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 12, 8, 8).

        Returns:
            Tuple[Tensor, Tensor]: Policy logits (batch_size, 4672), value (batch_size, 1).
        """
        if x.shape[1:] != (12, 8, 8):
            raise ValueError(f"Expected input shape (*, 12, 8, 8), got {x.shape}")
        
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
        """
        Returns all trainable parameters of the network.

        Returns:
            List[Tensor]: List of parameter tensors.
        """
        params = [
            self.w1, self.b1,
            self.policy_conv, self.policy_b, self.policy_w, self.policy_b2,
            self.value_conv, self.value_b, self.value_w1, self.value_b1, self.value_w2, self.value_b2,
            *self.bn1.parameters(), *self.policy_bn.parameters(), *self.value_bn.parameters()
        ]
        for w1, b1, bn1, w2, b2, bn2 in self.res_blocks:
            params.extend([w1, b1, w2, b2, *bn1.parameters(), *bn2.parameters()])
        return params

    def copy(self):
        """Returns a deep copy of the model."""
        return copy.deepcopy(self)

# --- MCTS with Dirichlet Noise ---
class Node:
    def __init__(self, board, parent=None, prior=0.0):
        """Initializes an MCTS node."""
        self.board = board
        self.parent = parent
        self.children = {}
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.visits = 0  # Track visits to prevent infinite loops

    def is_expanded(self):
        """Checks if the node is expanded."""
        return bool(self.children)

    def expand(self, model, is_root=False):
        """
        Expands the node by evaluating legal moves with the model.

        Args:
            model (ChessNet): Neural network model.
            is_root (bool): If True, applies Dirichlet noise to priors.

        Returns:
            float: Value estimate from the model.
        """
        if self.visits > 100:  # Prevent excessive recursion
            return 0.0
        
        tensor = board_to_tensor(self.board)
        batch_tensor = Tensor(tensor.data, shape=(1, 12, 8, 8))
        policy_logits, value = model(batch_tensor)
        
        policy_data = [min(max(x, -10.0), 10.0) for x in policy_logits.data]
        value_data = min(max(float(value.data[0]), -1.0), 1.0)
        
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return value_data
        
        total_logits = [math.exp(policy_data[get_policy_index(self.board, m)]) for m in legal_moves]
        total = sum(total_logits) + 1e-8
        priors = [p / total for p in total_logits]

        if is_root:
            n = len(legal_moves)
            dirichlet = [0.0] * n
            r = sum(random.random() for _ in range(n))
            for i in range(n):
                dirichlet[i] = -math.log(random.random() + 1e-8) / (r + 1e-8) * 0.3
            total_noise = sum(dirichlet) + 1e-8
            dirichlet = [x / total_noise for x in dirichlet]
            priors = [0.75 * p + 0.25 * n for p, n in zip(priors, dirichlet)]

        for move, prior in zip(legal_moves, priors):
            new_board = self.board.copy()
            new_board.push(move)
            self.children[move] = Node(new_board, parent=self, prior=max(prior, 1e-8))
        
        self.visits += 1
        return value_data

    def select_child(self):
        """
        Selects a child node based on UCB score.

        Returns:
            Tuple[chess.Move, Node]: Selected move and child node, or (None, None) if no children.
        """
        if not self.children:
            return None, None
        def ucb_score(child):
            return child.Q + 1.0 * child.P * math.sqrt(self.N + 1e-8) / (1 + child.N)
        return max(self.children.items(), key=lambda item: ucb_score(item[1]))

    def backprop(self, value):
        """
        Backpropagates value through the tree.

        Args:
            value (float): Value to backpropagate.
        """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backprop(-value)

class MCTS:
    def __init__(self, model):
        """Initializes MCTS with a model and 2 simulations."""
        self.model = model
        self.sims = 25

    def run(self, board):
        """
        Runs MCTS simulations to compute move policy.

        Args:
            board (chess.Board): Current board state.

        Returns:
            Tuple[dict, Node]: Move policy (move: probability), root node.
        """
        root = Node(board)
        try:
            value = root.expand(self.model, is_root=True)
        except Exception as e:
            print(f"⚠️ Error during root expansion: {e}")
            return {}, root

        for sim in range(self.sims):
            node = root
            search_path = [node]
            max_depth = 100  # Prevent infinite recursion
            depth = 0
            
            try:
                while node.is_expanded() and depth < max_depth:
                    move, child = node.select_child()
                    if child is None:
                        break
                    node = child
                    search_path.append(node)
                    depth += 1
                
                if depth >= max_depth:
                    print("⚠️ Max depth reached in MCTS")
                    continue
                
                if node.board.is_game_over():
                    result = node.board.result()
                    value = 1.0 if result == '1-0' else -1.0 if result == '0-1' else 0.0
                    if node.board.turn == chess.BLACK:
                        value = -value
                else:
                    value = node.expand(self.model)
                
                for n in reversed(search_path):
                    n.backprop(value)
            except Exception as e:
                print(f"⚠️ Exception during simulation {sim+1}/{self.sims}: {e}")
                continue

        move_visits = {m: child.N for m, child in root.children.items()}
        total_visits = sum(move_visits.values()) + 1e-8
        policy = {m: n / total_visits for m, n in move_visits.items()}
        return policy, root

# --- Dataset Loader with Augmentation ---
class ChessDataset:
    def __init__(self, games):
        """Initializes dataset with game data."""
        self.games = games

    def __len__(self):
        """Returns number of games."""
        return len(self.games)

    def __getitem__(self, idx):
        """
        Returns a single game sample.

        Args:
            idx (int): Index of the game.

        Returns:
            Tuple[str, list, dict, float]: FEN, state tensor data, policy, value.
        """
        return self.games[idx]

    def get_batch(self, batch_size):
        """
        Returns a batch of game data with augmentation.

        Args:
            batch_size (int): Number of samples in the batch.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: State, policy, and value tensors.
        """
        indices = random.choices(range(len(self.games)), k=batch_size)
        states, policies, values = [], [], []
        for idx in indices:
            fen, state, policy, value = self.games[idx]
            flipped = random.random() < 0.5
            states.append(board_to_tensor(chess.Board(fen), augment=flipped).data)
            policies.append(self.encode_policy(chess.Board(fen), policy, flipped))
            values.append(value)
        
        state_data = [x for state in states for x in state]
        policy_data = [x for policy in policies for x in policy]
        value_data = values
        
        return (
            Tensor(state_data, shape=(batch_size, 12, 8, 8)),
            Tensor(policy_data, shape=(batch_size, POLICY_SIZE)),
            Tensor(value_data, shape=(batch_size, 1))
        )

    def encode_policy(self, board, move_probs, flipped=False):
        """
        Encodes move probabilities into a policy vector.

        Args:
            board (chess.Board): The chess board.
            move_probs (dict): Move to probability mapping.
            flipped (bool): If True, applies horizontal flip.

        Returns:
            list: Policy vector of length 4672.
        """
        arr = [0.0] * POLICY_SIZE
        for move, prob in move_probs.items():
            idx = get_policy_index(board, move, flipped)
            if 0 <= idx < POLICY_SIZE:
                arr[idx] = max(prob, 0.0)
        return arr

# --- Model Evaluation ---
def evaluate_models(model_new, model_old, num_games=5):
    """
    Evaluates two models by playing games against each other.

    Args:
        model_new (ChessNet): New model to evaluate.
        model_old (ChessNet): Old model for comparison.
        num_games (int): Number of games to play.

    Returns:
        Tuple[int, int, int]: Wins, losses, draws for the new model.
    """
    wins, losses, draws = 0, 0, 0
    for game_num in range(num_games):
        print(f"Evaluating game {game_num + 1}/{num_games}")
        board = chess.Board()
        mcts_new = MCTS(model_new)
        mcts_old = MCTS(model_old)
        move_count = 0

        while not board.is_game_over() and move_count < 512:
            mcts = mcts_new if board.turn == chess.WHITE else mcts_old
            try:
                policy, _ = mcts.run(board)
                moves = list(policy.keys())
                weights = list(policy.values())
                if not moves or any(math.isnan(w) or math.isinf(w) for w in weights) or sum(weights) == 0:
                    print("⚠️ Invalid policy, using random move")
                    move = random.choice(list(board.legal_moves))
                else:
                    total = sum(weights) + 1e-8
                    weights = [w / total for w in weights]
                    move = random.choices(moves, weights=weights, k=1)[0]
            except Exception as e:
                print(f"⚠️ MCTS error: {e}")
                move = random.choice(list(board.legal_moves))
            
            board.push(move)
            move_count += 1

        if move_count >= 512:
            draws += 1
            continue

        result = board.result()
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1

    return wins, losses, draws

# --- Training ---
class Trainer:
    def __init__(self):
        """Initializes the trainer with model, optimizer, and game memory."""
        self.model = ChessNet()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.games = []
        self.best_model = self.model.copy()

    def self_play(self, num_games):
        """
        Generates self-play games using MCTS.

        Args:
            num_games (int): Number of games to play.
        """
        mcts = MCTS(self.best_model)
        for _ in tqdm(range(num_games), desc="Self-Play"):
            board = chess.Board()
            game = []
            move_count = 0
            while not board.is_game_over() and move_count < 512:
                temperature = 1.0 if move_count < 30 else 0.1
                policy, root = mcts.run(board)
                moves, probs = list(policy.keys()), list(policy.values())
                total = sum(probs) + 1e-8
                probs = [p / total for p in probs]
                probs = [p ** (1 / temperature) for p in probs]
                total = sum(probs) + 1e-8
                probs = [p / total for p in probs]
                move = random.choices(moves, weights=probs, k=1)[0]
                game.append((board.fen(), board_to_tensor(board).data, policy.copy(), None))
                board.push(move)
                move_count += 1
            
            result = board.result()
            outcome = 1.0 if result == '1-0' else -1.0 if result == '0-1' else 0.0
            for i, (fen, tensor_data, move_probs, _) in enumerate(game):
                outcome_signed = outcome if chess.Board(fen).turn == chess.WHITE else -outcome
                game[i] = (fen, tensor_data, move_probs, outcome_signed)
            
            self.games.extend(game)
        
        if len(self.games) > 5000:
            self.games = self.games[-5000:]

    def loss_function(self, policy_logits, target_policy, value, target_value):
        """
        Computes combined policy and value loss.

        Args:
            policy_logits (Tensor): Predicted policy logits (batch_size, 4672).
            target_policy (Tensor): Target policy (batch_size, 4672).
            value (Tensor): Predicted value (batch_size, 1).
            target_value (Tensor): Target value (batch_size, 1).

        Returns:
            Tensor: Combined loss.
        """
        log_softmax = policy_logits.log_softmax(dim=-1)
        batch_size, policy_dim = policy_logits.shape
        
        policy_loss_data = 0.0
        for b in range(batch_size):
            for i in range(policy_dim):
                policy_loss_data -= target_policy.data[b * policy_dim + i] * log_softmax.data[b * policy_dim + i]
        
        policy_loss = Tensor([policy_loss_data / batch_size], shape=(1,), requires_grad=True)
        value_loss = ((value - target_value) ** 2).mean()
        
        return policy_loss + value_loss

    def train(self, epochs=5, batch_size=32, games_per_epoch=10):
        """
        Trains the model using self-play and MCTS.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            games_per_epoch (int): Number of self-play games per epoch.
        """
        dataset = ChessDataset(self.games)
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            self.self_play(num_games=games_per_epoch)
            dataset.games = self.games
            num_batches = max(1, len(dataset) // batch_size)
            total_loss = 0.0

            for _ in tqdm(range(num_batches), desc=f"Training Epoch {epoch+1}"):
                inputs, target_policy, target_value = dataset.get_batch(batch_size)
                policy_logits, value = self.model(inputs)
                
                if any(math.isnan(x) or math.isinf(x) for x in policy_logits.data):
                    print("⚠️ NaN/Inf in policy_logits")
                    continue
                if any(math.isnan(x) or math.isinf(x) for x in value.data):
                    print("⚠️ NaN/Inf in value")
                    continue
                
                loss = self.loss_function(policy_logits, target_policy, value, target_value)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += float(loss.data[0])

            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

            wins, losses, draws = evaluate_models(self.model, self.best_model)
            print(f"Evaluation: Wins={wins}, Losses={losses}, Draws={draws}")
            if wins > losses:
                self.best_model = self.model.copy()
                self.save_model(f"{MODEL_DIR}/best_model.pkl")
                print("Updated best model")

            epoch_file = f"{MODEL_DIR}/model_epoch_{epoch+1}.pkl"
            self.save_model(epoch_file)

    def save_model(self, filename):
        """
        Saves the model parameters to a file.

        Args:
            filename (str): Path to save the model.
        """
        def to_list(ctypes_array):
            return [ctypes_array[i] for i in range(len(ctypes_array))]
        
        parameters = {
            'num_blocks': self.model.num_blocks,
            'num_filters': self.model.num_filters,
            'w1': to_list(self.model.w1.data),
            'b1': to_list(self.model.b1.data),
            'policy_conv': to_list(self.model.policy_conv.data),
            'policy_b': to_list(self.model.policy_b.data),
            'policy_w': to_list(self.model.policy_w.data),
            'policy_b2': to_list(self.model.policy_b2.data),
            'value_conv': to_list(self.model.value_conv.data),
            'value_b': to_list(self.model.value_b.data),
            'value_w1': to_list(self.model.value_w1.data),
            'value_b1': to_list(self.model.value_b1.data),
            'value_w2': to_list(self.model.value_w2.data),
            'value_b2': to_list(self.model.value_b2.data),
            'bn1': {
                'gamma': to_list(self.model.bn1.gamma.data),
                'beta': to_list(self.model.bn1.beta.data),
                'running_mean': to_list(self.model.bn1.running_mean) if hasattr(self.model.bn1, 'running_mean') else [0.0] * self.model.num_filters,
                'running_var': to_list(self.model.bn1.running_var) if hasattr(self.model.bn1, 'running_var') else [1.0] * self.model.num_filters
            },
            'policy_bn': {
                'gamma': to_list(self.model.policy_bn.gamma.data),
                'beta': to_list(self.model.policy_bn.beta.data),
                'running_mean': to_list(self.model.policy_bn.running_mean) if hasattr(self.model.policy_bn, 'running_mean') else [0.0] * self.model.num_filters,
                'running_var': to_list(self.model.policy_bn.running_var) if hasattr(self.model.policy_bn, 'running_var') else [1.0] * self.model.num_filters
            },
            'value_bn': {
                'gamma': to_list(self.model.value_bn.gamma.data),
                'beta': to_list(self.model.value_bn.beta.data),
                'running_mean': to_list(self.model.value_bn.running_mean) if hasattr(self.model.value_bn, 'running_mean') else [0.0] * 1,
                'running_var': to_list(self.model.value_bn.running_var) if hasattr(self.model.value_bn, 'running_var') else [1.0] * 1
            },
            'res_blocks': [
                {
                    'w1': to_list(w1.data),
                    'b1': to_list(b1.data),
                    'w2': to_list(w2.data),
                    'b2': to_list(b2.data),
                    'bn1': {
                        'gamma': to_list(bn1.gamma.data),
                        'beta': to_list(bn1.beta.data),
                        'running_mean': to_list(bn1.running_mean) if hasattr(bn1, 'running_mean') else [0.0] * self.model.num_filters,
                        'running_var': to_list(bn1.running_var) if hasattr(bn1, 'running_var') else [1.0] * self.model.num_filters
                    },
                    'bn2': {
                        'gamma': to_list(bn2.gamma.data),
                        'beta': to_list(bn2.beta.data),
                        'running_mean': to_list(bn2.running_mean) if hasattr(bn2, 'running_mean') else [0.0] * self.model.num_filters,
                        'running_var': to_list(bn2.running_var) if hasattr(bn2, 'running_var') else [1.0] * self.model.num_filters
                    }
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
    trainer.train(epochs=1, batch_size=32, games_per_epoch=10)