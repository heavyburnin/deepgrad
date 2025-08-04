import random
import math
import os
import dill
import copy
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import chess
from ctypes import c_float
from tqdm import tqdm
import itertools
from deepgrad.tensor import Tensor
from deepgrad.batchnorm import BatchNorm2D
from deepgrad.optim import Adam

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
@dataclass
class Config:
    policy_size: int = 4672  # 73 planes * 8 * 8
    model_dir: str = "deepgrad/examples/models"
    num_filters: int = 64    # Reduced from 128 for faster convs
    num_res_blocks: int = 8  # Reduced from 10
    mcts_sims: int = 2      # Reduced from 25 for faster MCTS
    batch_size: int = 16     # Reduced to lower memory usage
    games_per_epoch: int = 1 # Reduced for faster epochs
    max_game_length: int = 256  # Reduced from 512
    max_dataset_size: int = 2000  # Reduced from 5000
    learning_rate: float = 0.002  # Increased for faster convergence
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

CONFIG = Config()

# --- Board Encoding ---
def board_to_tensor(board: chess.Board, augment: bool = False) -> Tensor:
    """
    Convert a chess board to a tensor with 12 planes (6 piece types x 2 colors).

    Args:
        board: The chess board.
        augment: If True, applies random horizontal flip.

    Returns:
        Tensor with shape (12, 8, 8).
    """
    try:
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
            flipped_data = (c_float * (12 * 8 * 8))()
            for c in range(12):
                for r in range(8):
                    for f in range(8):
                        flipped_data[c * 64 + r * 8 + (7 - f)] = data[c * 64 + r * 8 + f]
            tensor = Tensor(flipped_data, shape=(12, 8, 8), requires_grad=False)

        return tensor
    except Exception as e:
        logger.error(f"Error in board_to_tensor: {e}")
        raise

# --- Policy Indexing ---
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KNIGHT_MOVES = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]
UNDER_PROMOTIONS = [(56, chess.QUEEN), (64, chess.ROOK), (65, chess.BISHOP), (66, chess.KNIGHT)]

def move_to_policy_coords(board: chess.Board, move: chess.Move, flipped: bool = False) -> Tuple[int, int, int]:
    """
    Map a chess move to policy plane, row, and column.

    Args:
        board: The chess board.
        move: The move to encode.
        flipped: If True, applies horizontal flip.

    Returns:
        Tuple of (plane, row, col).
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
        logger.warning(f"No piece at square {from_sq}")
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

def get_policy_index(board: chess.Board, move: chess.Move, flipped: bool = False) -> int:
    plane, row, col = move_to_policy_coords(board, move, flipped)
    if flipped:
        col = 7 - col
    idx = plane * 64 + row * 8 + col
    if not 0 <= idx < CONFIG.policy_size:
        logger.error(f"Invalid policy index: {idx}, plane={plane}, row={row}, col={col}, move={move}, board={board.fen()}")
        return 0
    logger.debug(f"Policy index: {idx} for move {move}")
    return idx

# --- Neural Network ---
class ChessNet:
    def __init__(self):
        """Initialize a ResNet-style chess network."""
        self.num_filters = CONFIG.num_filters
        self.num_blocks = CONFIG.num_res_blocks

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
        self.policy_w = Tensor.randn((self.num_filters * 64, CONFIG.policy_size), std=0.1, requires_grad=True)
        self.policy_b2 = Tensor.zeros((CONFIG.policy_size,), requires_grad=True)

        # Value head
        self.value_conv = Tensor.randn((1, self.num_filters, 1, 1), std=0.1, requires_grad=True)
        self.value_b = Tensor.zeros((1,), requires_grad=True)
        self.value_bn = BatchNorm2D(1)
        self.value_w1 = Tensor.randn((64, 512), std=0.1, requires_grad=True)
        self.value_b1 = Tensor.zeros((512,), requires_grad=True)
        self.value_w2 = Tensor.randn((512, 1), std=0.1, requires_grad=True)
        self.value_b2 = Tensor.zeros((1,), requires_grad=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 12, 8, 8).

        Returns:
            Policy logits (batch_size, 4672), value (batch_size, 1).
        """
        try:
            if x.shape[1:] != (12, 8, 8):
                raise ValueError(f"Expected input shape (*, 12, 8, 8), got {x.shape}")
            logger.debug(f"Input shape: {x.shape}, sample: {[x.data[i] for i in range(min(10, x.size))]}")

            # Initial conv
            x = x.conv2d(self.w1, self.b1, stride=(1, 1), padding=(1, 1))
            if x.shape[1:] != (self.num_filters, 8, 8):
                logger.error(f"Invalid shape after conv1: {x.shape}")
                raise ValueError("Invalid shape after conv1")
            x = self.bn1(x).relu()
            logger.debug(f"After conv1+bn+relu shape: {x.shape}, sample: {[x.data[i] for i in range(min(10, x.size))]}")

            # Residual blocks
            for i, (w1, b1, bn1, w2, b2, bn2) in enumerate(self.res_blocks):
                residual = x
                x = x.conv2d(w1, b1, stride=(1, 1), padding=(1, 1))
                x = bn1(x).relu()
                x = x.conv2d(w2, b2, stride=(1, 1), padding=(1, 1))
                x = bn2(x)
                if x.shape != residual.shape:
                    logger.error(f"Shape mismatch in res_block {i+1}: x={x.shape}, residual={residual.shape}")
                    raise ValueError("Shape mismatch in residual block")
                x = x + residual
                x = x.relu()
                logger.debug(f"After res_block {i+1} shape: {x.shape}, sample: {[x.data[i] for i in range(min(10, x.size))]}")

            # Policy head
            policy = x.conv2d(self.policy_conv, self.policy_b, stride=(1, 1), padding=(0, 0))
            policy = self.policy_bn(policy).relu()
            policy = policy.flatten(start_dim=1)
            if policy.shape[1] != self.num_filters * 64:
                logger.error(f"Invalid policy flatten shape: {policy.shape}")
                raise ValueError("Invalid policy flatten shape")
            policy = policy.matmul(self.policy_w) + self.policy_b2
            if policy.shape != (x.shape[0], CONFIG.policy_size):
                logger.error(f"Invalid policy output shape: {policy.shape}")
                raise ValueError("Invalid policy output shape")
            logger.debug(f"Policy shape: {policy.shape}, sample: {[policy.data[i] for i in range(min(10, policy.size))]}")

            # Value head
            value = x.conv2d(self.value_conv, self.value_b, stride=(1, 1), padding=(0, 0))
            value = self.value_bn(value).relu()
            value = value.flatten(start_dim=1)
            value = (value.matmul(self.value_w1) + self.value_b1).relu()
            value = (value.matmul(self.value_w2) + self.value_b2).tanh()
            if value.shape != (x.shape[0], 1):
                logger.error(f"Invalid value output shape: {value.shape}")
                raise ValueError("Invalid value output shape")
            logger.debug(f"Value shape: {value.shape}, data: {value.data[0]}")

            return policy, value
        except Exception as e:
            logger.error(f"Error in ChessNet forward: {e}")
            raise

    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters."""
        params = [
            self.w1, self.b1,
            self.policy_conv, self.policy_b, self.policy_w, self.policy_b2,
            self.value_conv, self.value_b, self.value_w1, self.value_b1, self.value_w2, self.value_b2,
            *self.bn1.parameters(), *self.policy_bn.parameters(), *self.value_bn.parameters()
        ]
        for w1, b1, bn1, w2, b2, bn2 in self.res_blocks:
            params.extend([w1, b1, w2, b2, *bn1.parameters(), *bn2.parameters()])
        return params

    def copy(self) -> 'ChessNet':
        """Return a deep copy of the model."""
        return copy.deepcopy(self)

# --- MCTS ---
class MCTSNode:
    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None, prior: float = 0.0):
        """Initialize an MCTS node."""
        self.board = board
        self.parent = parent
        self.children: Dict[chess.Move, 'MCTSNode'] = {}
        self.prior = max(prior, 1e-8)
        self.visits = 0
        self.value_sum = 0.0
        self.legal_moves: Optional[List[chess.Move]] = None

    def is_expanded(self) -> bool:
        """Check if the node is expanded."""
        return bool(self.children)

    def expand(self, model: ChessNet, is_root: bool = False) -> float:
        """
        Expand the node using the model.

        Args:
            model: Neural network model.
            is_root: If True, apply Dirichlet noise.

        Returns:
            Value estimate.
        """
        try:
            if self.visits > 100:
                logger.warning("Max visits reached, returning 0.0")
                return 0.0

            # Create input tensor
            tensor = board_to_tensor(self.board)
            if tensor.shape != (12, 8, 8) or tensor.size != 12 * 8 * 8:
                logger.error(f"Invalid board tensor shape: {tensor.shape}, size: {tensor.size}")
                raise ValueError("Invalid board tensor shape")
            logger.debug(f"Board tensor shape: {tensor.shape}, sample: {[tensor.data[i] for i in range(min(10, tensor.size))] if tensor.data else 'None'}")

            # Reshape for batch processing
            batch_tensor = tensor.reshape((1, 12, 8, 8))
            if batch_tensor.size != tensor.size:
                logger.error(f"Reshape failed: {batch_tensor.shape}, size: {batch_tensor.size}")
                raise ValueError("Reshape failed")

            # Forward pass
            policy_logits, value = model.forward(batch_tensor)
            if policy_logits.shape != (1, CONFIG.policy_size) or value.shape != (1, 1):
                logger.error(f"Invalid output shapes: policy_logits={policy_logits.shape}, value={value.shape}")
                raise ValueError("Invalid network output shapes")
            logger.debug(f"Policy logits shape: {policy_logits.shape}, sample: {[policy_logits.data[i] for i in range(min(10, policy_logits.size))]}")
            logger.debug(f"Value shape: {value.shape}, data: {value.data[0]}")

            # Clamp value to prevent extreme values
            value_data = min(max(float(value.data[0]), -1.0), 1.0)
            if not (-1e10 < value_data < 1e10):
                logger.warning(f"Invalid value: {value_data}")
                value_data = 0.0

            # Get legal moves
            self.legal_moves = list(self.board.legal_moves)
            if not self.legal_moves:
                logger.info("No legal moves, returning value")
                return value_data

            # Compute policy for legal moves
            policy_logits = policy_logits.reshape((CONFIG.policy_size,))
            if policy_logits.size != CONFIG.policy_size:
                logger.error(f"Policy reshape failed: size={policy_logits.size}")
                raise ValueError("Policy reshape failed")

            legal_indices = []
            for move in self.legal_moves:
                idx = get_policy_index(self.board, move)
                if not 0 <= idx < CONFIG.policy_size:
                    logger.error(f"Invalid policy index for move {move}: {idx}")
                    idx = 0  # Fallback to avoid segfault
                legal_indices.append(idx)

            # Create legal logits tensor with bounds checking
            legal_logits_data = (c_float * len(legal_indices))()
            for i, idx in enumerate(legal_indices):
                if idx >= policy_logits.size:
                    logger.error(f"Index out of bounds: {idx} >= {policy_logits.size}")
                    raise ValueError("Index out of bounds")
                legal_logits_data[i] = policy_logits.data[idx]
            legal_logits = Tensor(legal_logits_data, shape=(len(legal_indices),), requires_grad=False)
            logger.debug(f"Legal logits shape: {legal_logits.shape}, sample: {[legal_logits.data[i] for i in range(min(10, legal_logits.size))]}")

            # Compute log_softmax with numerical stability
            log_probs = legal_logits.log_softmax(dim=0)
            for i in range(log_probs.size):
                if not (-1e10 < log_probs.data[i] < 1e10):
                    logger.warning(f"Invalid log_prob at index {i}: {log_probs.data[i]}")
                    log_probs.data[i] = -1e8  # Replace with large negative value

            # Compute priors
            priors = [0.0] * len(legal_indices)
            total = 0.0
            for i in range(len(legal_indices)):
                exp_val = math.exp(min(float(log_probs.data[i]), 100.0))  # Cap to prevent overflow
                priors[i] = exp_val
                total += exp_val
            total += 1e-8
            priors = [p / total for p in priors]

            # Apply Dirichlet noise for root node
            if is_root:
                n = len(self.legal_moves)
                dirichlet = [random.random() + 1e-8 for _ in range(n)]  # Simplified Dirichlet
                r = sum(dirichlet) + 1e-8
                dirichlet = [x / r * CONFIG.dirichlet_alpha for x in dirichlet]
                priors = [(1 - CONFIG.dirichlet_epsilon) * p + CONFIG.dirichlet_epsilon * d for p, d in zip(priors, dirichlet)]

            # Create child nodes
            for move, prior in zip(self.legal_moves, priors):
                if not 0 <= prior <= 1:
                    logger.warning(f"Invalid prior: {prior} for move {move}")
                    prior = max(min(prior, 1.0), 1e-8)
                new_board = self.board.copy()
                new_board.push(move)
                self.children[move] = MCTSNode(new_board, parent=self, prior=prior)

            self.visits += 1
            return value_data
        except Exception as e:
            logger.error(f"Error in node expansion: {e}")
            return 0.0
    
    def select_child(self) -> Tuple[Optional[chess.Move], Optional['MCTSNode']]:
        """Select a child node based on UCB score."""
        if not self.children:
            return None, None
        def ucb_score(child: 'MCTSNode') -> float:
            q = child.value_sum / child.visits if child.visits > 0 else 0.0
            u = child.prior * math.sqrt(self.visits + 1e-8) / (1 + child.visits)
            return q + u
        return max(self.children.items(), key=lambda item: ucb_score(item[1]))

    def backprop(self, value: float):
        """Backpropagate value through the tree."""
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backprop(-value)

class MCTS:
    def __init__(self, model: ChessNet):
        """Initialize MCTS."""
        self.model = model
        self.sims = CONFIG.mcts_sims

    def run(self, board: chess.Board) -> Tuple[Dict[chess.Move, float], MCTSNode]:
        """
        Run MCTS simulations.

        Args:
            board: Current board state.

        Returns:
            Move policy, root node.
        """
        root = MCTSNode(board)
        try:
            value = root.expand(self.model, is_root=True)
        except Exception as e:
            logger.error(f"Error during root expansion: {e}")
            return {}, root

        for sim in range(self.sims):
            node = root
            search_path = [node]
            depth = 0
            max_depth = 100

            try:
                while node.is_expanded() and depth < max_depth:
                    move, child = node.select_child()
                    if child is None:
                        break
                    node = child
                    search_path.append(node)
                    depth += 1

                if depth >= max_depth:
                    logger.warning("Max depth reached in MCTS")
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
                logger.error(f"Exception in simulation {sim+1}/{self.sims}: {e}")
                continue

        policy = {m: child.visits / (sum(c.visits for c in root.children.values()) + 1e-8)
                  for m, child in root.children.items()}
        return policy, root

# --- Dataset ---
class ChessDataset:
    def __init__(self, games: List[Tuple[str, List[float], Dict[chess.Move, float], float]]):
        """Initialize dataset."""
        self.games = games

    def __len__(self) -> int:
        """Return number of games."""
        return len(self.games)

    def get_batch(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Get a batch of data with augmentation.

        Args:
            batch_size: Number of samples.

        Returns:
            State, policy, and value tensors.
        """
        try:
            indices = random.choices(range(len(self.games)), k=batch_size)
            states, policies, values = [], [], []
            for idx in indices:
                fen, _, policy, value = self.games[idx]
                board = chess.Board(fen)
                flipped = random.random() < 0.5
                states.append(board_to_tensor(board, augment=flipped).data)
                policies.append(self.encode_policy(board, policy, flipped).data)
                values.append(value)

            state_data = list(itertools.chain.from_iterable(states))
            policy_data = list(itertools.chain.from_iterable(policies))
            value_data = values

            return (
                Tensor(state_data, shape=(batch_size, 12, 8, 8), requires_grad=False),
                Tensor(policy_data, shape=(batch_size, CONFIG.policy_size), requires_grad=False),
                Tensor(value_data, shape=(batch_size, 1), requires_grad=False)
            )
        except Exception as e:
            logger.error(f"Error in get_batch: {e}")
            raise

    def encode_policy(self, board: chess.Board, move_probs: Dict[chess.Move, float], flipped: bool = False) -> Tensor:
        """
        Encode move probabilities into a policy vector.

        Args:
            board: The chess board.
            move_probs: Move to probability mapping.
            flipped: If True, apply horizontal flip.

        Returns:
            Policy tensor of shape (4672,).
        """
        arr = [0.0] * CONFIG.policy_size
        total_prob = sum(max(prob, 0.0) for prob in move_probs.values()) + 1e-8
        for move, prob in move_probs.items():
            idx = get_policy_index(board, move, flipped)
            arr[idx] = max(prob, 0.0) / total_prob
        return Tensor(arr, shape=(CONFIG.policy_size,), requires_grad=False)

def evaluate_models(model_new: ChessNet, model_old: ChessNet, num_games: int = 5) -> Tuple[int, int, int]:
    """
    Evaluate the new model against the old model by playing games.

    Args:
        model_new: The new ChessNet model to evaluate.
        model_old: The baseline ChessNet model.
        num_games: Number of games to play.

    Returns:
        Tuple of (wins, losses, draws) for the new model as White.
    """
    wins, losses, draws = 0, 0, 0
    for game_num in range(num_games):
        logger.info(f"Starting evaluation game {game_num + 1}/{num_games}")
        board = chess.Board()
        mcts_new = MCTS(model_new)
        mcts_old = MCTS(model_old)
        move_count = 0

        while not board.is_game_over() and move_count < CONFIG.max_game_length:
            mcts = mcts_new if board.turn == chess.WHITE else mcts_old
            logger.debug(f"Move {move_count + 1}, turn: {'White' if board.turn == chess.WHITE else 'Black'}, FEN: {board.fen()}")
            try:
                policy, root = mcts.run(board)
                logger.debug(f"Policy size: {len(policy)}, sample: {list(policy.items())[:5]}")
                if not policy or not board.legal_moves:
                    logger.warning(f"Empty policy or no legal moves in game {game_num + 1}, move {move_count + 1}")
                    break
                moves, weights = list(policy.keys()), list(policy.values())
                logger.debug(f"Policy weights: {weights[:10]}, num moves: {len(moves)}")
                if any(not (-1e10 < w < 1e10) for w in weights):
                    logger.warning(f"Invalid weights in game {game_num + 1}, move {move_count + 1}: {weights[:10]}")
                    move = random.choice(list(board.legal_moves))
                else:
                    move = random.choices(moves, weights=weights, k=1)[0]
                    logger.debug(f"Selected move: {move}")
                board.push(move)
            except Exception as e:
                logger.error(f"MCTS error in game {game_num + 1}, move {move_count + 1}: {e}")
                break
            finally:
                # Release memory for root node and its children
                if 'root' in locals():
                    for node in root.children.values():
                        if hasattr(node, 'tensor'):
                            node.tensor.release_graph()
            move_count += 1

        if move_count >= CONFIG.max_game_length:
            logger.info(f"Game {game_num + 1} reached max length, counted as draw")
            draws += 1
            continue

        result = board.result()
        logger.info(f"Game {game_num + 1} result: {result}")
        if result == '1-0':
            wins += 1
        elif result == '0-1':
            losses += 1
        else:
            draws += 1

    logger.info(f"Evaluation results: Wins={wins}, Losses={losses}, Draws={draws}")
    return wins, losses, draws

# --- Trainer ---
class Trainer:
    def __init__(self):
        """Initialize trainer."""
        self.model = ChessNet()
        self.optimizer = Adam(self.model.parameters(), lr=CONFIG.learning_rate)
        self.games: List[Tuple[str, List[float], Dict[chess.Move, float], float]] = []
        self.best_model = self.model.copy()

    def self_play(self, num_games: int):
        """Generate self-play games."""
        mcts = MCTS(self.best_model)
        for _ in tqdm(range(num_games), desc="Self-Play"):
            board = chess.Board()
            game = []
            move_count = 0

            while not board.is_game_over() and move_count < CONFIG.max_game_length:
                temperature = 1.0 if move_count < 30 else 0.1
                policy, _ = mcts.run(board)
                if not policy:
                    logger.warning("Empty policy, breaking game")
                    break

                moves, probs = list(policy.keys()), list(policy.values())
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

        if len(self.games) > CONFIG.max_dataset_size:
            self.games = self.games[-CONFIG.max_dataset_size:]

    def loss_function(self, policy_logits: Tensor, target_policy: Tensor, value: Tensor, target_value: Tensor) -> Tensor:
        """
        Computes combined loss for AlphaZero:
        - Policy loss: Cross-entropy between predicted policy logits and target policy (MCTS visit counts).
        - Value loss: Mean squared error between predicted value and target value.

        Args:
            policy_logits: Predicted policy logits, shape (batch_size, 4672).
            target_policy: Target policy from MCTS, shape (batch_size, 4672).
            value: Predicted value, shape (batch_size, 1).
            target_value: Target value from game outcome, shape (batch_size, 1).

        Returns:
            Scalar tensor containing the total loss.

        Raises:
            ValueError: If input tensor shapes are invalid.
            RuntimeError: If numerical instability is detected.
        """
        try:
            # Validate shapes
            batch_size = policy_logits.shape[0]
            if (policy_logits.shape != (batch_size, CONFIG.policy_size) or
                target_policy.shape != (batch_size, CONFIG.policy_size) or
                value.shape != (batch_size, 1) or
                target_value.shape != (batch_size, 1)):
                raise ValueError(
                    f"Shape mismatch: policy_logits={policy_logits.shape}, "
                    f"target_policy={target_policy.shape}, value={value.shape}, "
                    f"target_value={target_value.shape}"
                )

            # Check for invalid values in inputs using data access
            for tensor, name in [(policy_logits, "policy_logits"), (target_policy, "target_policy"),
                                (value, "value"), (target_value, "target_value")]:
                for i in range(tensor.size):
                    val = tensor.data[i]
                    if not (-1e10 < val < 1e10):
                        logger.warning(f"Invalid value in {name} at index {i}: {val}")
                        tensor.data[i] = 0.0  # Replace with safe value

            # Policy loss: Cross-entropy
            log_probs = policy_logits.log_softmax(dim=1)  # Shape: (batch_size, 4672)
            # Ensure target_policy is normalized and non-negative
            target_policy = target_policy.clone()
            for i in range(target_policy.size):
                target_policy.data[i] = max(target_policy.data[i], 0.0)
            target_sum = target_policy.sum(dim=1)  # Shape: (batch_size,)
            for i in range(target_sum.size):
                if target_sum.data[i] == 0.0:
                    target_sum.data[i] = 1e-8  # Avoid division by zero
            target_policy = target_policy / target_sum.reshape((batch_size, 1))  # Normalize

            # Compute negative log likelihood
            cross_entropy = -(log_probs * target_policy).sum(dim=1)  # Shape: (batch_size,)
            policy_loss = cross_entropy.mean()  # Scalar tensor

            # Value loss: Mean squared error
            value_diff = value - target_value  # Shape: (batch_size, 1)
            squared_error = value_diff * value_diff  # Shape: (batch_size, 1)
            value_loss = squared_error.mean()  # Scalar tensor

            # Total loss
            total_loss = policy_loss + value_loss

            # Check for NaN/Inf in loss
            loss_val = total_loss.data[0]
            if math.isnan(loss_val) or math.isinf(loss_val):
                logger.error("NaN or Inf detected in total loss")
                raise RuntimeError("Numerical instability in loss computation")

            return total_loss
        except Exception as e:
            logger.error(f"Error in loss_function: {e}")
            raise

    def train(self, epochs: int = 5):
        """Train the model."""
        dataset = ChessDataset(self.games)
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            self.self_play(num_games=CONFIG.games_per_epoch)
            dataset.games = self.games
            num_batches = max(1, len(dataset) // CONFIG.batch_size)
            total_loss = 0.0

            for _ in tqdm(range(num_batches), desc=f"Training Epoch {epoch+1}"):
                try:
                    inputs, target_policy, target_value = dataset.get_batch(CONFIG.batch_size)
                    policy_logits, value = self.model.forward(inputs)

                    # Compute loss
                    loss = self.loss_function(policy_logits, target_policy, value, target_value)

                    # Check for invalid loss
                    if math.isnan(float(loss.data[0])) or math.isinf(float(loss.data[0])):
                        logger.warning("Invalid loss detected, skipping batch")
                        continue

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_loss += float(loss.data[0])

                    # Release computation graph
                    loss.release_graph()
                except Exception as e:
                    logger.error(f"Error in training batch: {e}")
                    continue

            avg_loss = total_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

            wins, losses, draws = evaluate_models(self.model, self.best_model)
            logger.info(f"Evaluation: Wins={wins}, Losses={losses}, Draws={draws}")
            if wins > losses:
                self.best_model = self.model.copy()
                self.save_model(f"{CONFIG.model_dir}/best_model.pkl")
                logger.info("Updated best model")

            self.save_model(f"{CONFIG.model_dir}/model_epoch_{epoch+1}.pkl")

    def save_model(self, filename: str):
        """Save model parameters."""
        try:
            def to_list(ctypes_array):
                return [float(ctypes_array[i]) for i in range(len(ctypes_array))]

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
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

# --- Main ---
if __name__ == "__main__":
    try:
        trainer = Trainer()
        trainer.train(epochs=1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise