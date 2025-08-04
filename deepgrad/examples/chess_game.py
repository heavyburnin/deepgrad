import dill
import chess
import random
import math
import os
from flask import Flask, render_template, request, jsonify
from deepgrad.tensor import Tensor
from deepgrad.batchnorm import BatchNorm2D

app = Flask(__name__)

# ChessNet class (matched to chess_alphazero.py)
class ChessNet:
    def __init__(self, num_blocks=10, num_filters=128):
        self.num_filters = num_filters
        # Initial conv layer
        self.w1 = Tensor.zeros((num_filters, 12, 3, 3), requires_grad=False)
        self.b1 = Tensor.zeros((num_filters,), requires_grad=False)
        self.bn1 = BatchNorm2D(num_filters)
        
        # Residual blocks
        self.res_blocks = []
        for _ in range(num_blocks):
            w1 = Tensor.zeros((num_filters, num_filters, 3, 3), requires_grad=False)
            b1 = Tensor.zeros((num_filters,), requires_grad=False)
            bn1 = BatchNorm2D(num_filters)
            w2 = Tensor.zeros((num_filters, num_filters, 3, 3), requires_grad=False)
            b2 = Tensor.zeros((num_filters,), requires_grad=False)
            bn2 = BatchNorm2D(num_filters)
            self.res_blocks.append((w1, b1, bn1, w2, b2, bn2))
        
        # Policy head
        self.policy_conv = Tensor.zeros((num_filters, num_filters, 1, 1), requires_grad=False)
        self.policy_b = Tensor.zeros((num_filters,), requires_grad=False)
        self.policy_bn = BatchNorm2D(num_filters)
        self.policy_w = Tensor.zeros((num_filters * 8 * 8, 4672), requires_grad=False)
        self.policy_b2 = Tensor.zeros((4672,), requires_grad=False)
        
        # Value head
        self.value_conv = Tensor.zeros((1, num_filters, 1, 1), requires_grad=False)
        self.value_b = Tensor.zeros((1,), requires_grad=False)
        self.value_bn = BatchNorm2D(1)
        self.value_w1 = Tensor.zeros((1 * 8 * 8, 512), requires_grad=False)
        self.value_b1 = Tensor.zeros((512,), requires_grad=False)
        self.value_w2 = Tensor.zeros((512, 1), requires_grad=False)
        self.value_b2 = Tensor.zeros((1,), requires_grad=False)

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

# Board encoding (from chess_alphazero.py)
def board_to_tensor(board):
    data = [0.0] * (12 * 8 * 8)
    for square, piece in board.piece_map().items():
        rank, file = divmod(square, 8)
        rank = 7 - rank if board.turn == chess.BLACK else rank
        color_offset = 0 if piece.color == chess.WHITE else 6
        piece_type = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}[piece.symbol().upper()]
        channel = color_offset + piece_type
        data[channel * 64 + rank * 8 + file] = 1.0
    return Tensor(data, shape=(12, 8, 8), requires_grad=False)

# Policy indexing (from chess_alphazero.py)
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KNIGHT_MOVES = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]

def move_to_plane_row_col(board, move):
    from_sq, to_sq = move.from_square, move.to_square
    from_row, from_col = divmod(from_sq, 8)
    to_row, to_col = divmod(to_sq, 8)
    if not board.turn:
        from_row, to_row = 7 - from_row, 7 - to_row
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

def get_policy_index(board, move):
    plane, row, col = move_to_plane_row_col(board, move)
    return plane * 64 + row * 8 + col

# Load model
def load_model(filename):
    try:
        with open(filename, "rb") as f:
            params = dill.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file {filename} not found")
    except Exception as e:
        raise RuntimeError(f"Error loading model from {filename}: {e}")

    model = ChessNet(num_blocks=params.get('num_blocks', 10), num_filters=params.get('num_filters', 128))
    
    # Load initial conv layer
    model.w1 = Tensor(params['w1'], shape=(model.num_filters, 12, 3, 3), requires_grad=False)
    model.b1 = Tensor(params['b1'], shape=(model.num_filters,), requires_grad=False)
    model.bn1.gamma = Tensor(params['bn1_gamma'], shape=(model.num_filters,), requires_grad=False)
    model.bn1.beta = Tensor(params['bn1_beta'], shape=(model.num_filters,), requires_grad=False)
    if 'bn1_running_mean' in params:
        model.bn1.running_mean = params['bn1_running_mean']
    if 'bn1_running_var' in params:
        model.bn1.running_var = params['bn1_running_var']
    
    # Load residual blocks
    for i, block_params in enumerate(params.get('res_blocks', [])):
        model.res_blocks[i][0] = Tensor(block_params['w1'], shape=(model.num_filters, model.num_filters, 3, 3), requires_grad=False)
        model.res_blocks[i][1] = Tensor(block_params['b1'], shape=(model.num_filters,), requires_grad=False)
        model.res_blocks[i][2].gamma = Tensor(block_params['bn1_gamma'], shape=(model.num_filters,), requires_grad=False)
        model.res_blocks[i][2].beta = Tensor(block_params['bn1_beta'], shape=(model.num_filters,), requires_grad=False)
        if 'bn1_running_mean' in block_params:
            model.res_blocks[i][2].running_mean = block_params['bn1_running_mean']
        if 'bn1_running_var' in block_params:
            model.res_blocks[i][2].running_var = block_params['bn1_running_var']
        model.res_blocks[i][3] = Tensor(block_params['w2'], shape=(model.num_filters, model.num_filters, 3, 3), requires_grad=False)
        model.res_blocks[i][4] = Tensor(block_params['b2'], shape=(model.num_filters,), requires_grad=False)
        model.res_blocks[i][5].gamma = Tensor(block_params['bn2_gamma'], shape=(model.num_filters,), requires_grad=False)
        model.res_blocks[i][5].beta = Tensor(block_params['bn2_beta'], shape=(model.num_filters,), requires_grad=False)
        if 'bn2_running_mean' in block_params:
            model.res_blocks[i][5].running_mean = block_params['bn2_running_mean']
        if 'bn2_running_var' in block_params:
            model.res_blocks[i][5].running_var = block_params['bn2_running_var']
    
    # Load policy head
    model.policy_conv = Tensor(params['policy_conv'], shape=(model.num_filters, model.num_filters, 1, 1), requires_grad=False)
    model.policy_b = Tensor(params['policy_b'], shape=(model.num_filters,), requires_grad=False)
    model.policy_bn.gamma = Tensor(params['policy_bn_gamma'], shape=(model.num_filters,), requires_grad=False)
    model.policy_bn.beta = Tensor(params['policy_bn_beta'], shape=(model.num_filters,), requires_grad=False)
    if 'policy_bn_running_mean' in params:
        model.policy_bn.running_mean = params['policy_bn_running_mean']
    if 'policy_bn_running_var' in params:
        model.policy_bn.running_var = params['policy_bn_running_var']
    model.policy_w = Tensor(params['policy_w'], shape=(model.num_filters * 8 * 8, 4672), requires_grad=False)
    model.policy_b2 = Tensor(params['policy_b2'], shape=(4672,), requires_grad=False)
    
    # Load value head
    model.value_conv = Tensor(params['value_conv'], shape=(1, model.num_filters, 1, 1), requires_grad=False)
    model.value_b = Tensor(params['value_b'], shape=(1,), requires_grad=False)
    model.value_bn.gamma = Tensor(params['value_bn_gamma'], shape=(1,), requires_grad=False)
    model.value_bn.beta = Tensor(params['value_bn_beta'], shape=(1,), requires_grad=False)
    if 'value_bn_running_mean' in params:
        model.value_bn.running_mean = params['value_bn_running_mean']
    if 'value_bn_running_var' in params:
        model.value_bn.running_var = params['value_bn_running_var']
    model.value_w1 = Tensor(params['value_w1'], shape=(1 * 8 * 8, 512), requires_grad=False)
    model.value_b1 = Tensor(params['value_b1'], shape=(512,), requires_grad=False)
    model.value_w2 = Tensor(params['value_w2'], shape=(512, 1), requires_grad=False)
    model.value_b2 = Tensor(params['value_b2'], shape=(1,), requires_grad=False)
    
    return model

# Get AI move
def get_move(board, model):
    tensor = board_to_tensor(board)
    policy_logits, value = model(Tensor(tensor.data, shape=(1, 12, 8, 8)))
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    legal_indices = [get_policy_index(board, m) for m in legal_moves]
    
    # Normalize logits for numerical stability
    max_logit = max(policy_logits.data[0, idx] for idx in legal_indices)
    probs = [math.exp(policy_logits.data[0, idx] - max_logit) for idx in legal_indices]
    total = sum(probs) + 1e-8
    probs = [p / total for p in probs]
    
    move = random.choices(legal_moves, weights=probs)[0]
    return move

# Load the model
try:
    model = load_model("deepgrad/examples/models/best_model.pkl")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = ChessNet()  # Fallback to uninitialized model

@app.route('/')
def index():
    return render_template('chess.html')

@app.route('/move', methods=['POST'])
def make_move():
    data = request.json
    fen = data.get('fen')
    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({'error': 'Invalid FEN', 'fen': chess.Board().fen()})
    
    # Check game status
    if board.is_game_over():
        result = board.result()
        status = "Game Over: " + ("White wins" if result == "1-0" else "Black wins" if result == "0-1" else "Draw")
        return jsonify({'status': status, 'move': None, 'fen': board.fen()})
    
    # Human move (if provided)
    human_move = data.get('move')
    if human_move:
        try:
            move = chess.Move.from_uci(human_move)
            if move in board.legal_moves:
                board.push(move)
            else:
                return jsonify({'error': 'Illegal move', 'fen': board.fen()})
        except ValueError:
            return jsonify({'error': 'Invalid move format', 'fen': board.fen()})
    
    # AI move
    if not board.is_game_over():
        ai_move = get_move(board, model)
        if ai_move is None:
            return jsonify({'error': 'No legal moves available', 'fen': board.fen()})
        board.push(ai_move)
        return jsonify({'move': ai_move.uci(), 'fen': board.fen()})
    else:
        result = board.result()
        status = "Game Over: " + ("White wins" if result == "1-0" else "Black wins" if result == "0-1" else "Draw")
        return jsonify({'status': status, 'move': None, 'fen': board.fen()})

if __name__ == '__main__':
    app.run(debug=True)