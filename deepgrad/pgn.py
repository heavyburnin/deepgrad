import re
import chess
from deepgrad.tensor import Tensor

class PGNParser:
    def __init__(self, pgn_file_path):
        self.pgn_file_path = pgn_file_path
        self.games = self.parse_pgn_file()

    def parse_pgn_file(self):
        """
        Parse the PGN file to extract FEN strings and corresponding outcomes.
        """
        games = []
        with open(self.pgn_file_path, 'r') as pgn_file:
            pgn_data = pgn_file.read()

        # Split the PGN data into individual games using PGN tags
        raw_games = pgn_data.split("\n\n")
        for game in raw_games:
            if game.strip():  # Avoid empty blocks
                game_info = self.parse_single_game(game)
                if game_info:
                    games.append(game_info)
        return games

    def parse_single_game(self, game):
        """
        Parse a single PGN game into FEN strings and outcome.
        """
        try:
            game_lines = game.split("\n")
            # Extract the header (metadata) and moves
            header = game_lines[:6]  # PGN headers
            moves = game_lines[6:]  # PGN moves

            # Get the game outcome from the header
            result = self.extract_result(header)

            # Generate the FEN for each move
            fen_sequence = self.get_fen_from_moves(moves)
            
            if not fen_sequence:
                return None

            return {
                'fen_sequence': fen_sequence,
                'result': result
            }
        except Exception as e:
            print(f"Error parsing game: {e}")
            return None

    def extract_result(self, header):
        """
        Extract the result of the game (win/loss/draw) from PGN header.
        """
        result_pattern = re.compile(r"\[Result \"(\w+)\"\]")
        for line in header:
            match = result_pattern.search(line)
            if match:
                return match.group(1)
        return "draw"  # Default result

    def get_fen_from_moves(self, moves):
        """
        Convert the moves to FEN after each move.
        """
        board = chess.Board()
        fen_sequence = []

        for move in moves:
            # Strip any comments or extraneous data
            move = move.split(' ')[0]  # Get the move without annotations

            # Ignore lines that aren't actual moves (metadata like [UTCTime], etc.)
            if move.startswith("["):
                continue  # Skip metadata lines

            if move:
                try:
                    board.push_san(move)  # Apply the move
                    fen_sequence.append(board.fen())  # Save FEN after the move
                except ValueError as e:
                    print(f"Invalid move in PGN: {move} (Error: {str(e)})")
                    continue  # Skip invalid moves

        print(f"Generated FEN sequence: {fen_sequence}")  # Debugging FEN sequence
        return fen_sequence

    def to_tensor(self, game_info, input_size):
        """
        Convert the parsed game info (FEN sequence) to a format suitable for the model.
        """
        tensor_sequence = []
        for fen in game_info['fen_sequence']:
            tensor_sequence.append(self.fen_to_tensor(fen, input_size))

        return tensor_sequence

    def fen_to_tensor(self, fen, input_size):
        """
        Convert a FEN string to a tensor. The tensor format should match
        the expected input shape for the model.
        """
        board = chess.Board(fen)
        board_array = self.fen_string_to_tensor(board.fen())
        return Tensor(board_array, requires_grad=False)

    def fen_string_to_tensor(self, fen):
        """
        Convert a FEN string into a 1D tensor representation of the board state.
        """
        rows = fen.split(' ')[0].split('/')  # Get the board part of FEN
        board = []

        for row in rows:
            for char in row:
                if char.isdigit():
                    board.extend([0] * int(char))  # Empty squares
                elif char.islower():  # Black piece
                    board.append(-1)  # For simplicity, map black pieces as -1
                elif char.isupper():  # White piece
                    board.append(1)  # Map white pieces as 1

        # Ensure the board is always 64 squares long (8x8 board)
        return board[:64]

    def get_all_games_as_tensors(self, input_size):
        """
        Convert all games to tensors for training.
        """
        tensor_data = []
        for game in self.games:
            print(f"Parsing game: {game}")  # Add this to debug
            tensor_sequence = self.to_tensor(game, input_size)
            if tensor_sequence:  # Only add if sequence is valid
                tensor_data.append({
                    'fen_sequence': tensor_sequence,
                    'result': game['result']
                })
            else:
                print(f"Invalid game: {game}")  # Debugging invalid games
        return tensor_data
