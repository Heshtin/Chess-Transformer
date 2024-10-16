import chess

class MoveDictionary:
    def __init__(self):
        all_moves = self.generate_all_moves()
        self.move_index_dict = {move: index for index, move in enumerate(all_moves)}
        self.index_move_dict = {index: move for index, move in enumerate(all_moves)}
        #return move_index_dict


    def get_all_legal_moves(self, fen):
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)  # Get an iterator of legal moves and convert to a list
        moves = [move.uci() for move in legal_moves]
        return [self.move_index_dict[move] for move in moves]

    def generate_all_squares(self):
        files = 'abcdefgh'
        ranks = '12345678'
        return [f + r for f in files for r in ranks]

    def is_within_board(self, file, rank):
        return 'a' <= file <= 'h' and '1' <= rank <= '8'

    def move_in_direction(self, start_square, file_step, rank_step, steps=8):
        moves = []
        start_file, start_rank = start_square[0], start_square[1]
        for step in range(1, steps + 1):
            new_file = chr(ord(start_file) + file_step * step)
            new_rank = chr(ord(start_rank) + rank_step * step)
            if self.is_within_board(new_file, new_rank):
                moves.append(new_file + new_rank)
            else:
                break
        return moves

    def generate_fairy_moves(self, start_square):
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Rook-like moves
            (1, 1), (1, -1), (-1, 1), (-1, -1),  # Bishop-like moves
            (2, 1), (2, -1), (-2, 1), (-2, -1),  # Knight-like moves
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        moves = []
        for file_step, rank_step in directions:
            if abs(file_step) == 2 or abs(rank_step) == 2:  # Knight-like moves
                moves.extend(self.move_in_direction(start_square, file_step, rank_step, steps=1))
            else:
                moves.extend(self.move_in_direction(start_square, file_step, rank_step))
        return moves

    def generate_promotion_moves(self, start_square, end_square):
        promotion_pieces = ['b', 'n', 'r', 'q']
        return [start_square + end_square + piece for piece in promotion_pieces]

    def generate_all_moves(self):
        all_squares = self.generate_all_squares()
        all_moves = []

        for start_square in all_squares:
            fairy_moves = self.generate_fairy_moves(start_square)
            for end_square in fairy_moves:
                all_moves.append(start_square + end_square)
                # Add promotion moves for pawns
                if start_square[1] == '7' and end_square[1] == '8' and abs(int(ord(start_square[0]))-int(ord(end_square[0]))) <= 1:  # White pawn promotion
                    all_moves.extend(self.generate_promotion_moves(start_square, end_square))
                if start_square[1] == '2' and end_square[1] == '1' and abs(int(ord(start_square[0]))-int(ord(end_square[0]))) <= 1:  # Black pawn promotion
                    all_moves.extend(self.generate_promotion_moves(start_square, end_square))
        return all_moves
    