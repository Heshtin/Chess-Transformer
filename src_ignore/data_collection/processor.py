import json


# def read_first_lines(file_path, num_lines=1):
#     with open(file_path, 'r') as file:
#         for i in range(num_lines):
#             line = file.readline()
#             if not line:  # Break if the file has fewer lines than num_lines
#                 break
#             print(line.strip())  # Strip removes any trailing newline characters

# # Example usage
file_path = 'lichess_db_eval.jsonl'  # Replace with your actual file path
# read_first_lines(file_path)

# with open(file_path) as file:
#     lines = file.read().splitlines()


# for i, json in enumerate(lines):
#     lines[i] = json.loads(json)
import sqlite3

def create_database(db_path):
    # Connect to SQLite database at db_path, will create if not exist
    with sqlite3.connect(db_path) as conn:
        # Create a new cursor object using the connection
        cursor = conn.cursor()
        
        # SQL statement to create a new table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chess_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fen TEXT,
                fen_list_string TEXT,
                eval REAL,
                move_prob_string TEXT
            );
        ''')
        
        # Commit changes
        conn.commit()
        print("Database created and table initialized.")

# Specify the path where the database will be created
db_path = 'chess_data.db'
create_database(db_path)

import sqlite3
import json
import chess

def batch_insert_data(db_path, fens, fen_lists, evals, move_prob_strings, batch_size=1000):
    # Open the database connection
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Prepare the batch of tuples for insertion
        batch = []
        for fen, fen_list, Eval, move_prob_string in zip(fens, fen_lists, evals, move_prob_strings):
            # Serialize fen_list and move_prob_string to JSON format
            batch.append((fen, fen_list, Eval, move_prob_string))
            
            # Check if the batch size is reached
            if len(batch) >= batch_size:
                cursor.executemany('''
                    INSERT INTO chess_analysis (fen, fen_list_string, eval, move_prob_string) 
                    VALUES (?, ?, ?, ?)
                ''', batch)
                conn.commit()
                batch = []  # Clear the batch after commit
        
        # Insert any remaining data in the batch
        if batch:
            cursor.executemany('''
                INSERT INTO chess_analysis (fen, fen_list_string, eval, move_prob_string) 
                VALUES (?, ?, ?, ?)
            ''', batch)
            conn.commit()



db_path = 'chess_data.db'







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
    
move_dict_obj = MoveDictionary()
move_to_index = move_dict_obj.move_index_dict
index_to_move = move_dict_obj.index_move_dict




def fen_to_vector(fen):
    fen_parts = fen.split(" ")
    if fen_parts[1] == "b":
        fen_parts[1] = 1
        fen_parts = [part.swapcase() for part in fen_parts]
    else:
        fen_parts[1] = 0
    fen_parts = fen.split(" ")
    castling_rights_dict = {"K":0,"Q":1,"k":2,"q":3,"-":4}
    special_tokens = [0] * 13
    for char in fen_parts[2]:
        special_tokens[castling_rights_dict[char]] = 1
    if fen_parts[3] == "-":
        special_tokens[4] = 0
    else:
        special_tokens[ord(fen_parts[3][0] - 97 + 5)] = 1
    position=""
    piece_dict = {" ":"1,", "p":"2,", "n":"3,", "b":"4,", "r":"5,", "q":"6,", "k":"7,", "P":"8,", "N":"9,", "B":"10,", "R":"11,", "Q":"12,", "K":"13,"}
    for row in fen_parts[0]:
        for square in row:
            if square.isalpha():
                position+=piece_dict[square]
            else:
                position+=int(square)*"0,"
    castling_rights = fen_parts[2]
    position += "1," if "K" in castling_rights else "0,"
    position += "1," if "Q" in castling_rights else "0,"
    position += "1," if "k" in castling_rights else "0,"
    position += "1," if "q" in castling_rights else "0,"
    return position[:-1]



import math

def softmax(scores):
    """Compute the softmax of a list of numbers."""
    # Step 1: Compute the exponent of each score
    exp_scores = [math.exp(score) for score in scores]
    
    # Step 2: Compute the sum of the exponentiated scores
    sum_of_exp = sum(exp_scores)
    
    # Step 3: Divide each exponentiated score by the sum of all exponentiated scores
    softmax_scores = [exp_score / sum_of_exp for exp_score in exp_scores]
    
    return softmax_scores

# Example usage
scores = [2.0, 1.0, 0.1]
softmax_output = softmax(scores)
print("Softmax Output:", softmax_output)





batch_size=10000

file = open(file_path, "r")

for count in range(1):
    fens=[]
    positions=[]
    pos_evals=[]
    move_string=[]
    lines=[]
    for i in range(batch_size):
        lines.append(json.loads(file.readline()))
    for i, line in enumerate(lines):
        fen = line["fen"]
        fen_parts = fen.split(" ")
        turn_multiplier = -1 if fen_parts[1] == "b" else 1
        moves=[]
        evals=[]
        move_dict = {}
        for pvs in line["evals"]:
            for pv in pvs["pvs"]:
                #print(f"{pv=}")
                pv_list = pv["line"].split(" ")
                if len(pv_list)==0:
                    continue
                moves.append(pv_list[0])
                evals.append(turn_multiplier * float(pv["cp"]) / 100.0)
        softmax_evals = softmax(evals)
        moves_dict = {}
        for i in range(len(moves)):
            moves_dict[moves[i]] = softmax_evals[i]
        moves_dict = json.dumps(moves_dict)
        fens.append(fen)
        move_string.append(moves_dict)
        pos_evals.append(max(evals))
        positions.append(fen_to_vector(fen_parts[0], turn_multiplier))
    batch_insert_data(db_path, fens, positions, evals, move_string)


file.close()


