import chess
board = chess.board()
def list_to_sorted_dict(input_list): #descending order
    return {element: index for index, element in sorted(enumerate(input_list), key=lambda x: x[1], reverse=True)}

class DataInterface:
    pass

class MoveDictionary:
    pass
data_interface = DataInterface()
move_dict_obj = MoveDictionary()
move_to_index, index_to_move = move_dict_obj.move_to_index, move_dict_obj.index_to_move

while True:
    while data_interface.current_size < data_interface.batch_size:
        model_outputs = []
        game_move_indices = []
        game_legal_moves = []
        n_moves = 0
        while not board.evaluate_terminal_state() and n<100:
            policy = API.call_model(board.fen())
            policy_list = policy.tolist()
            sorted_policy = list_to_sorted_dict(policy_list)
            legal_moves = [move_to_index[move] for move in board.legal_moves()]
            for key, value in sorted_policy.items():
                if value in legal_moves:
                    model_outputs.append(policy_list)
                    game_move_indices.append(move_index)
                    game_legal_moves.append(legal_moves)
                    board.push(index_to_move[value])
                    break
                else:
                    data_interface.add_data_illegal(policy, data_interface.construct_illegal_target_policy(value, legal_moves)) #value=move_index
            n += 1
        
        game_result = board.evaluate_terminal_state() if board.evaluate_terminal_state() else 0 # game_result = value based on terminal state if terminal state reached else 0
        data_interface.add_data_game(game_result, model_outputs, game_move_indices, game_legal_moves)
    API.call_model.train(data_interface.batch)





    