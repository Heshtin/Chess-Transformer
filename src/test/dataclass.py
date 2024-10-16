from dataclasses import dataclass

@dataclass
class VariableRunConfig:
    train_type: str = "normal"
    masking: bool = False
    

@dataclass
class HyperParamConfig:
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    

@dataclass
class ChessConfig:
    squares_size: int = 65 # n_squares + 1 for special token
    special_size: int = 13 # n piece embedding types
    vocab_size: int = 27 # 1 special token, 1 empty square, 6 own pieces, 6 opponent pieces, 4 castling rights, 9 en_passant (1st for availabiltiy, other 8 to indicate file)
    n_layer: int = HyperParamConfig.n_layer # [16, 24, 32]
    n_head: int = HyperParamConfig.n_head # [8, 16, 32]
    n_embd: int = HyperParamConfig.n_embd # [128, 256, 512]
    dropout: float = HyperParamConfig.dropout # [0.2, 0.3, 0.4, 0.5]
    n_possible_moves: int = 1968 #no of uci moves (moves in output)
    n_moves_reevaluate: int = 3 #no of moves that are reevaluated in 2nd forward pass
