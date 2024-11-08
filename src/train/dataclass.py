from dataclasses import dataclass


@dataclass
class VariableRunConfig:
    train_steps: int = 150000
    n_limit: int = None
    masking: bool = False
    gpu_batch_size: int = 1024
    token_encoding_scheme: int = 1


@dataclass
class DataConfig:
    n_workers: int = 1
    n1: float = 0.8
    n2: float = 0.1
    
#hyperparameters
@dataclass
class HyperParamConfig:
    total_batch_size: int = 1024
    adamw_weight_decay: float = 0.0
    gradient_clipping: float = 1.0
    max_lr: float = 5e-3
    constant_lr: float = 4e-4
    max_steps: float = 0.80
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 64
    dropout: float = 0.0


@dataclass
class RunConfig():
    total_batch_size: int = HyperParamConfig.total_batch_size # [1024, 4096, 16384]
    adamw_weight_decay = HyperParamConfig.adamw_weight_decay # [1e-3, 1e-4]
    gradient_clipping = HyperParamConfig.gradient_clipping
    max_lr: float = HyperParamConfig.max_lr #[1e-4, 1e-5, 1e-6, 1e-7]
    min_lr: float = 0.01 * max_lr
    warmup_steps: float = 0.0075
    max_steps: float = HyperParamConfig.max_steps #[0.70, 0.75, 0.80]
    

@dataclass
class ChessConfig():
    total_tokens: int = 78
    squares_size: int = 65 # n_squares + 1 for special token
    special_size: int = 13 # n piece embedding types
    vocab_size: int = 13 if VariableRunConfig.token_encoding_scheme < 3 else 9 # 1 special token, 1 empty square, 6 own pieces, 6 opponent pieces, 4 castling rights, 9 en_passant (1st for availabiltiy, other 8 to indicate file)
    n_layer: int = HyperParamConfig.n_layer # [16, 24, 32]
    n_head: int = HyperParamConfig.n_head # [8, 16, 32]
    n_embd: int = HyperParamConfig.n_embd # [128, 256, 512]
    dropout: float = HyperParamConfig.dropout # [0.2, 0.3, 0.4, 0.5]
    n_possible_moves: int = 1968 #no of uci moves (moves in output)
    n_moves_reevaluate: int = 3 #no of moves that are reevaluated in 2nd forward pass
    token_encoding_scheme: int = VariableRunConfig.token_encoding_scheme
