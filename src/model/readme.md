This folder contains all the files necessary to train, validate and play against the model.

chess_model.py:
Defines the overall model

blocks.py:
Defines the subparts of the model (like attention layer, MLP etc)

dataloader.py:
Defines IterableDataLoader class for loading data from the database

dataclass.py:
Defines dataclasses that hold important variables (may contain overlap)
HyperParamConfig - defines hyperparameters
DataConfig - defines variables related to dataloader.py
ChessConfig - defines variables related to the model architecture
RunConfig - defines variables related to the run
VariableRunConfig - defines variables that should be in RunConfig but are frequently changed (for convenience)

auxilliary.py:
Defines supporting functions that dont fit anywhere else, usually related to logging

uci_move_dict.py:
Defines a mapping from every possible uci move in chess to a unique index from 0 to 1967.

train.py:
Training/validation/testing script for the model.

play.py:
Loads a trained model to enable someone to play against it.

