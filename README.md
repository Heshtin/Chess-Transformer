# Chess-Transformer

Welcome to the **Chess Transformer** project repository. This project is an exploration of applying state-of-the-art transformer architectures, specifically a decoder-only model, to the domain of chess. The primary objective is to harness the sequential learning capabilities of transformers to accurately predict subsequent moves in a chess game based on the current board configuration and historical moves.

## Data acquisition

we took 5 million board state from this github repo (https://github.com/TCEC-Chess/tcecgames) and we encoded the the 

64 by 64 board state as a one line 64 array and encoded such that:


| Friendly   | Enemy      |
| ---------- | :--------- |
| 0 - Empty  |            |
| 1 - Pawn   | 7 - Pawn   |
| 2 - Bishop | 8 - Bishop |
| 3 - Knight | 9 - Knight |
| 4 - Rook   | 10 - Rook  |
| 5 - Queen  | 11 - Queen |
| 6 - King   | 12 - King  |

## Training Model 

We are using the decoder model
<p align="center" style="background-color: white; padding: 10px;">
  <img src="https://i.sstatic.net/bWnx0.png" alt="Decoder-Only Transformer Architecture" width="600"/>
</p>

input -> embedding layer -> tensor(B,n_squares,n_emb)=tensor(4096,64,128)
-> attention blocks -> tensor(4096,64,128) -> tensor(B,n_emb,a,a)=tensor(4096,128,8,8)
