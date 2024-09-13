# A novel approach to the no-search chess engine

## Intro
Earlier this year, the team at google DeepMind released the first every no-search chess engine (https://arxiv.org/pdf/2402.04494). This engine utilised a self-attention transformer to determine the best move in a position using just the board state through supervised learning. The engine achieved a peak rating of 2895 with its 270 million parameter model. However, as was pointed by a github blog post (https://gist.github.com/yoavg/8b98bbd70eb187cf1852b3485b8cda4f), this engine has several drawbacks, namely:
1) It does not actually learn the rules of the game of chess
2) It is learning moves that maximise probability of winning, but not speed of winning. This may hinder it from making moves that advance the position towards a win
3) It uses supervised learning, so it is basically trying to mimick the output of stockfish, an engine that itself uses search

Here are the changes we are making to our model to resolve the issues above:
1) When we envision that in an engine with no explicit search, the engine would be able to conduct searches internally within the model itself. That is to say, we would hope that a model is able to pick up on the ramifications of a particular move without further explicit play on the move, but we imagine that internally it is looking through specific lines and discarding bad ones until it decides on a good one, in the way a human might (this is not to suggest that the model would actually work like this, but we are hoping for something that produces the same effect as this line of reasoning). To search through specific lines, the engine will require an internal understanding of what moves are legal and what moves are illegal. Hence, the model is allowed to suggest illegal moves during the training phase. These moves will be masked out during actual play.
2) The target values during training take into account how many moves are left to a victory/defeat
3) Reinforcement learning is used

There are some additional changes we made that we felt would help performance:
4) We use symmetry in the input, where for positions in which it is black's turn, we flip the board. We also encode pieces based on own piece or enemy piece, rather than white or black. This is because locking the model in symmetry reduces the complexity of the problem for the model

## Model
We are choosing to use a transformer model as opposed to a CNN. This is because we believe the long-distance communication in self-attention will better be able to capture the patterns in the board state. It also allow for easier comparison to the google deepmind research; we are using very similar model architecture.

The input board is converted to a 71-length list, consisting of a special token (to aggregate information from self-attention), 64 tokens to represent the 64 squares of the chess board, 4 tokens to be binary bits for castling rights, and the last 9 binary tokens represent en-passant availability and file. The special token is encoded as 0. The 64 squares are encoded from 1-13 inclusive, to represent the possible state of the square (1 empty square, 6 of own pieces, 6 of enemy pieces). 

The input is passed through an embedding layer, which feeds into a decoder-only self attention transformer. The resultant embedding at index 0 (from the special token) is passed through a linear layer to output a policy vector with 1968 dimensions, to represent all 1968 possible uci moves in chess. Note these are moves that are possible within the game of chess but may be illegal in this specific position. The output vector undergoes softmax and the move corresponding to the dimension with the highest probability is taken as the move played by the model.

## Training
We train the model using reinforcement learning. We simulate games by having the model play against itself. For each position, the model outputs a policy vector of 1968 dim, with each dim corresponding to a possible (legal or illegal) uci move. The vector undergoes softmax and the dimension corresponding to the highest value is chosen as the move to be played.

During training, after each move, the move is checked to see if it is legal. If it is, the game continues. If not, the move is retracted, and a datapoint is added. This continues until the game terminates, where all positions from the game are added as individual datapoints, and a new game simulation starts. This repeates until we attain enough datapoints for a batch, at which point the model is updated using the batch. The batch size increases throughout training.

for each illegal move, we construct the target policy as such:
all legal moves are assigned a score of 0
all illegal moves are assigned a score of -10
the above vector is constructed and undergoes softmax to achieve the target vector.


for each position in the game, we construct the target policy as such:
all legal moves other than the one played are assigned a score of 0
all llegal moves are assigned a score of -10
for positions where it is the winner's turn:
the move played is optimal and is assigned a value of 4 + 1/(no of moves till victory)
for positions where it is the loser's turn:
the move play is incorrect and is assigned a value of -4 - 1/(no of moves till defeat)
the above vector is constructed and undergoes softmax to achieve the target vector.
these datapoints are added 

*for player moves, we choose 4 and -4 because a value of 4-5 will give a probability of about 0.6-0.75 when no other good moves are present which is high enough. However, it is not too high to the point where it would overshadow other potentially good moves.











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
![Decoder-Only Transformer Architecture](https://i.sstatic.net/bWnx0.png)

input -> embedding layer -> tensor(B,n_squares,n_emb)=tensor(4096,64,128)
-> attention blocks -> tensor(4096,64,128) -> tensor(B,n_emb,a,a)=tensor(4096,128,8,8)


## Results

You can find that our loss  ... 



We would hope that the model would be able to internally search up moves within its calculations rather than explicitly using a search tree. That is to say, we would hope that a model is able to pick up on the ramifications of a particular move without further explicit play on the move. The hope is that the model is complex enough to support this kind of thinking. To do this, the model needs to understand the game rules and what moves are legal. Importantly, it must be able to do this itself, and not by a human guide. To do this, 

the model is trained by reinforcement learning but is allowed to play any one of the 1968 possible uci moves. After each move, the move is checked to see if it is legal. If it is, the game continues. If not, the move is retracted, and a datapoint is added. This continues until a game terminates. We assume that play by the winning player is optimal. So we assign 

we train the model using reinforcement learning. We simulate games by having the model play against itself. For each position, the model outputs a policy vector of 1968 dim, with each dim corresponding to a possible (legal or illegal) uci move. The vector undergoes softmax and the dimension corresponding to the highest value is chosen as the move to be played.

During training, after each move, the move is checked to see if it is legal. If it is, the game continues. If not, the move is retracted, and a datapoint is added. This continues until the game terminates, where all positions from the game are added as individual datapoint, and a new game simulation starts. This repeates until we attain 4096 datapoints (for 1 batch size)

for each illegal move, we construct the target policy as such:
all legal moves are assigned a score of 0
all illegal moves are assigned a score of -10
the above vector is constructed and undergoes softmax to achieve the target vector.


for each position in the game, we construct the target policy as such:
all legal moves other than the one played are assigned a score of 0
all llegal moves are assigned a score of -10
for positions where it is the winner's turn:
the move played is optimal and is assigned a value of 1 + 1/(no of moves till victory)
for positions where it is the loser's turn:
the move play is incorrect and is assigned a value of -1 - 1/(no of moves till defeat)
the above vector is constructed and undergoes softmax to achieve the target vector.
these datapoints are added 


Google deepmind also used separate token embeddings for showing tha t

We would hope that the model would be able to internally search up moves within its calculations rather than explicitly using a search tree. That is to say, we would hope that a model is able to pick up on the ramifications of a particular move without further explicit play on the move. The hope is that the model is complex enough to support this kind of thinking. To do this, the model needs to understand the game rules and what moves are legal. Importantly, it must be able to do this itself, and not by a human guide.

I am building a chess deep learning model with a similar architecture to google deepmind transformer in the Grand-master Level Chess without Search paper. This is how I intend to train it:
the model is trained by reinforcement learning but is allowed to play any one of the 1968 possible uci moves. After each move, the move is checked to see if it is legal. If it is, the game continues. If not, the move is retracted, and a datapoint is added. This continues until a game terminates. We assume that play by the winning player is optimal. So we assign 

we train the model using reinforcement learning. We simulate games by having the model play against itself. For each position, the model outputs a policy vector of 1968 dim, with each dim corresponding to a possible (legal or illegal) uci move. The vector undergoes softmax and the dimension corresponding to the highest value is chosen as the move to be played.

During training, after each move, the move is checked to see if it is legal. If it is, the game continues. If not, the move is retracted, and a datapoint is added. This continues until the game terminates, where all positions from the game are added as individual datapoint, and a new game simulation starts. This repeates until we attain 4096 datapoints (for 1 batch size)

for each illegal move, we construct the target policy as such:
all legal moves are assigned a score of 0
all illegal moves are assigned a score of -10
the above vector is constructed and undergoes softmax to achieve the target vector.


for each position in the game, we construct the target policy as such:
all legal moves other than the one played are assigned a score of 0
the suggested move is assigned a score of -25
all illegal moves are assigned a score of -10
for positions where it is the winner's turn:
the move played is optimal and is assigned a value of 1 + 1/(no of moves till victory)
for positions where it is the loser's turn:
the move play is incorrect and is assigned a value of -1 - 1/(no of moves till defeat)
the above vector is constructed and undergoes softmax to achieve the target vector.

This enables the model to 
1) learn how to distinguish legal from illegal moves. This is because of my own reasoning as follows:
We would hope that the model would be able to internally search up moves within its calculations rather than explicitly using a search tree. That is to say, we would hope that a model is able to pick up on the ramifications of a particular move without further explicit play on the move. The hope is that the model is complex enough to support this kind of thinking. To do this, the model needs to understand the game rules and what moves are legal. Importantly, it must be able to do this itself, and not by a human guide.
2) the model prefers moves that win more quickly and lose more slowly. This is important because the softmaxed values of the policy vector are meant to show the probability of winning if that move is played. But it doesnt take into account in how many moves. This can have pitfalls. For example, in a K+Q vs k endgame, almost any move by white has a 100% chance of winning, but only a few will advance the position towards a win. This may result in the engine shuffling aimlessly. Additionally, it stands to reason that a move that would win more quickly is a stronger move and has a higher chance of yielding a win if played