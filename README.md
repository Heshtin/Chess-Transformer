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

The input board is converted to a 78-length list, consisting of a special token (to aggregate information from self-attention), 64 tokens to represent the 64 squares of the chess board, 4 tokens to be binary bits for castling rights, and the last 9 binary tokens represent en-passant availability and file. The special token is encoded as 0. The 64 squares are encoded from 1-13 inclusive, to represent the possible state of the square (1 empty square, 6 of own pieces, 6 of enemy pieces). 

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
all illegal moves are assigned a score of -10
for positions where it is the winner's turn:
the move played is optimal and is assigned a value of 4 + 1/(no of moves till victory)
for positions where it is the loser's turn:
the move play is incorrect and is assigned a value of -4 - 1/(no of moves till defeat)
the above vector is constructed and undergoes softmax to achieve the target vector.

Note: for player moves, we choose 4 and -4 because a value of 4-5 will give a probability of about 0.6-0.75 after softmax (assuming 30 legal moves) when no other good moves are present which is high enough. However, it is not too high to the point where it would overshadow other potentially good moves.

## Understanding castling and en passant
Right now, the model only understands castling rights and en passant based on the input embedding bits designated for castling and en_passant. But since we are always telling the model in the input whether castling and en passant is possible or not. So the model never learns how to tell whether a position has castling rights or en_passant rights, since this is based on past moves. This means that the model cannot do implicit search since it cannot tell how moves it is calculating affects castling rights and en passant rights. 

Solution: 
The special embedding is duplicated and independently passed through other linear layers to output vectors that predict whether the castling and en passant rights for the next move are valid. The target vector is a vector of the same size as the embedding that is filled with 1s to represent the rights are still valid, or 0 to represent to rights are no longer valid.