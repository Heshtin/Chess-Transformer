# A novel approach to the no-search chess engine

## Intro

## Model
The model architecture extends and also slightly modifies the vision transformer based architecture used by google Deepmind to create the first chess engine without explicit search (https://arxiv.org/pdf/2402.04494). It aims to further refine the input as well as introduce a refinement technique while maintaining no explicit search.

Each input consists of 84 tokens. The first token is a CLS token for aggregating all the information from the attention blocks. After the forward pass through the transformer, this CLS token alone is passed through a final linear layer fc1 to output a vector that undergoes softmax to produce a policy vector that is a probability distribution across all 1968 possible uci moves in chess (possible moves, not just legal moves). There is no masking of illegal moves, as explained in the next section. Additionally, we want the model to be able to learn the consequences of its moves to future positions, specifically to castling and en passant rights (further explained in the next section). As such, the CLS token in passed through another linear layer fc3 to output an 11 dimensional encoding, where the target for each dimension is a binary encoding of a castling or en passant right.

The next 64 tokens represent the contents of each square. There are 13 possible states for each square (1 for empty, 6 for white piece and 6 for black pieces). Each of these 13 states have their own learned embedding. We also use learned positional embeddings. We use relative rather than absolute positional embeddings following work by the Leela Chess Zero team (https://arxiv.org/html/2409.12272v1).

The next token is a binary token to indicate whose turn it is.

The next 4 tokens are binary tokens to represent castling rights. The next 9 tokens are also binary tokens used to encode en passant rights, with the first token representing whether en passant is available or not, and the next 8 to encode which of the 8 files the en passant square is on. The first of these tokens is a form of redundancy that is introduced as a way to provide a stronger signal that we hope helps the model learn about en passant more quickly, as we fear the rarity of this move being played may make it difficult to learn.

The next token corresponds to the number of times the current position has been reached before in the game, to help with understanding of 3 fold repetition. We define a set of 4 embeddings for this, to represent that the position has occured 0, 1 or 2 times (including this time). Although it is impossible for the position to have occured 3 times (as it would have resulted in a draw), these embeddings are reused later and will require the 4th embedding.

The next token corresponds to the number of half moves. We define another embedding called the half-move embedding and derive the token as follows:

half_move_token = (number of half moves / 50.0) * half_move_embedding

However, linearly scaling using (number of half moves / 50.0) may not be the most optimal approach. The number of half moves mainly convery information regarding the number of moves left until a draw is declared if no captures or pawn moves are made. The significance of this is not linearly related to the number of half moves. For example, the difference in positional significance between 9 and 19 half moves isnt much, but it is a lot for 39 and 49, since for 49 half moves the position is drawn on the very next move. This may lead us to use an exponential relationship. However, the half move counter can also help indicate other information like whether the position is fixed or a fortress is present, which may not be the most ideal. Hence, our model aims to use a polynomial of a small degree (8) with learnable coefficients, where x = (number of half moves / 50.0). This allows the model itself to determine the most optimal relationship between positional significance and number of half moves. We will also consider passing the scalar through ReLU to prevent negative values, as this would cause the direction of the embedding to flip which could potentially confuse the model.

The next token corresponds to the number of full moves. Rather than simply scaling a single defined embedding, we scale a smaller embedding with 16 dimensions then matrix multiply with a 16 x n_token_embd matrix, as follows:

full_move_token = ((number of full moves / 100.0) * full_move embedding) x full_move_matrix

Due to the complexity of matrix multiplication after linear scaling, we feel that adding complexity to the way of scaling is not required as it was in the half-move counter token.

The reason for the difference in the way half and full moves are handled is as follows. The information from the number of half moves is simply how far away the game is from being declared a draw if no pawn moves are captured or made. It could potentially also give more game-specific information rather than just how far from a draw they are, like being in a fixed/closed position. However this again sounds more like a continuous rather than categorical classification. This information/signal has a very clear, singular meaning, that just needs to vary in strength, ie it is the same type of information that just varies between 2 extremes, in the same way that a number line encodes the same type of information (real numbers) with 2 extremes (-infinity, infinity). Hence, we can simply scale the vector to vary its magnitude while keeping it pointing in the same direction which would likely imply the same information (borrowing the concept of how directions of embeddings in LLMs encode meaning). However, the information from the full move vector is more aimed at telling the model which stage of the game we are in. The game is generally classified as 3 different stages (opening, middlegame, endgame) that all have different strategies attached to them, which is a categorical rather than continuous encoding. Assuming the engine uses categorical classification as well (which indeed makes more sense), we should use tokens that have different directions in the vector space to encode different information/meaning.

We believe that this encoding of the half and full move vectors is better than the one used by deepmind (where they used 10 embeddings to encode each digit in the numbers) and will produce better results. This is a novel approach of the paper.

The purpose of the next 3 tokens, which are not present in deepmind's model, is iterative refinement. We generate an embedding for each of the 1968 possible uci moves, and an additional one called the no-move embedding. During the first forward pass, we use the no-move embedding for all 3 of these tokens. We will then use the output policy and extract the indices of the moves with the top 3 probabilities. During the second forward pass, we use the same input for the first 78 tokens, but this time the last 3 tokens will be replaced by their respective uci tokens, added to the appropriate 3-move embeddings, using the same set of embeddings used in the initial input. We also define another embedding called the scaling embedding. This is used to pass information about the probability of the move in the policy, as follows:

input_uci_embedding = uci_embedding + 3_fold_repitition_embedding + (prob_move) * (scaling_embedding)

We extract the CLS token from the output of the transformer from the second forward pass and pass it through a different final linear layer fc2, to output a policy with just 3 values, corresponding to each of the moves. However there needs to be positional information about the input moves, ie the model needs to know which neuron in the output of fc2 corresponds to which token. Hence we need an additional 3 tokens for positional (or more aptly, "rank") embedding. The formula for the 3 input embeddings is thus modified as follows:

input_uci_embedding = uci_embedding + 3_fold_repitition_embedding + (prob_move) * (scaling_embedding) + rank_embedding


The purpose of these additional 3 tokens and fc2 is to allow the model to reconsider its moves by directly comparing the top options, which is especially important in situations where the probabilities of the moves are close.  Since we are no longer just blindly choosing the top move this means we will need to test out different values for the temperature of the softmax to achieve the policy vector.

This move-refinement mechanism while maintaining explicit search is the main novelty that this research is introducing to the study of chess engines.

Finally, we believe the use of the 3_fold_repitition embeddings give the engine an idea of the repitition of the current position and the future positions of only the moves being considered introduces a stronger way to deal with the three-fold repition rule. Of course it doesnt give the model the complete information about the game history, and also doesnt include consideration for all the legal moves that werent considered, since including the move embeddings for all possible uci moves in the input would be too computationally expensive. However, we feel that this approach still gives sufficient information to make the most informed decision in most cases. The use of 3 fold repetition embeddings as described is another novelty of this research.

The architecture for the attention is similar to those used in LLMs. We use 8 attention heads and the MLP block expands the embedding dimension to 4 times its initial value. 

## Evidence of implicit search vs pattern recognition
In this research, since the model does not undergo any explicit/hand-crafted search, we also investigate whether the model is learning to play by conducting an "implicit" search within itself or purely by pattern recognition of the relative piece positions on the board. This is important because understanding how the model learns/works can motivate better design choices/optimizations for the model architecture, leading to faster convergence and stronger play.

Our investigation is two fold. Firstly, we manipulate the architecture to give and withhold information about how the game works from the model. Without this information, the model will be forced to use pattern recognition. However, with this information the model can conduct an implicit search. At the same time, the presence of this information should not significantly impact the model's pattern recognition ability, since pattern recognition is just by analyzing the current (static) board state. From this we can concur that if we observe that allowing the model to learn certain information about how the game works improves performance, the improvement is likely attributed to internal search.

We control things the model is able to learn in the following ways:
1) mask/no mask
We compare the effect of masking vs not masking illegal moves in the output of the transformer before softmax. Teaching the model to assign near 0 probabilities to illegal moves and thereby learn what illegal moves are is crucial for an implicit search to occur.

2) flip/no flip
We compare 2 different input schemes. The first one is as described in the previous section, where the pieces embeddings are created for each piece type by whether they are white or black. The players turn is then input as a binary token. A second scheme involves treating the position from the perspective of the current player. This means creating piece embeddings for each piece type based on whether they belong to the "current player" or "enemy", and the board is vertically flipped. No turn token is required since it is the current players turn. The second scheme actually prevents proper implicit search, because it does not allow the model to learn how enemy pieces move, since it always predicts how the current player moves. Enemy pieces may very well work the same as the player's pieces but the model cannot conclude that from the data. It should be noted that using the player/enemy scheme would reduce the complexity of the problem by utilizing symmetry and make pattern recognition even stronger. Hence, if the first scheme outperforms the second, it provides even stronger evidence of an implicit search.

3) flipping while using 9 embeddings scheme
This experiment compares the same 2 types of schemes as the last, the first labelling the pieces as black and white and using a turn token, and the second labelling the pieces as belonging to player and enemy and flipping the board. But instead of using 13 embeddings, it uses just 9, 1 for empty square, 6 for each piece type, and 2 either for white/black or player/enemy. The idea is that we conjecture that most of the information about the movement/dynamics of the piece itself will be completely or mostly stored in the 6 type embeddings. It is unlikely to be stored in the 2 colour/person embeddings since movement and piece dynamics is individual to every piece. Hence, when these embeddings are learned using data for the most optimal move for the current player, this information can also be used for enemy pieces. This should hence now allow for implicit search even if labelling pieces by player/enemy. Thus, the gap between the 2 schemes should decrease. We first combine the type and colour/player embeddings by addition, but rerun the experiment by just inputting them as 2 different tokens, hence increasing the number of tokens encoding the position from 64 to 128.

4) Understanding how moves affect castling and en passant rights
To conduct an implicit search, the model needs to understand how its current move affects the future board state. Hence, it needs to understand how moves affect castling and en passant rights. To allow this we experiment with passing the CLS token through the another linear layer to output an 13 dimension vector, wherethe scalar at each dimension corresponds to a castling or en passant right in the same way that this information is encoded as 13 tokens in the input.


Secondly, we curate specific test uncommon positions that require deep search. However, the search sequences have many forcing moves and do not significantly "branch out", ie the implicit search tree would be of significant depth but not breadth. We believe these types of sequences would not be easily found by pattern recognition, but would be manageable by searching since forcing moves means fewer nodes.

It should also be noted that implicit search can only arise with sufficient "space" (ie embedding dimension) and depth (ie number of attention blocks). We hypothesize that the model relies more on pattern recognition at low embedding dimension and number of attention blocks, and gradually conducts deeper and broader implicit searches as these values increase. Hence, we perform the above experiments for a range of embedding dimensions and number of attention blocks, and compare the results.

## 





































## Datasets
We employ curriculum learning. We feel that this would be better than directly exposing the model to very complex patterns, while being much less computationally expensive than reinforcement learning. Datasets are generated from game shards on lichess.com. The moves played in the game are taken to be the target move. We take the average rating of the players to determine which rating range they fall into, and only use games with time controls of at least 15 minutes to ensure data quality is not low.

The data is split into the following rating buckets: 500-1000, 1000-1500, 1500-2000, 2000-2500, and 2500+. They contain 10 million, 40 million, 50 million, 100 million and 250 million datapoints respectively. The first 4 buckets are run for 1 epoch each, and the remaining steps are run for the final bucket.



## Training










