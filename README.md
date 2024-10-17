# A Novel Approach to the No-Search Chess Engine
hello

## Introduction

### Model

The model architecture extends and slightly modifies the Vision Transformer-based architecture used by Google DeepMind to create the first chess engine without explicit search ([DeepMind paper](https://arxiv.org/pdf/2402.04494)). It aims to further refine the input and introduce a refinement technique while maintaining no explicit search.

### Input Tokens

- **84 tokens** in total:
  - The first token is a **CLS token** for aggregating information from attention blocks.
  - After passing through the transformer, the **CLS token** is processed by a linear layer (**fc1**) to output a vector, which undergoes **softmax** to produce a policy vector across all **1968 UCI moves** (possible moves, not just legal moves).
  - The model learns the consequences of moves related to **castling** and **en passant** rights, which is explained in the next section. The CLS token is passed through another linear layer (**fc3**) to output an **13-dimensional encoding**, representing the **castling** or **en passant** rights.

- **64 tokens** represent the chessboard squares. Each square can be in one of **13 possible states**:
  - 1 for **empty**.
  - 6 for **white pieces**.
  - 6 for **black pieces**.
  
  These states have **learned embeddings**. Additionally, we use **relative positional embeddings**, following work by the Leela Chess Zero team ([Leela Chess Zero paper](https://arxiv.org/html/2409.12272v1)).

- **Next token**: a binary token indicating whose turn it is.

- **Next 4 tokens**: binary tokens representing castling rights.

- **Next 9 tokens**: binary tokens encoding **en passant** rights:
  - The first token indicates whether **en passant** is available.
  - The remaining 8 tokens encode which file the **en passant** square is on.

- **Next token**: represents the number of times the current position has occurred during the game, aiding in detecting **three-fold repetition**. There are **4 embeddings** representing:
  - 0, 1, or 2 occurrences.
  - The 4th embedding accounts for matrix reuse.

- **Next token**: represents the number of **half moves**. The model uses a **polynomial of degree 8** to scale the token based on the number of half moves:
  - The formula for scaling is:

    ```plaintext
    half_move_token = (number of half moves / 50.0) * half_move_embedding
    ```

  - This scaling method is refined to account for the **non-linear significance** of the number of half moves as it approaches the 50-move draw rule.

- **Next token**: represents the number of **full moves**. This is handled by scaling a smaller **16-dimensional embedding** using matrix multiplication:

    ```plaintext
    full_move_token = ((number of full moves / 100.0) * full_move embedding) x full_move_matrix
    ```

  - Unlike half moves, **full moves** are more categorical, representing the stage of the game (opening, middlegame, endgame). The scaling helps the model understand which phase of the game it is in.

### Iterative Refinement Mechanism

- The final 3 tokens are used for **iterative refinement**. Initially, a **no-move embedding** is used for these tokens during the first forward pass.
- The model then identifies the **top 3 moves** with the highest probabilities.
- In the second forward pass, these top 3 moves are passed in, replacing the no-move tokens, along with their **scaling embedding** based on the probability of each move:

    ```plaintext
    input_uci_embedding = uci_embedding + 3_fold_repetition_embedding + (prob_move) * (scaling_embedding)
    ```

- The output of the **second forward pass** produces a policy with **3 values**, corresponding to each of the 3 moves. An additional **rank embedding** is introduced to encode the positional information about these moves.

This refinement mechanism allows the model to reconsider its top moves directly, avoiding the issue of choosing blindly based on probability alone.

### Additional Features

- The **3-fold repetition embeddings** provide insight into the repetition of the current position and the positions of the considered moves. This technique helps the model adhere to the **three-fold repetition rule** without considering all legal moves, as that would be computationally expensive.

### Attention Architecture

The attention architecture is similar to that used in **large language models (LLMs)**, with:
- **8 attention heads**.
- The **MLP block** expands the embedding dimension to **4x** its initial size.

## Implicit search architecture
To aid implicit search, we are also considering a mechanism where outputs of later attention blocks are fed back into earlier blocks. This is to simulate the nature or search where one performs a series of calculations, and if it doesn't work, goes back and tries again. What layer is fed back into what layer and for how many cycles could be determined using evolution.

---

# Evidence of Implicit Search vs Pattern Recognition

In this research, since the model does not undergo any explicit or hand-crafted search, we investigate whether the model is learning to play by conducting an **implicit search** within itself or purely by **pattern recognition** of the relative piece positions on the board. Understanding how the model learns can guide us in making better design choices and optimizations for the model architecture, leading to faster convergence and stronger play.

## Investigation

Our investigation is twofold:

### 1. Manipulating Model Information

We manipulate the architecture to give or withhold information about how the game works. Without this information, the model relies on **pattern recognition**, whereas with this information, it can perform an **implicit search**. The presence of this information should not significantly impact the model’s pattern recognition ability, as pattern recognition is based solely on the current (static) board state.

If we observe an improvement in performance when the model learns game-specific information, it suggests that the improvement is likely due to internal search.

### Control Mechanisms

We control what the model learns using the following methods:

- **Mask vs. No Mask**:  
  We compare the effect of masking vs. not masking **illegal moves** in the transformer's output before softmax. Teaching the model to assign near-zero probabilities to illegal moves is crucial for enabling an **implicit search**.

- **Flip vs. No Flip**:  
  We compare two different input schemes:
  1. **Scheme 1**: Piece embeddings are created based on whether pieces are white or black, and the player’s turn is represented by a binary token.
  2. **Scheme 2**: The board is flipped, and pieces are labeled as **current player** or **enemy** with no turn token needed. This scheme prevents proper implicit search since it doesn't allow the model to learn how enemy pieces move.

  The second scheme uses **symmetry** to reduce complexity, thus improving pattern recognition. If **Scheme 1** outperforms **Scheme 2**, it provides stronger evidence for implicit search.

- **Flipping with 9 Embeddings Scheme**:  
  We compare the same two schemes, but instead of using 13 embeddings, we use **9 embeddings**:
  - 1 for **empty square**
  - 6 for each piece type
  - 2 for **white/black** or **player/enemy**

  The hypothesis is that most information about piece dynamics is encoded in the **6 type embeddings**, not the color/player embeddings. If this holds, implicit search could still occur even when pieces are labeled by player/enemy. We run experiments combining type and color/player embeddings by addition and by inputting them as separate tokens, increasing the token count from 64 to 128.

- **Understanding Castling and En Passant Rights**:  
  For implicit search to occur, the model must understand how its moves affect the future board state, particularly **castling** and **en passant rights**. To explore this, we pass the CLS token through another linear layer to output a **13-dimensional vector**, where each dimension corresponds to a castling or en passant right (as encoded in the input).

### 2. Testing with Uncommon Positions

We curate specific test positions that require **deep search**. These positions involve **forcing move sequences** with minimal branching, meaning the implicit search tree would be deep but not broad. Such positions are challenging for pattern recognition but manageable for search due to the forcing nature of the moves.

## Hypotheses

- **Embedding Dimension and Attention Blocks**:  
  Implicit search only arises with sufficient **embedding dimension** and **attention block depth**. At low embedding dimensions and block numbers, the model relies more on **pattern recognition**. As these values increase, the model is expected to perform deeper and broader implicit searches.

### Experimental Range

We perform the above experiments for a range of **embedding dimensions** and **number of attention blocks** and compare the results to observe when and how implicit search begins to dominate over pattern recognition.

---

# Attention Patching

In the paper *"Evidence of Learned Look-Ahead in a Chess-Playing Neural Network"* ([link to paper](https://arxiv.org/pdf/2406.00877)), a technique called **activation patching** is used to check the importance of each embedding to the ultimate output of the chess transformer. They analyzed **Leela Chess Zero's** model, which uses a slightly different approach to calculating the output. 

## Model Architecture in Leela Chess Zero

Leela Chess Zero’s transformer takes **64 tokens** as input, each representing a specific square on the chessboard. The model’s output embeddings correspond to these squares, and the embeddings for the **start and end squares** of each possible move are used to calculate the logit for the **UCI move**. Afterward, the logits undergo **softmax** to produce the policy vector. 

### Board Flipping and Color Swap

Leela Chess Zero uses **board flipping** and **piece color swapping** when processing positions where it is black to move. This approach prevents the model from learning how black pieces move, severely impeding its ability to **look ahead**.

## Activation Patching in Leela Chess Zero

The paper used **activation patching** to analyze specific positions in Leela Chess Zero’s model. The technique involves "patching in" **corrupted activations** for embeddings corresponding to specific squares and allowing the forward pass to proceed normally. The output is then analyzed for deviations from the original output.

### Key Findings

1. **Patching the target square** of either the **1st** (immediate move) or **3rd** move (after opponent’s response) **significantly altered** the output, showing these embeddings were crucial for accurate prediction.
2. **Patching the 2nd move** (opponent’s immediate response) **did not significantly change** the output, an unexpected result that the authors could not fully explain.

### Hypothesis on Learning Black Piece Movements

Our theory is that once the model learns how white pieces move, it **implicitly** gains an understanding of how black pieces move by observing white’s responses to black. For instance, if white does not place a rook on a diagonal with a black bishop, the model learns that **bishops influence diagonally**. This understanding is weaker than knowing that the bishop **can move diagonally and capture**, but it still provides some insight.

This slower, weaker learning of black piece dynamics leads to a **weak implicit search**. While this is not an issue for models like Leela Chess Zero that use **explicit search**, it severely hinders the performance of models without explicit search. This could explain why patching the 2nd move did not significantly affect output: the model’s understanding of black’s moves is weaker.

## Future Experiment

We intend to recreate a similar experiment using our model and test this hypothesis, analyzing how the understanding of black's moves develops without explicit search.

---

## Datasets

We employ **curriculum learning**, as we believe it to be more effective than directly exposing the model to very complex patterns, and it is less computationally expensive than **reinforcement learning**.

### Dataset Sources

Datasets are generated from **game shards** on [lichess.com](https://lichess.com). The moves played in each game are considered the **target move**. We use the **average rating** of the players to categorize games into different rating ranges, and we only include games with time controls of at least 15 minutes to ensure data quality.

### Rating Buckets

The data is split into the following rating buckets:

- **1000-1500**: 50 million datapoints
- **1500-2000**: 50 million datapoints
- **2000-2500**: 100 million datapoints
- **2500+**: 100 million datapoints

The first three buckets are run for **1 epoch** each, while the remaining steps are run for the **2500+ bucket**.

## Training













