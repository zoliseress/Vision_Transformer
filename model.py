
import torch
import torch.nn as nn

from utils import get_device, get_positional_encodings, make_patches


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention module.
    
    For a single image each patch gets updated based on some similarity measure
    with the other patches.

    Notes
    -----
        This is the implementation of sub-figure (c) in the model_architecture.png.
    """

    def __init__(self, dim: int, num_heads: int = 2):
        """Init.

        Parameters
        ----------
        dim : int
            Dimensionality of the input sequences (token dimension).
        num_heads : int, optional
            Number of attention heads, by default 2.
        """

        super(MultiHeadSelfAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads

        # Linearly maps each patch to 3 distinct vectors: q, k, v (query, key, value).
        # This is done for each head separately.
        self.dim_head = self.dim // self.num_heads
        # The Query vector represents what a token (patch) is seeking in the context of
        # other tokens (patches).
        self.q = nn.ModuleList(
            [nn.Linear(self.dim_head, self.dim_head) for _ in range(self.num_heads)]
        )
        # The Key vector encodes information about a token (patch) in a way that makes
        # it searchable, acting as a descriptor of what the token (patch) can offer or
        # contribute.
        self.k = nn.ModuleList(
            [nn.Linear(self.dim_head, self.dim_head) for _ in range(self.num_heads)]
        )
        # The Value vector contains the actual semantic information or content that the
        # token (patch) provides, which will be used to update the representation of
        # the querying token (patch).
        self.v = nn.ModuleList(
            [nn.Linear(self.dim_head, self.dim_head) for _ in range(self.num_heads)]
        )

        self.softmax = nn.Softmax(dim=-1)

        # Output projection to learn the weighted combination of the attention heads.
        self.last = nn.Linear(self.dim, self.dim)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        sequences : torch.Tensor
            Input patches as sequences of shape (batch size, seq_length, token_dim).

        Returns
        -------
        torch.Tensor
            Attention mapped sequences of the same shape as the input.
        """

        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.num_heads):
                
                # Map each patch to distinct q, k, v vectors: 

                q = self.q[head]
                k = self.k[head]
                v = self.v[head]

                # Slicing the sequence to extract the portion of the embedding that
                # belongs to the current attention head. E.g. head 0 gets columns 0–3
                # (sequence[:, 0:4]) and head 1 gets columns 4–7 (sequence[:, 4:8]).
                seq = sequence[:, head * self.dim_head: (head + 1) * self.dim_head]
                q, k, v = q(seq), k(seq), v(seq)

                # For a single patch, compute the dot product between its q vector
                # with all of the k vectors, and divide by the square root of the
                # dimensionality of these vectors to get the "attention cues".
                # This is a similarity matrix showing how much each token (patch)
                # attends to every other token (patch).
                attn = q @ k.T / (self.dim_head ** 0.5)  # (seq_length, seq_length)
                attn = self.softmax(attn)
                attn = attn @ v  # (seq_length, dim_head)

                seq_result.append(attn)

            result.append(torch.hstack(seq_result))  # (N, seq_length, self.dim)

        # Concatenate results (combine attention heads).
        output = torch.cat([torch.unsqueeze(r, dim=0) for r in result])

        # The final linear layer.
        output = self.last(output)

        return output


class VisionTransformerBlock(nn.Module):
    """Vision Transformer Encoder.
         - First layer normalization
         - Multi-headed self attention
         - First residual connection
         - Second layer normalization
         - Multi-Layer Perceptron
         - Second residual connection
    
    Notes
    -----
        This is the implementation of sub-figure (b) in the model_architecture.png.
        (Not in 100%.)
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4):
        """Init.

        Parameters
        ----------
        hidden_dim : int
             Dimensionality of the encoder layers and the pooler layer.
        num_heads : int
            Number of attention heads for each attention layer in the encoder.
        mlp_ratio : int, optional
            The inner layer of the classification MLP is multiplied by this value.
            By default 4.
        """
        
        super(VisionTransformerBlock, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Subtracts the mean and divides by the standard deviation for
        # stabilizing the training. (Layer normalization is applied to
        # the last dimension only.)
        self.norm1 = nn.LayerNorm(self.hidden_dim)

        # Multi-Head Self Attention for modeling the relations between the patches.
        self.mhsa = MultiHeadSelfAttention(hidden_dim, num_heads)

        # Second layer normalization.
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        # Multi Layer Perceptron for classification.
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_dim, hidden_dim),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        With Pre-Layer Normalization (differs from the architecture image).

        Parameters
        ----------
        images : torch.Tensor
            The input images. Its shape is (batch size, channel, height, width).

        Returns
        -------
        torch.Tensor
            The encoder output. Its shape is the same dimensionality as the input.
        """

        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class ViT(nn.Module):
    """Vision Transformer model.
    """

    def __init__(
            self,
            chw=(1, 28, 28),
            num_patches: int = 7,
            num_blocks: int = 2,
            hidden_dim: int = 8,  # try 64, 128 or 256
            num_heads: int = 2,
            out_dim: int = 10
        ):
        """Init the model.

        Parameters
        ----------
        chw : tuple, optional
            The input data format (channel, height, width). By default (1, 28, 28).
        num_patches : int, optional
            Along the H and W dimension the image is splitted into this number of
            patches (sub-images).
            In case of a 28x28 image, it is splitted into:
                - 49 patches (each one is 4x4), if num_patches=7.
                - 16 patches (each one is 7x7), if num_patches=4.
            By default it is 7.
        num_blocks : int, optional
            Number of Transformer blocks. By default it is 2.
        hidden_dim: int, optional
            The number of hidden dimensions. It reduces the patch size to this size.
            By default it is 8.
        num_heads: int, optional
            Number of heads. By default it is 2.
        out_dim: int, optional
            Output dimension. By default it is 10.
        """

        # Base constructor.
        super(ViT, self).__init__()

        # Attributes.
        self.chw = chw # (Channel, Height, Width)
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim

        if chw[1] % num_patches != 0 or chw[2] % num_patches != 0:
            raise ValueError(
                "The Input shape shall be divisible by the number of patches."
            )

        if not hidden_dim % num_heads == 0:
            raise ValueError(
                f"Cannot divide dimension {hidden_dim} with {num_heads} heads."
            )

        self.patch_size = (chw[1] // num_patches, chw[2] // num_patches)

        # 1. Linear projection of patch embeddings.
        # Map each 16-dimensional patch to an 8-dimensional patch.
        # The `self.hidden_dim` is the “width” of the internal feature space,
        # the number of channels in the embedding that the Transformer operates on.
        self.input_d = chw[0] * self.patch_size[0] * self.patch_size[1]
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_dim)
        
        # 2. Positional embedding.
        # Added to the embeddings so that the model knows the order of patches.
        # Add it to the module, but it will not be in the state dict.
        self.register_buffer('positional_embeddings', get_positional_encodings(
            self.num_patches ** 2 + 1, self.hidden_dim), persistent=False
        )

        # 3. Learnable classifiation token.
        # A special token is responsible for capturing information about the other
        # tokens. This is done after the MultiHeadedSelfAttention block. When the
        # information about all other tokens are present here, we will be able to
        # classify the image using only this special token. It is a learned parameter.
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        # 4. Transformer encoder blocks.
        self.blocks = nn.ModuleList(
            [VisionTransformerBlock(hidden_dim, num_heads) for _ in range(num_blocks)]
        )

        # 5. Classification.
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, out_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        images : torch.Tensor
            The input images. Its shape is (batch size, channel, height, width).

        Returns
        -------
        torch.Tensor
            The output logits. Its shape is (batch size, num_classes).
        """
        
        n, _, _, _ = images.shape  # batch, channel, height, width

        # Make patches from the input and map them to the given size.
        patches = make_patches(images, self.num_patches).to(get_device())

        # Tokenization:
        # Map the vector corresponding to each patch to the hidden size dimension.
        # The size of the hidden dimnsions determine how much information each patch
        # can encode.
        tokens = self.linear_mapper(patches)

        # Add classification token to the first position of each token sequence.
        # It creates a batch-sized class token (n, 1, hidden_dim) and concatenate.
        class_tokens = self.class_token.unsqueeze(0).expand(n, -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)

        # Add positional embedding: The tokens have size (N, 50, 8), so we have to
        # repeat the (50, 8) positional encoding matrix N times.
        pos_embed = self.positional_embeddings.repeat(n, 1, 1)
        out = tokens + pos_embed

        # Transformer encoder blocks.
        for block in self.blocks:
            out = block(out)

        # Get the classification token (first token).
        out = out[:, 0]

        # Map to output dimension, output category distribution.
        out = self.mlp(out)

        return out
