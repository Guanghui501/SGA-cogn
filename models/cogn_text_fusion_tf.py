"""
coGN with Text Middle Fusion using TensorFlow/Keras (kgcnn-based)

This module implements a multimodal crystal property prediction model that combines:
1. kgcnn's coGN (Connectivity-optimized Graph Network) for crystal structure encoding
2. MatSciBERT for text description encoding
3. Middle fusion to inject text features into graph processing layers
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# kgcnn imports
from kgcnn.layers.geom import EuclideanNorm
from kgcnn.literature.coGN._graph_network.graph_networks import (
    GraphNetwork, SequentialGraphNetwork, CrystalInputBlock
)
from kgcnn.literature.coGN._embedding_layers._atom_embedding import AtomEmbedding
from kgcnn.literature.coGN._embedding_layers._edge_embedding import EdgeEmbedding
from kgcnn.crystal.periodic_table.periodic_table import PeriodicTable
from kgcnn.layers.mlp import MLP
from kgcnn.layers.modules import LazyConcatenate, LazyAdd
from kgcnn.layers.pooling import PoolingNodes

# For text encoding
try:
    from transformers import TFAutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Text fusion will not work.")


class CrossModalAttention(layers.Layer):
    """Cross-modal attention layer for fusing text and graph features."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout

        # Multi-head attention
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout
        )

        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(hidden_dim * 4, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(hidden_dim),
            layers.Dropout(dropout)
        ])

        # Layer normalization
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, graph_features, text_features, training=False):
        """
        Args:
            graph_features: [batch, num_nodes, hidden_dim] or RaggedTensor
            text_features: [batch, hidden_dim]

        Returns:
            Fused features with same shape as graph_features
        """
        # Expand text features for attention
        # text_features: [batch, hidden_dim] -> [batch, 1, hidden_dim]
        text_expanded = tf.expand_dims(text_features, axis=1)

        # Handle ragged tensors by converting to dense for attention
        if isinstance(graph_features, tf.RaggedTensor):
            # Get row splits for reconstruction
            row_splits = graph_features.row_splits

            # Convert to dense with padding
            graph_dense = graph_features.to_tensor()
            batch_size = tf.shape(graph_dense)[0]

            # Expand text to match batch size
            text_expanded = tf.broadcast_to(text_expanded, [batch_size, 1, self.hidden_dim])

            # Cross-attention: graph queries, text keys/values
            attn_output = self.mha(
                query=graph_dense,
                key=text_expanded,
                value=text_expanded,
                training=training
            )

            # Residual connection and layer norm
            x = self.ln1(graph_dense + attn_output)

            # Feed-forward
            ffn_output = self.ffn(x, training=training)
            x = self.ln2(x + ffn_output)

            # Convert back to ragged
            return tf.RaggedTensor.from_row_splits(
                values=tf.reshape(x, [-1, self.hidden_dim])[:tf.reduce_sum(graph_features.row_lengths())],
                row_splits=row_splits
            )
        else:
            # Dense tensor path
            text_expanded = tf.broadcast_to(text_expanded, tf.shape(graph_features)[:1] + [1, self.hidden_dim])

            attn_output = self.mha(
                query=graph_features,
                key=text_expanded,
                value=text_expanded,
                training=training
            )

            x = self.ln1(graph_features + attn_output)
            ffn_output = self.ffn(x, training=training)
            return self.ln2(x + ffn_output)


class TextEncoder(layers.Layer):
    """Text encoder using pre-trained MatSciBERT."""

    def __init__(self, model_name: str = "m3rg-iitd/matscibert",
                 output_dim: int = 128,
                 freeze_bert: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.output_dim = output_dim
        self.freeze_bert = freeze_bert

        if HAS_TRANSFORMERS:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = TFAutoModel.from_pretrained(model_name, from_pt=True)

            if freeze_bert:
                self.bert.trainable = False

            # Projection layer
            self.projection = layers.Dense(output_dim, activation='gelu')
        else:
            self.projection = layers.Dense(output_dim)

    def tokenize(self, texts: List[str], max_length: int = 512):
        """Tokenize text inputs."""
        if not HAS_TRANSFORMERS:
            return None

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )
        return encoded

    def call(self, input_ids, attention_mask=None, training=False):
        """
        Args:
            input_ids: Tokenized input IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Text embeddings [batch, output_dim]
        """
        if HAS_TRANSFORMERS and hasattr(self, 'bert'):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                training=training
            )
            # Use [CLS] token representation
            cls_output = outputs.last_hidden_state[:, 0, :]
            return self.projection(cls_output)
        else:
            # Fallback: simple embedding
            return self.projection(tf.zeros([tf.shape(input_ids)[0], 768]))


class GraphNetworkWithFusion(GraphNetwork):
    """Graph Network layer with optional text fusion."""

    def __init__(self, edge_mlp, node_mlp, global_mlp,
                 fusion_layer: Optional[CrossModalAttention] = None,
                 **kwargs):
        super().__init__(edge_mlp, node_mlp, global_mlp, **kwargs)
        self.fusion_layer = fusion_layer

    def call(self, inputs, text_features=None, training=False):
        """Forward pass with optional text fusion."""
        edges, nodes, globals_, edge_indices = inputs

        # Standard GN update
        edges_new, nodes_new, globals_new, edge_indices_new = super().call(inputs)

        # Apply text fusion to node features if fusion layer exists
        if self.fusion_layer is not None and text_features is not None:
            node_features = self.get_features(nodes_new)
            fused_features = self.fusion_layer(node_features, text_features, training=training)
            nodes_new = self.update_features(nodes_new, fused_features)

        return edges_new, nodes_new, globals_new, edge_indices_new


class SequentialGraphNetworkWithFusion(layers.Layer):
    """Sequential GN blocks with text fusion at specified layers."""

    def __init__(self, graph_network_blocks: List,
                 fusion_layers: Optional[Dict[int, CrossModalAttention]] = None,
                 **kwargs):
        """
        Args:
            graph_network_blocks: List of GraphNetwork blocks
            fusion_layers: Dict mapping block index to fusion layer
        """
        super().__init__(**kwargs)
        self.graph_network_blocks = graph_network_blocks
        self.fusion_layers = fusion_layers or {}

    def call(self, inputs, text_features=None, training=False):
        edges, nodes, globals_, edge_indices = inputs

        for i, block in enumerate(self.graph_network_blocks):
            out = block([edges, nodes, globals_, edge_indices])
            edges_new, nodes_new, globals_new, _ = out

            # Apply text fusion at specified layers
            if i in self.fusion_layers and text_features is not None:
                fusion_layer = self.fusion_layers[i]
                node_features = block.get_features(nodes_new)
                fused_features = fusion_layer(node_features, text_features, training=training)
                nodes_new = block.update_features(nodes_new, fused_features)

            edges = edges_new
            nodes = nodes_new
            globals_ = globals_new

        return edges, nodes, globals_, edge_indices


class coGNTextFusion(keras.Model):
    """
    coGN model with text middle fusion for multimodal crystal property prediction.

    Architecture:
    1. Crystal graph encoding using kgcnn coGN layers
    2. Text encoding using MatSciBERT
    3. Middle fusion: text features are injected into node features at specified layers
    4. Final prediction from pooled graph features
    """

    def __init__(self,
                 # Graph parameters
                 node_dim: int = 128,
                 edge_dim: int = 128,
                 num_layers: int = 5,
                 # Text parameters
                 text_model: str = "m3rg-iitd/matscibert",
                 freeze_text_encoder: bool = True,
                 # Fusion parameters
                 fusion_layers: List[int] = [2, 3],  # Which layers to apply fusion
                 fusion_heads: int = 4,
                 fusion_dropout: float = 0.1,
                 # Output parameters
                 output_dim: int = 1,
                 # Other
                 **kwargs):
        super().__init__(**kwargs)

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.fusion_layer_indices = fusion_layers

        # Build input block
        periodic_table = PeriodicTable()
        self.atom_embedding = AtomEmbedding(
            atomic_number_embedding_args={'input_dim': 119, 'output_dim': node_dim},
            atomic_mass=periodic_table.get_atomic_mass(),
            atomic_radius=periodic_table.get_atomic_radius(),
            electronegativity=periodic_table.get_electronegativity(),
            ionization_energy=periodic_table.get_ionization_energy(),
            oxidation_states=periodic_table.get_oxidation_states()
        )
        self.edge_embedding = EdgeEmbedding(
            bins_distance=32,
            max_distance=8.0,
            distance_log_base=1.0
        )
        self.input_block = CrystalInputBlock(
            self.atom_embedding,
            self.edge_embedding,
            atom_mlp=MLP([node_dim]),
            edge_mlp=MLP([edge_dim])
        )

        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_model,
            output_dim=node_dim,
            freeze_bert=freeze_text_encoder
        )

        # Build fusion layers
        fusion_layer_dict = {}
        for idx in fusion_layers:
            if 0 <= idx < num_layers:
                fusion_layer_dict[idx] = CrossModalAttention(
                    hidden_dim=node_dim,
                    num_heads=fusion_heads,
                    dropout=fusion_dropout
                )

        # Build processing blocks
        processing_blocks = []
        for i in range(num_layers):
            block = GraphNetwork(
                edge_mlp=MLP([node_dim] * 5, activation=['swish'] * 5),
                node_mlp=MLP([node_dim], activation=['swish']),
                global_mlp=None,
                aggregate_edges_local='sum',
                residual_node_update=True,
                update_edges_input=[True, True, True, False],
                update_nodes_input=[True, False, False],
            )
            processing_blocks.append(block)

        self.sequential_gn = SequentialGraphNetworkWithFusion(
            processing_blocks,
            fusion_layers=fusion_layer_dict
        )

        # Output block
        self.output_block = GraphNetwork(
            edge_mlp=None,
            node_mlp=None,
            global_mlp=MLP([node_dim, output_dim], activation=['swish', 'linear']),
            aggregate_edges_local='sum',
            aggregate_nodes='mean',
            return_updated_globals=True,
        )

        # Distance computation
        self.euclidean_norm = EuclideanNorm()

    def call(self, inputs, training=False):
        """
        Forward pass.

        Args:
            inputs: Dict containing:
                - offset: Edge displacement vectors [batch, num_edges, 3] (ragged)
                - atomic_number: Atomic numbers [batch, num_atoms] (ragged)
                - edge_indices: Edge connectivity [batch, num_edges, 2] (ragged)
                - input_ids: Tokenized text [batch, seq_len]
                - attention_mask: Text attention mask [batch, seq_len]

        Returns:
            Predictions [batch, output_dim]
        """
        # Extract inputs
        offset = inputs['offset']
        atomic_number = inputs['atomic_number']
        edge_indices = inputs['edge_indices']

        # Compute distances
        distance = self.euclidean_norm(offset)

        # Encode text
        text_features = None
        if 'input_ids' in inputs and inputs['input_ids'] is not None:
            text_features = self.text_encoder(
                inputs['input_ids'],
                inputs.get('attention_mask'),
                training=training
            )

        # Input embedding
        edge_features, node_features, _, _ = self.input_block([
            distance,
            atomic_number,
            None,
            edge_indices
        ])

        # Process through GN layers with fusion
        edges, nodes, globals_, _ = self.sequential_gn(
            [edge_features, node_features, None, edge_indices],
            text_features=text_features,
            training=training
        )

        # Output
        _, _, out, _ = self.output_block([edges, nodes, globals_, edge_indices])
        output = self.output_block.get_features(out)

        return output

    def get_config(self):
        return {
            'node_dim': self.node_dim,
            'edge_dim': self.edge_dim,
            'num_layers': self.num_layers,
            'fusion_layer_indices': self.fusion_layer_indices,
        }


def make_cogn_text_fusion_model(
    node_dim: int = 128,
    edge_dim: int = 128,
    num_layers: int = 5,
    text_model: str = "m3rg-iitd/matscibert",
    freeze_text_encoder: bool = True,
    fusion_layers: List[int] = [2, 3],
    fusion_heads: int = 4,
    fusion_dropout: float = 0.1,
    output_dim: int = 1,
):
    """
    Factory function to create coGN with text fusion model.

    Args:
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
        num_layers: Number of graph network layers
        text_model: Name of pretrained text model
        freeze_text_encoder: Whether to freeze text encoder weights
        fusion_layers: List of layer indices where to apply text fusion
        fusion_heads: Number of attention heads for fusion
        fusion_dropout: Dropout rate for fusion layers
        output_dim: Output dimension for prediction

    Returns:
        coGNTextFusion model instance
    """
    return coGNTextFusion(
        node_dim=node_dim,
        edge_dim=edge_dim,
        num_layers=num_layers,
        text_model=text_model,
        freeze_text_encoder=freeze_text_encoder,
        fusion_layers=fusion_layers,
        fusion_heads=fusion_heads,
        fusion_dropout=fusion_dropout,
        output_dim=output_dim,
    )


if __name__ == "__main__":
    # Test the model
    print("Testing coGN with Text Fusion...")

    # Create model
    model = make_cogn_text_fusion_model(
        node_dim=64,
        edge_dim=64,
        num_layers=3,
        fusion_layers=[1, 2],
    )

    # Create dummy inputs
    batch_size = 2
    num_atoms = [5, 7]
    num_edges = [12, 18]

    # Create ragged tensors
    offset_values = np.random.randn(sum(num_edges), 3).astype(np.float32)
    offset = tf.RaggedTensor.from_row_lengths(offset_values, num_edges)

    atomic_number_values = np.random.randint(1, 100, sum(num_atoms)).astype(np.int32)
    atomic_number = tf.RaggedTensor.from_row_lengths(atomic_number_values, num_atoms)

    edge_indices_values = np.random.randint(0, 5, (sum(num_edges), 2)).astype(np.int32)
    edge_indices = tf.RaggedTensor.from_row_lengths(edge_indices_values, num_edges)

    # Create dummy text inputs
    input_ids = tf.constant([[101, 2054, 2003, 1037, 7953, 102, 0, 0],
                              [101, 2023, 2003, 1037, 3231, 102, 0, 0]], dtype=tf.int32)
    attention_mask = tf.constant([[1, 1, 1, 1, 1, 1, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 0, 0]], dtype=tf.int32)

    inputs = {
        'offset': offset,
        'atomic_number': atomic_number,
        'edge_indices': edge_indices,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    # Forward pass
    try:
        output = model(inputs, training=False)
        print(f"Output shape: {output.shape}")
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
