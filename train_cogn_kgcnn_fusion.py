"""
Training script for kgcnn-based coGN with text middle fusion.

This script provides:
1. Data loading and preprocessing for crystal graphs
2. Text tokenization for MatSciBERT
3. TensorFlow dataset creation with ragged tensors
4. Training loop with validation and metrics tracking
"""

import os
import sys
import json
import argparse
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from functools import partial
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# JARVIS imports for data loading
from jarvis.core.atoms import Atoms as JarvisAtoms
from jarvis.db.figshare import data as jdata

# Pymatgen for structure conversion
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element

# kgcnn imports
from kgcnn.crystal import graph_builder
from kgcnn.crystal.preprocessor import KNNUnitCell

# Transformers for text encoding
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed")

# Import the model
from models.cogn_text_fusion_tf import make_cogn_text_fusion_model


def jarvis_to_pymatgen(atoms: JarvisAtoms) -> Structure:
    """Convert jarvis Atoms to pymatgen Structure."""
    lattice = Lattice(atoms.lattice.matrix)
    species = [Element(el) for el in atoms.elements]
    coords = atoms.frac_coords
    return Structure(lattice, species, coords, coords_are_cartesian=False)


def networkx_to_kgcnn_dict(nx_graph, ensure_bidirectional: bool = True) -> Dict:
    """
    Convert NetworkX graph to kgcnn-compatible dictionary format.

    Args:
        nx_graph: NetworkX MultiDiGraph from kgcnn graph builder
        ensure_bidirectional: Whether to ensure all edges have reverse edges

    Returns:
        Dictionary with graph data
    """
    num_nodes = nx_graph.number_of_nodes()

    # Extract node data
    atomic_numbers = []
    frac_coords = []

    for node_idx in range(num_nodes):
        node_data = nx_graph.nodes[node_idx]
        atomic_numbers.append(node_data.get('atomic_number', 1))
        frac_coords.append(node_data.get('frac_coords', [0, 0, 0]))

    # Extract edge data with bidirectional edges
    src_nodes = []
    dst_nodes = []
    offsets = []
    cell_translations = []

    existing_edges = set()

    for edge in nx_graph.edges(data=True):
        src, dst, edge_data = edge

        offset = edge_data.get('offset', np.zeros(3))
        cell_translation = edge_data.get('cell_translation', np.zeros(3))

        edge_key = (src, dst, tuple(np.round(cell_translation).astype(int)))

        if edge_key not in existing_edges:
            src_nodes.append(src)
            dst_nodes.append(dst)
            offsets.append(offset)
            cell_translations.append(cell_translation)
            existing_edges.add(edge_key)

        # Add reverse edge
        if ensure_bidirectional:
            reverse_cell_translation = -np.array(cell_translation)
            reverse_edge_key = (dst, src, tuple(np.round(reverse_cell_translation).astype(int)))

            if reverse_edge_key not in existing_edges:
                src_nodes.append(dst)
                dst_nodes.append(src)
                offsets.append(-np.array(offset))
                cell_translations.append(reverse_cell_translation)
                existing_edges.add(reverse_edge_key)

    # Create edge indices
    if len(src_nodes) > 0:
        edge_indices = np.stack([src_nodes, dst_nodes], axis=-1).astype(np.int32)
        offsets = np.array(offsets, dtype=np.float32)
    else:
        edge_indices = np.zeros((0, 2), dtype=np.int32)
        offsets = np.zeros((0, 3), dtype=np.float32)

    return {
        'atomic_number': np.array(atomic_numbers, dtype=np.int32),
        'offset': offsets,
        'edge_indices': edge_indices,
        'frac_coords': np.array(frac_coords, dtype=np.float32),
        'lattice_matrix': getattr(nx_graph, 'lattice_matrix', np.eye(3)).astype(np.float32),
    }


def build_crystal_graph(atoms: JarvisAtoms, k: int = 12, cutoff: float = 8.0) -> Dict:
    """
    Build crystal graph from JARVIS atoms using kgcnn.

    Args:
        atoms: JARVIS Atoms object
        k: Number of nearest neighbors
        cutoff: Cutoff distance for neighbor search

    Returns:
        Dictionary with graph data
    """
    # Convert to pymatgen
    structure = jarvis_to_pymatgen(atoms)

    # Build graph using kgcnn
    nx_graph = graph_builder.structure_to_empty_graph(structure, symmetrize=False)
    nx_graph = graph_builder.add_knn_bonds(nx_graph, k=k, max_radius=cutoff, inplace=True)
    nx_graph = graph_builder.add_edge_information(nx_graph, inplace=True)

    # Convert to dictionary format
    return networkx_to_kgcnn_dict(nx_graph, ensure_bidirectional=True)


def load_dataset(
    dataset_name: str = "dft_3d",
    target: str = "formation_energy_peratom",
    max_samples: Optional[int] = None,
    text_column: str = "text",
) -> Tuple[List[Dict], List[float], List[str], List[str]]:
    """
    Load crystal dataset from JARVIS.

    Args:
        dataset_name: Name of JARVIS dataset
        target: Target property to predict
        max_samples: Maximum number of samples to load
        text_column: Column name for text descriptions

    Returns:
        graphs: List of graph dictionaries
        labels: List of target values
        texts: List of text descriptions
        ids: List of sample IDs
    """
    print(f"Loading {dataset_name} dataset...")

    # Load JARVIS data
    data = jdata(dataset_name)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Filter samples with target property
    df = df[df[target].notna()]
    df = df[df[target] != "na"]

    # Limit samples
    if max_samples is not None:
        df = df.head(max_samples)

    print(f"Processing {len(df)} samples...")

    graphs = []
    labels = []
    texts = []
    ids = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
        try:
            # Get atoms
            atoms = JarvisAtoms.from_dict(row['atoms'])

            # Build graph
            graph = build_crystal_graph(atoms, k=12, cutoff=8.0)

            # Get label
            label = float(row[target])

            # Get text (generate if not available)
            if text_column in df.columns and pd.notna(row.get(text_column)):
                text = str(row[text_column])
            else:
                # Generate basic description from composition
                text = f"Crystal structure with composition {atoms.composition.reduced_formula}"

            # Get ID
            jid = row.get('jid', f'sample_{idx}')

            graphs.append(graph)
            labels.append(label)
            texts.append(text)
            ids.append(jid)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

    print(f"Successfully processed {len(graphs)} samples")
    return graphs, labels, texts, ids


def create_tf_dataset(
    graphs: List[Dict],
    labels: List[float],
    texts: List[str],
    tokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    max_text_length: int = 256,
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset with ragged tensors.

    Args:
        graphs: List of graph dictionaries
        labels: List of target values
        texts: List of text descriptions
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        shuffle: Whether to shuffle data
        max_text_length: Maximum text sequence length

    Returns:
        TensorFlow Dataset
    """
    # Tokenize texts
    if tokenizer is not None:
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_text_length,
            return_tensors='np'
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
    else:
        # Dummy tokenization
        input_ids = np.zeros((len(texts), max_text_length), dtype=np.int32)
        attention_mask = np.ones((len(texts), max_text_length), dtype=np.int32)

    # Create ragged tensors for variable-size graph data
    def generator():
        indices = list(range(len(graphs)))
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            graph = graphs[idx]
            yield {
                'atomic_number': graph['atomic_number'],
                'offset': graph['offset'],
                'edge_indices': graph['edge_indices'],
                'input_ids': input_ids[idx],
                'attention_mask': attention_mask[idx],
            }, np.array([labels[idx]], dtype=np.float32)

    # Define output signature with ragged tensors
    output_signature = (
        {
            'atomic_number': tf.RaggedTensorSpec(shape=[None], dtype=tf.int32, ragged_rank=0),
            'offset': tf.RaggedTensorSpec(shape=[None, 3], dtype=tf.float32, ragged_rank=0),
            'edge_indices': tf.RaggedTensorSpec(shape=[None, 2], dtype=tf.int32, ragged_rank=0),
            'input_ids': tf.TensorSpec(shape=[max_text_length], dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=[max_text_length], dtype=tf.int32),
        },
        tf.TensorSpec(shape=[1], dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    # Batch with ragged tensors
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def collate_batch(batch):
    """
    Collate function to handle ragged batch data.

    Converts the batch dictionary to format expected by model.
    """
    inputs, labels = batch

    # Stack ragged tensors properly
    return {
        'atomic_number': inputs['atomic_number'],
        'offset': inputs['offset'],
        'edge_indices': inputs['edge_indices'],
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
    }, labels


class MAEMetric(keras.metrics.Metric):
    """Mean Absolute Error metric."""

    def __init__(self, name='mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_true - y_pred)
        self.total.assign_add(tf.reduce_sum(error))
        self.count.assign_add(tf.cast(tf.size(error), tf.float32))

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.)
        self.count.assign(0.)


def train_model(
    model,
    train_dataset,
    val_dataset,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs',
):
    """
    Train the model.

    Args:
        model: Keras model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        learning_rate: Learning rate
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for logs
    """
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Optimizer and loss
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.MeanSquaredError()

    # Metrics
    train_loss = keras.metrics.Mean(name='train_loss')
    train_mae = MAEMetric(name='train_mae')
    val_loss = keras.metrics.Mean(name='val_loss')
    val_mae = MAEMetric(name='val_mae')

    # Callbacks
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'model_{epoch:03d}.weights.h5'),
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

    # Training history
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}

    best_val_mae = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Reset metrics
        train_loss.reset_state()
        train_mae.reset_state()
        val_loss.reset_state()
        val_mae.reset_state()

        # Training loop
        train_pbar = tqdm(train_dataset, desc='Training')
        for batch_inputs, batch_labels in train_pbar:
            with tf.GradientTape() as tape:
                predictions = model(batch_inputs, training=True)
                loss = loss_fn(batch_labels, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss.update_state(loss)
            train_mae.update_state(batch_labels, predictions)

            train_pbar.set_postfix({
                'loss': f'{train_loss.result():.4f}',
                'mae': f'{train_mae.result():.4f}'
            })

        # Validation loop
        val_pbar = tqdm(val_dataset, desc='Validation')
        for batch_inputs, batch_labels in val_pbar:
            predictions = model(batch_inputs, training=False)
            loss = loss_fn(batch_labels, predictions)

            val_loss.update_state(loss)
            val_mae.update_state(batch_labels, predictions)

            val_pbar.set_postfix({
                'loss': f'{val_loss.result():.4f}',
                'mae': f'{val_mae.result():.4f}'
            })

        # Record history
        history['train_loss'].append(float(train_loss.result()))
        history['train_mae'].append(float(train_mae.result()))
        history['val_loss'].append(float(val_loss.result()))
        history['val_mae'].append(float(val_mae.result()))

        print(f"Train Loss: {train_loss.result():.4f}, Train MAE: {train_mae.result():.4f}")
        print(f"Val Loss: {val_loss.result():.4f}, Val MAE: {val_mae.result():.4f}")

        # Save best model
        current_val_mae = float(val_mae.result())
        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            model.save_weights(os.path.join(checkpoint_dir, 'best_model.weights.h5'))
            print(f"New best model saved with Val MAE: {best_val_mae:.4f}")

    # Save final history
    with open(os.path.join(log_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return history


def main():
    parser = argparse.ArgumentParser(description='Train coGN with text fusion (kgcnn-based)')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='dft_3d', help='JARVIS dataset name')
    parser.add_argument('--target', type=str, default='formation_energy_peratom', help='Target property')
    parser.add_argument('--max_samples', type=int, default=10000, help='Maximum number of samples')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')

    # Model arguments
    parser.add_argument('--node_dim', type=int, default=128, help='Node feature dimension')
    parser.add_argument('--edge_dim', type=int, default=128, help='Edge feature dimension')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of GN layers')
    parser.add_argument('--fusion_layers', type=str, default='2,3', help='Comma-separated fusion layer indices')
    parser.add_argument('--fusion_heads', type=int, default=4, help='Number of attention heads for fusion')
    parser.add_argument('--text_model', type=str, default='m3rg-iitd/matscibert', help='Text encoder model')
    parser.add_argument('--freeze_text', action='store_true', default=True, help='Freeze text encoder')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_kgcnn', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs_kgcnn', help='Log directory')

    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')

    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Set seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Parse fusion layers
    fusion_layers = [int(x) for x in args.fusion_layers.split(',')]

    print("=" * 60)
    print("coGN with Text Fusion Training (kgcnn-based)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Target: {args.target}")
    print(f"Max samples: {args.max_samples}")
    print(f"Model: node_dim={args.node_dim}, num_layers={args.num_layers}")
    print(f"Fusion layers: {fusion_layers}")
    print(f"Text model: {args.text_model}")
    print("=" * 60)

    # Load data
    graphs, labels, texts, ids = load_dataset(
        dataset_name=args.dataset,
        target=args.target,
        max_samples=args.max_samples
    )

    # Split data
    n_samples = len(graphs)
    n_val = int(n_samples * args.val_split)
    indices = np.random.permutation(n_samples)

    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_graphs = [graphs[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_texts = [texts[i] for i in train_indices]

    val_graphs = [graphs[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    val_texts = [texts[i] for i in val_indices]

    print(f"Train samples: {len(train_graphs)}")
    print(f"Val samples: {len(val_graphs)}")

    # Load tokenizer
    tokenizer = None
    if HAS_TRANSFORMERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.text_model)
            print(f"Loaded tokenizer: {args.text_model}")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")

    # Create datasets
    train_dataset = create_tf_dataset(
        train_graphs, train_labels, train_texts, tokenizer,
        batch_size=args.batch_size, shuffle=True
    )

    val_dataset = create_tf_dataset(
        val_graphs, val_labels, val_texts, tokenizer,
        batch_size=args.batch_size, shuffle=False
    )

    # Create model
    model = make_cogn_text_fusion_model(
        node_dim=args.node_dim,
        edge_dim=args.edge_dim,
        num_layers=args.num_layers,
        text_model=args.text_model,
        freeze_text_encoder=args.freeze_text,
        fusion_layers=fusion_layers,
        fusion_heads=args.fusion_heads,
    )

    print("\nModel created successfully!")

    # Train
    history = train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    print("\nTraining completed!")
    print(f"Best Val MAE: {min(history['val_mae']):.4f}")


if __name__ == '__main__':
    main()
