"""coGN graph construction using kgcnn library.

This module provides graph construction for coGN models using the kgcnn library's
graph building utilities. It supports multiple edge construction strategies:
- k-NN (k-nearest neighbors)
- Radius-based cutoff
- Voronoi tessellation

The graphs are built using pymatgen/kgcnn and converted to DGL format for training.
"""

from collections import defaultdict
from typing import Tuple, Optional, Literal, Union
import warnings

import dgl
import numpy as np
import torch
from jarvis.core.atoms import Atoms as JarvisAtoms
from jarvis.core.specie import get_node_attributes

# kgcnn imports
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from kgcnn.crystal import graph_builder
from networkx import MultiDiGraph


# =============================================================================
# Structure Conversion Utilities
# =============================================================================

def jarvis_to_pymatgen(atoms: JarvisAtoms) -> Structure:
    """Convert jarvis Atoms to pymatgen Structure.

    Args:
        atoms: jarvis Atoms object

    Returns:
        pymatgen Structure object
    """
    lattice = Lattice(atoms.lattice.matrix)
    species = [Element(el) for el in atoms.elements]
    coords = atoms.frac_coords

    return Structure(lattice, species, coords, coords_are_cartesian=False)


def pymatgen_to_jarvis(structure: Structure) -> JarvisAtoms:
    """Convert pymatgen Structure to jarvis Atoms.

    Args:
        structure: pymatgen Structure object

    Returns:
        jarvis Atoms object
    """
    from jarvis.core.lattice import Lattice as JarvisLattice

    lattice = JarvisLattice(structure.lattice.matrix)
    elements = [str(site.specie) for site in structure.sites]
    coords = structure.frac_coords

    return JarvisAtoms(
        lattice_mat=lattice.matrix,
        elements=elements,
        coords=coords,
        cartesian=False
    )


# =============================================================================
# NetworkX to DGL Conversion
# =============================================================================

def networkx_to_dgl(
    nx_graph: MultiDiGraph,
    atom_features: str = "atomic_number",
    include_line_graph: bool = False,
) -> Union[dgl.DGLGraph, Tuple[dgl.DGLGraph, dgl.DGLGraph]]:
    """Convert a kgcnn NetworkX graph to DGL format.

    Args:
        nx_graph: NetworkX MultiDiGraph from kgcnn graph_builder
        atom_features: Type of atom features to use
        include_line_graph: Whether to also return the line graph

    Returns:
        DGL graph (and optionally line graph)
    """
    num_nodes = nx_graph.number_of_nodes()

    if num_nodes == 0:
        warnings.warn("Empty graph with no nodes!")
        g = dgl.graph(([], []))
        if include_line_graph:
            return g, dgl.graph(([], []))
        return g

    # Extract node data
    atomic_numbers = []
    frac_coords = []
    cart_coords = []
    multiplicities = []

    for node_idx in range(num_nodes):
        node_data = nx_graph.nodes[node_idx]
        atomic_numbers.append(node_data.get('atomic_number', 1))
        frac_coords.append(node_data.get('frac_coords', [0, 0, 0]))
        cart_coords.append(node_data.get('coords', [0, 0, 0]))
        multiplicities.append(node_data.get('multiplicity', 1))

    # Extract edge data
    src_nodes = []
    dst_nodes = []
    offsets = []
    distances = []
    cell_translations = []

    for edge in nx_graph.edges(data=True):
        src, dst, edge_data = edge
        src_nodes.append(src)
        dst_nodes.append(dst)

        # Get edge attributes
        offset = edge_data.get('offset', np.zeros(3))
        distance = edge_data.get('distance', 0.0)
        cell_translation = edge_data.get('cell_translation', np.zeros(3))

        offsets.append(offset)
        distances.append(distance)
        cell_translations.append(cell_translation)

    # Create DGL graph
    if len(src_nodes) > 0:
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    else:
        g = dgl.graph(([], []), num_nodes=num_nodes)

    # Set node features
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
    frac_coords = torch.tensor(np.array(frac_coords), dtype=torch.get_default_dtype())
    cart_coords = torch.tensor(np.array(cart_coords), dtype=torch.get_default_dtype())
    multiplicities = torch.tensor(multiplicities, dtype=torch.long)

    # Generate atom features based on atom_features type
    if atom_features == "atomic_number":
        atom_feat = atomic_numbers.unsqueeze(-1).float()
    else:
        # Use jarvis get_node_attributes for other feature types
        atom_feat = []
        for z in atomic_numbers.tolist():
            try:
                # Map atomic number to element symbol
                el = Element.from_Z(z).symbol
                feat = list(get_node_attributes(el, atom_features=atom_features))
            except:
                feat = [float(z)]
            atom_feat.append(feat)
        atom_feat = torch.tensor(np.array(atom_feat), dtype=torch.get_default_dtype())

    g.ndata["atom_features"] = atom_feat
    g.ndata["atomic_number"] = atomic_numbers
    g.ndata["frac_coords"] = frac_coords
    g.ndata["coords"] = cart_coords
    g.ndata["multiplicity"] = multiplicities

    # Set edge features
    if len(src_nodes) > 0:
        offsets = torch.tensor(np.array(offsets), dtype=torch.get_default_dtype())
        distances = torch.tensor(distances, dtype=torch.get_default_dtype())
        cell_translations = torch.tensor(np.array(cell_translations), dtype=torch.int64)

        g.edata["offset"] = offsets
        g.edata["r"] = offsets  # Alias for compatibility
        g.edata["distance"] = distances
        g.edata["cell_translation"] = cell_translations
        g.edata["edge_indices"] = torch.stack(
            (torch.tensor(src_nodes), torch.tensor(dst_nodes)), dim=1
        )

    # Set graph-level attributes
    if hasattr(nx_graph, 'lattice_matrix'):
        g.lattice_matrix = torch.tensor(
            nx_graph.lattice_matrix, dtype=torch.get_default_dtype()
        )
    if hasattr(nx_graph, 'spacegroup'):
        g.spacegroup = nx_graph.spacegroup

    if include_line_graph and g.num_edges() > 0:
        lg = g.line_graph(shared=True)
        lg.apply_edges(_compute_bond_cosines)
        return g, lg

    return g


def _compute_bond_cosines(edges):
    """Compute bond angle cosines for line graphs."""
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]

    norm1 = torch.norm(r1, dim=1, keepdim=True)
    norm2 = torch.norm(r2, dim=1, keepdim=True)

    # Avoid division by zero
    norm1 = torch.clamp(norm1, min=1e-8)
    norm2 = torch.clamp(norm2, min=1e-8)

    bond_cosine = torch.sum(r1 * r2, dim=1) / (norm1.squeeze() * norm2.squeeze())
    bond_cosine = torch.clamp(bond_cosine, -1, 1)

    return {"h": bond_cosine}


# =============================================================================
# Main Graph Building Functions (using kgcnn)
# =============================================================================

def build_kgcnn_graph(
    atoms: JarvisAtoms,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    atom_features: str = "atomic_number",
    edge_strategy: Literal["knn", "radius", "voronoi"] = "knn",
    compute_line_graph: bool = False,
    use_symmetry: bool = False,
    tolerance: Optional[float] = 1e-9,
    min_ridge_area: Optional[float] = 0.0,
) -> Union[dgl.DGLGraph, Tuple[dgl.DGLGraph, dgl.DGLGraph]]:
    """Build a DGL graph using kgcnn's graph construction logic.

    This function uses pymatgen and kgcnn's graph_builder module to construct
    crystal graphs with proper periodic boundary conditions.

    Args:
        atoms: jarvis Atoms object
        cutoff: Distance cutoff for radius-based edges (Angstrom)
        max_neighbors: Maximum number of neighbors for k-NN
        atom_features: Type of atom features ("atomic_number", "cgcnn", etc.)
        edge_strategy: Edge construction strategy
            - "knn": k-nearest neighbors
            - "radius": radius-based cutoff
            - "voronoi": Voronoi tessellation
        compute_line_graph: Whether to also return the line graph
        use_symmetry: Whether to use asymmetric unit cell (with symmetry info)
        tolerance: Tolerance for k-NN edge selection
        min_ridge_area: Minimum ridge area for Voronoi edges

    Returns:
        DGL graph (and optionally line graph)
    """
    # Convert jarvis Atoms to pymatgen Structure
    structure = jarvis_to_pymatgen(atoms)

    # Build empty graph with node information
    nx_graph = graph_builder.structure_to_empty_graph(
        structure, symmetrize=use_symmetry
    )

    # Add edges based on strategy
    if edge_strategy == "knn":
        nx_graph = graph_builder.add_knn_bonds(
            nx_graph, k=max_neighbors, max_radius=cutoff,
            tolerance=tolerance, inplace=True
        )
    elif edge_strategy == "radius":
        nx_graph = graph_builder.add_radius_bonds(
            nx_graph, radius=cutoff, inplace=True
        )
    elif edge_strategy == "voronoi":
        nx_graph = graph_builder.add_voronoi_bonds(
            nx_graph, min_ridge_area=min_ridge_area, inplace=True
        )
    else:
        raise ValueError(f"Unknown edge_strategy: {edge_strategy}")

    # Add edge information (offsets and distances)
    nx_graph = graph_builder.add_edge_information(nx_graph, inplace=True)

    # Remove duplicate edges
    nx_graph = graph_builder.remove_duplicate_edges(nx_graph, inplace=True)

    # Convert to asymmetric unit if symmetry is used
    if use_symmetry:
        nx_graph = graph_builder.to_asymmetric_unit_graph(nx_graph)

    # Convert to DGL format
    return networkx_to_dgl(
        nx_graph,
        atom_features=atom_features,
        include_line_graph=compute_line_graph
    )


# =============================================================================
# Convenience Functions (backwards compatible API)
# =============================================================================

def build_cogn_dgl_graph(
    atoms: JarvisAtoms,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    atom_features: str = "atomic_number",
    compute_line_graph: bool = False,
    use_canonize: bool = False,
) -> Union[dgl.DGLGraph, Tuple[dgl.DGLGraph, dgl.DGLGraph]]:
    """Create a DGLGraph using kgcnn's graph construction logic.

    This is a backwards-compatible wrapper around build_kgcnn_graph.

    Args:
        atoms: jarvis Atoms object
        cutoff: Distance cutoff (Angstrom)
        max_neighbors: Maximum number of neighbors for k-NN
        atom_features: Type of atom features
        compute_line_graph: Whether to also return the line graph
        use_canonize: Whether to use canonical edge representation (ignored, for compatibility)

    Returns:
        DGL graph (and optionally line graph)
    """
    return build_kgcnn_graph(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        atom_features=atom_features,
        edge_strategy="knn",
        compute_line_graph=compute_line_graph,
        use_symmetry=False,
    )


def build_cogn_kgcnn_graph(
    atoms: JarvisAtoms,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    atom_features: str = "atomic_number",
    compute_line_graph: bool = False,
    use_canonize: bool = True,
) -> Union[dgl.DGLGraph, Tuple[dgl.DGLGraph, dgl.DGLGraph]]:
    """Create a DGLGraph mirroring the kgcnn coGN crystal graph inputs.

    This is a backwards-compatible wrapper around build_kgcnn_graph.

    Args:
        atoms: jarvis Atoms object
        cutoff: Distance cutoff (Angstrom)
        max_neighbors: Maximum number of neighbors for k-NN
        atom_features: Type of atom features
        compute_line_graph: Whether to also return the line graph
        use_canonize: Whether to use canonical edge representation (ignored, for compatibility)

    Returns:
        DGL graph (and optionally line graph)
    """
    return build_kgcnn_graph(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        atom_features=atom_features,
        edge_strategy="knn",
        compute_line_graph=compute_line_graph,
        use_symmetry=False,
    )


# =============================================================================
# Extended Graph Building Functions
# =============================================================================

def build_radius_graph(
    atoms: JarvisAtoms,
    radius: float = 5.0,
    atom_features: str = "atomic_number",
    compute_line_graph: bool = False,
    use_symmetry: bool = False,
) -> Union[dgl.DGLGraph, Tuple[dgl.DGLGraph, dgl.DGLGraph]]:
    """Build a radius-based crystal graph.

    Args:
        atoms: jarvis Atoms object
        radius: Cutoff radius (Angstrom)
        atom_features: Type of atom features
        compute_line_graph: Whether to also return the line graph
        use_symmetry: Whether to use asymmetric unit cell

    Returns:
        DGL graph (and optionally line graph)
    """
    return build_kgcnn_graph(
        atoms=atoms,
        cutoff=radius,
        atom_features=atom_features,
        edge_strategy="radius",
        compute_line_graph=compute_line_graph,
        use_symmetry=use_symmetry,
    )


def build_voronoi_graph(
    atoms: JarvisAtoms,
    atom_features: str = "atomic_number",
    compute_line_graph: bool = False,
    use_symmetry: bool = False,
    min_ridge_area: float = 0.0,
) -> Union[dgl.DGLGraph, Tuple[dgl.DGLGraph, dgl.DGLGraph]]:
    """Build a Voronoi-based crystal graph.

    Args:
        atoms: jarvis Atoms object
        atom_features: Type of atom features
        compute_line_graph: Whether to also return the line graph
        use_symmetry: Whether to use asymmetric unit cell
        min_ridge_area: Minimum ridge area for edge inclusion

    Returns:
        DGL graph (and optionally line graph)
    """
    return build_kgcnn_graph(
        atoms=atoms,
        cutoff=10.0,  # Not used for Voronoi, but needed for the function signature
        atom_features=atom_features,
        edge_strategy="voronoi",
        compute_line_graph=compute_line_graph,
        use_symmetry=use_symmetry,
        min_ridge_area=min_ridge_area,
    )


def build_supercell_graph(
    atoms: JarvisAtoms,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    atom_features: str = "atomic_number",
    edge_strategy: Literal["knn", "radius", "voronoi"] = "knn",
    compute_line_graph: bool = False,
    supercell_size: Tuple[int, int, int] = (3, 3, 3),
) -> Union[dgl.DGLGraph, Tuple[dgl.DGLGraph, dgl.DGLGraph]]:
    """Build a supercell crystal graph.

    Args:
        atoms: jarvis Atoms object
        cutoff: Distance cutoff (Angstrom)
        max_neighbors: Maximum number of neighbors for k-NN
        atom_features: Type of atom features
        edge_strategy: Edge construction strategy
        compute_line_graph: Whether to also return the line graph
        supercell_size: Size of the supercell in each dimension

    Returns:
        DGL graph (and optionally line graph)
    """
    # Convert to pymatgen
    structure = jarvis_to_pymatgen(atoms)

    # Build empty graph
    nx_graph = graph_builder.structure_to_empty_graph(structure, symmetrize=False)

    # Add edges based on strategy
    if edge_strategy == "knn":
        nx_graph = graph_builder.add_knn_bonds(
            nx_graph, k=max_neighbors, max_radius=cutoff, inplace=True
        )
    elif edge_strategy == "radius":
        nx_graph = graph_builder.add_radius_bonds(
            nx_graph, radius=cutoff, inplace=True
        )
    elif edge_strategy == "voronoi":
        nx_graph = graph_builder.add_voronoi_bonds(nx_graph, inplace=True)

    # Add edge information
    nx_graph = graph_builder.add_edge_information(nx_graph, inplace=True)

    # Convert to supercell
    nx_graph = graph_builder.to_supercell_graph(nx_graph, size=list(supercell_size))

    # Remove duplicate edges
    nx_graph = graph_builder.remove_duplicate_edges(nx_graph, inplace=True)

    return networkx_to_dgl(
        nx_graph,
        atom_features=atom_features,
        include_line_graph=compute_line_graph
    )


# =============================================================================
# Legacy Functions (for backwards compatibility with old cogn_graphs.py)
# =============================================================================

def _canonize_edge(
    src_id: int,
    dst_id: int,
    src_image: Tuple[int, int, int],
    dst_image: Tuple[int, int, int],
):
    """Compute canonical edge representation with periodic images.

    Note: This is kept for backwards compatibility but is no longer used
    in the main graph building functions.
    """
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    return src_id, dst_id, src_image, dst_image


def cogn_compute_bond_cosines(edges):
    """Compute bond angle cosines for coGN line graphs.

    Note: This is an alias for _compute_bond_cosines for backwards compatibility.
    """
    return _compute_bond_cosines(edges)
