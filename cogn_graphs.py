"""coGN-specific graph construction helpers.

These utilities mirror the neighbor search and edge feature generation used by
``graph.py`` but keep the implementation isolated for coGN so the model can
build its own graph representations without relying on ALIGNN utilities.
"""

from collections import defaultdict
from typing import Tuple

import dgl
import numpy as np
import torch
from jarvis.core.specie import get_node_attributes


def _canonize_edge(
    src_id: int,
    dst_id: int,
    src_image: Tuple[int, int, int],
    dst_image: Tuple[int, int, int],
):
    """Compute canonical edge representation with periodic images."""
    if dst_id < src_id:
        src_id, dst_id = dst_id, src_id
        src_image, dst_image = dst_image, src_image

    if not np.array_equal(src_image, (0, 0, 0)):
        shift = src_image
        src_image = tuple(np.subtract(src_image, shift))
        dst_image = tuple(np.subtract(dst_image, shift))

    return src_id, dst_id, src_image, dst_image


def cogn_nearest_neighbor_edges(
    atoms,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    use_canonize: bool = False,
):
    """Construct k-NN edge list for coGN graphs."""
    all_neighbors = atoms.get_all_neighbors(r=cutoff)
    min_nbrs = min(len(neighborlist) for neighborlist in all_neighbors)

    if min_nbrs < max_neighbors:
        lat = atoms.lattice
        r_cut = max(lat.a, lat.b, lat.c) if cutoff < max(lat.a, lat.b, lat.c) else 2 * cutoff
        return cogn_nearest_neighbor_edges(
            atoms=atoms,
            cutoff=r_cut,
            max_neighbors=max_neighbors,
            use_canonize=use_canonize,
        )

    edges = defaultdict(set)
    for site_idx, neighborlist in enumerate(all_neighbors):
        neighborlist = sorted(neighborlist, key=lambda x: x[2])
        distances = np.array([nbr[2] for nbr in neighborlist])
        ids = np.array([nbr[1] for nbr in neighborlist])
        images = np.array([nbr[3] for nbr in neighborlist])

        max_dist = distances[max_neighbors - 1]
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]

        for dst, image in zip(ids, images):
            src_id, dst_id, src_image, dst_image = _canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges


def _build_undirected_edgedata_with_translations(atoms, edges):
    """Build undirected graph data and keep periodic translations."""
    u, v, r, translations = [], [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            displacement = atoms.lattice.cart_coords(dst_coord - atoms.frac_coords[src_id])
            translation = np.asarray(dst_image)

            for uu, vv, dd, tt in [
                (src_id, dst_id, displacement, translation),
                (dst_id, src_id, -displacement, -translation),
            ]:
                u.append(uu)
                v.append(vv)
                r.append(dd)
                translations.append(tt)

    u = torch.tensor(u)
    v = torch.tensor(v)
    r = torch.tensor(r).type(torch.get_default_dtype())
    translations = torch.tensor(translations, dtype=torch.int64)

    return u, v, r, translations


def cogn_compute_bond_cosines(edges):
    """Compute bond angle cosines for coGN line graphs."""
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {"h": bond_cosine}


def build_cogn_dgl_graph(
    atoms,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    atom_features: str = "atomic_number",
    compute_line_graph: bool = False,
    use_canonize: bool = False,
):
    """Create a DGLGraph using coGN's neighbor construction logic."""
    edges = cogn_nearest_neighbor_edges(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_canonize=use_canonize,
    )

    u, v, r, translations = _build_undirected_edgedata_with_translations(atoms, edges)

    atom_attr = []
    for element in atoms.elements:
        feat = list(get_node_attributes(element, atom_features=atom_features))
        atom_attr.append(feat)
    atom_attr = torch.tensor(np.array(atom_attr)).type(torch.get_default_dtype())

    g = dgl.graph((u, v))
    g.ndata["atom_features"] = atom_attr
    g.ndata["frac_coords"] = torch.tensor(atoms.frac_coords).type(torch.get_default_dtype())
    g.edata["r"] = r
    g.edata["cell_translation"] = translations

    if compute_line_graph:
        lg = g.line_graph(shared=True)
        lg.apply_edges(cogn_compute_bond_cosines)
        return g, lg

    return g


def build_cogn_kgcnn_graph(
    atoms,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    atom_features: str = "atomic_number",
    compute_line_graph: bool = False,
    use_canonize: bool = True,
):
    """Create a DGLGraph mirroring the kgcnn coGN crystal graph inputs."""

    edges = cogn_nearest_neighbor_edges(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_canonize=use_canonize,
    )

    u, v, offsets, translations = _build_undirected_edgedata_with_translations(atoms, edges)

    atom_numbers = []
    for element in atoms.elements:
        feat = list(get_node_attributes(element, atom_features=atom_features))
        atom_numbers.append(feat)

    g = dgl.graph((u, v))
    g.ndata["atom_features"] = torch.tensor(np.array(atom_numbers)).type(torch.get_default_dtype())
    g.ndata["atomic_number"] = g.ndata["atom_features"]
    g.ndata["frac_coords"] = torch.tensor(atoms.frac_coords).type(torch.get_default_dtype())
    g.edata["offset"] = offsets
    g.edata["r"] = offsets
    g.edata["cell_translation"] = translations
    g.edata["edge_indices"] = torch.stack((u, v), dim=1)
    g.lattice_matrix = torch.tensor(atoms.lattice.matrix).type(torch.get_default_dtype())

    if compute_line_graph:
        lg = g.line_graph(shared=True)
        lg.apply_edges(cogn_compute_bond_cosines)
        return g, lg

    return g
