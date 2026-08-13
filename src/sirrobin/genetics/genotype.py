"""Fixed-capacity innovation-marked genotype tensors and donor-fixture ingestion."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import torch

from sirrobin.genetics.frame import donor_quat_to_flu, donor_vector_to_flu
from sirrobin.numerics.quat import euler_unity_deg, identity

SEGMENT = 0
SURFACE = 1
N_MAX = 24
E_MAX = 48


def _collect_donor_tree(
    gene: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[tuple[int, int, dict[str, Any]]],
    parent: int | None = None,
) -> int:
    index = len(nodes)
    if index >= N_MAX:
        return -1
    nodes.append(gene)
    if parent is not None:
        edges.append((parent, index, gene))
    for child in gene["children"]:
        _collect_donor_tree(child, nodes, edges, index)
    return index


@dataclass(slots=True)
class GenotypeBatch:
    alive: torch.Tensor
    stable_id: torch.Tensor
    node_mask: torch.Tensor
    node_iid: torch.Tensor
    node_type: torch.Tensor
    node_log_axes_flu_m: torch.Tensor
    node_density_gene: torch.Tensor
    node_intake: torch.Tensor
    node_sense: torch.Tensor
    node_expressed: torch.Tensor
    node_joint_amp_rad: torch.Tensor
    node_hinge_axis_flu: torch.Tensor
    edge_mask: torch.Tensor
    edge_iid: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_attach_parent_axes: torch.Tensor
    edge_rot_flu: torch.Tensor
    edge_scale: torch.Tensor
    edge_mirror: torch.Tensor
    edge_recursion: torch.Tensor
    swim_freq_hz: torch.Tensor
    swim_wave_rad_per_depth: torch.Tensor

    @property
    def worlds(self) -> int:
        return int(self.alive.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.alive.shape[1])

    @property
    def batch_size(self) -> int:
        return self.worlds * self.capacity

    def to(self, device: torch.device | str) -> GenotypeBatch:
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return type(self)(**values)

    def validate(self) -> None:
        lead = tuple(self.alive.shape)
        if len(lead) != 2 or self.alive.dtype != torch.bool:
            raise TypeError("alive must be bool [W,N]")
        if self.stable_id.dtype != torch.int64 or tuple(self.stable_id.shape) != lead:
            raise TypeError("stable_id must be int64 [W,N]")
        if tuple(self.node_mask.shape) != (*lead, N_MAX) or self.node_mask.dtype != torch.bool:
            raise TypeError("node_mask must be bool [W,N,24]")
        if tuple(self.edge_mask.shape) != (*lead, E_MAX) or self.edge_mask.dtype != torch.bool:
            raise TypeError("edge_mask must be bool [W,N,48]")
        if torch.any(self.alive & ~(self.node_mask[..., 0] & self.node_expressed[..., 0])):
            raise ValueError("every live genotype requires an expressed root in node slot 0")
        invalid_node_iid = torch.any(self.node_mask & (self.node_iid <= 0))
        invalid_edge_iid = torch.any(self.edge_mask & (self.edge_iid <= 0))
        if invalid_node_iid or invalid_edge_iid:
            raise ValueError("expressed genes require positive innovation ids")
        if torch.any(self.edge_mask & ((self.edge_src < 0) | (self.edge_src >= N_MAX))):
            raise ValueError("edge source outside node capacity")
        if torch.any(self.edge_mask & ((self.edge_dst < 0) | (self.edge_dst >= N_MAX))):
            raise ValueError("edge destination outside node capacity")
        if torch.any(self.edge_mask & (self.edge_recursion < 1)):
            raise ValueError("edge recursion must be at least one")
        if torch.any(~torch.isfinite(self.node_log_axes_flu_m)) or torch.any(
            ~torch.isfinite(self.edge_attach_parent_axes)
        ):
            raise ValueError("genotype contains nonfinite geometry")

    @classmethod
    def from_donor_rows(
        cls,
        rows: list[dict[str, Any]],
        *,
        worlds: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> GenotypeBatch:
        if len(rows) % worlds:
            raise ValueError("row count must divide evenly across worlds")
        capacity = len(rows) // worlds
        lead = (worlds, capacity)

        def zeros_n(*tail: int, dtype_: torch.dtype = dtype) -> torch.Tensor:
            return torch.zeros((*lead, N_MAX, *tail), dtype=dtype_)

        def zeros_e(*tail: int, dtype_: torch.dtype = dtype) -> torch.Tensor:
            return torch.zeros((*lead, E_MAX, *tail), dtype=dtype_)

        node_mask = zeros_n(dtype_=torch.bool)
        node_iid = zeros_n(dtype_=torch.int64)
        node_type = zeros_n(dtype_=torch.int8)
        node_log_axes = zeros_n(3)
        node_density = zeros_n()
        node_intake = zeros_n(dtype_=torch.bool)
        node_sense = zeros_n(dtype_=torch.bool)
        node_expressed = zeros_n(dtype_=torch.bool)
        node_amp = zeros_n()
        node_hinge = zeros_n(3)
        node_hinge[..., 2] = 1.0
        edge_mask = zeros_e(dtype_=torch.bool)
        edge_iid = zeros_e(dtype_=torch.int64)
        edge_src = zeros_e(dtype_=torch.int16)
        edge_dst = zeros_e(dtype_=torch.int16)
        edge_attach = zeros_e(3)
        edge_rot = identity((*lead, E_MAX), dtype=dtype, device="cpu")
        edge_scale = torch.ones((*lead, E_MAX), dtype=dtype)
        edge_mirror = zeros_e(dtype_=torch.bool)
        edge_recursion = torch.ones((*lead, E_MAX), dtype=torch.int8)
        swim_freq = torch.zeros(lead, dtype=dtype)
        swim_wave = torch.zeros(lead, dtype=dtype)

        for flat, row in enumerate(rows):
            wi, ni = divmod(flat, capacity)
            genome = row["genotype"]
            swim_freq[wi, ni] = float(genome["swim_freq_hz"])
            swim_wave[wi, ni] = float(genome["swim_wave_rad_per_depth"])
            nodes: list[dict[str, Any]] = []
            edges: list[tuple[int, int, dict[str, Any]]] = []
            _collect_donor_tree(genome["root"], nodes, edges)
            if len(edges) > E_MAX:
                raise ValueError("donor tree exceeds edge capacity")
            for node_index, gene in enumerate(nodes):
                size = torch.tensor(gene["size"], dtype=dtype)
                donor_dims = torch.stack(
                    (size[2].clamp_min(0.3), size[0].clamp_min(0.12), size[1].clamp_min(0.12))
                )
                axes = 0.5 * donor_dims
                node_mask[wi, ni, node_index] = True
                node_iid[wi, ni, node_index] = flat * 1000 + node_index + 1
                node_type[wi, ni, node_index] = SURFACE if gene["type"] == "Surface" else SEGMENT
                node_log_axes[wi, ni, node_index] = axes.log()
                node_density[wi, ni, node_index] = float(gene["density"])
                node_intake[wi, ni, node_index] = gene["port"] == "Intake"
                node_sense[wi, ni, node_index] = gene["port"] == "Sense"
                node_expressed[wi, ni, node_index] = True
                node_amp[wi, ni, node_index] = math.radians(
                    min(58.0, max(0.0, float(gene["joint_amp_deg"])))
                )
            for edge_index, (src, dst, gene) in enumerate(edges):
                parent_axes = node_log_axes[wi, ni, src].exp()
                donor_pos = torch.tensor(gene["attach"], dtype=dtype)
                local_pos = donor_vector_to_flu(donor_pos)
                donor_euler = torch.tensor(gene["orient_deg"], dtype=dtype)
                local_rot = donor_quat_to_flu(euler_unity_deg(donor_euler))
                edge_mask[wi, ni, edge_index] = True
                edge_iid[wi, ni, edge_index] = flat * 1000 + 500 + edge_index + 1
                edge_src[wi, ni, edge_index] = src
                edge_dst[wi, ni, edge_index] = dst
                edge_attach[wi, ni, edge_index] = local_pos / parent_axes
                edge_rot[wi, ni, edge_index] = local_rot
                edge_mirror[wi, ni, edge_index] = bool(gene["mirror"])

        result = cls(
            alive=torch.ones(lead, dtype=torch.bool),
            stable_id=torch.arange(1, len(rows) + 1, dtype=torch.int64).reshape(lead),
            node_mask=node_mask,
            node_iid=node_iid,
            node_type=node_type,
            node_log_axes_flu_m=node_log_axes,
            node_density_gene=node_density,
            node_intake=node_intake,
            node_sense=node_sense,
            node_expressed=node_expressed,
            node_joint_amp_rad=node_amp,
            node_hinge_axis_flu=node_hinge,
            edge_mask=edge_mask,
            edge_iid=edge_iid,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attach_parent_axes=edge_attach,
            edge_rot_flu=edge_rot,
            edge_scale=edge_scale,
            edge_mirror=edge_mirror,
            edge_recursion=edge_recursion,
            swim_freq_hz=swim_freq,
            swim_wave_rad_per_depth=swim_wave,
        ).to(device)
        result.validate()
        return result
