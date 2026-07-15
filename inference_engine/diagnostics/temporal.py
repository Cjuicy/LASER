"""Temporal segment-graph stability observations."""

from __future__ import annotations

import numpy as np


def summarize_temporal_graph(graphs) -> dict:
    if not graphs:
        return {"valid": False, "invalid_reason": "empty_graph"}
    incoming = {id(vertex): 0 for layer in graphs for vertex in layer}
    weights: list[float] = []
    one_to_many = 0
    matched_pixels = 0
    total_pixels = 0
    lifetimes = {id(vertex): 1 for layer in graphs for vertex in layer}
    for layer_index, layer in enumerate(graphs):
        for vertex in layer:
            area = int(np.asarray(vertex.data, bool).sum())
            total_pixels += area
            if vertex.connectivity or incoming[id(vertex)] > 0:
                matched_pixels += area
            if len(vertex.connectivity) > 1:
                one_to_many += 1
            for target, weight in zip(vertex.connectivity, vertex.edge_weights):
                incoming[id(target)] = incoming.get(id(target), 0) + 1
                weights.append(float(weight))
                lifetimes[id(target)] = max(lifetimes.get(id(target), 1), lifetimes[id(vertex)] + 1)
    # A second pass includes target vertices that only gained incoming edges after first visit.
    matched_pixels = sum(
        int(np.asarray(vertex.data, bool).sum())
        for layer in graphs for vertex in layer
        if vertex.connectivity or incoming.get(id(vertex), 0) > 0
    )
    many_to_one = sum(value > 1 for value in incoming.values())
    return {
        "valid": True,
        "invalid_reason": None,
        "edge_count": len(weights),
        "one_to_many_count": int(one_to_many),
        "many_to_one_count": int(many_to_one),
        "matched_area_ratio": float(matched_pixels / total_pixels) if total_pixels else None,
        "unmatched_area_ratio": float(1 - matched_pixels / total_pixels) if total_pixels else None,
        "weighted_mean_iou": float(np.mean(weights)) if weights else None,
        "iou_quantiles": ({key: float(np.quantile(weights, q)) for key, q in (("p10", .1), ("p50", .5), ("p90", .9))} if weights else {"p10": None, "p50": None, "p90": None}),
        "max_segment_lifetime": int(max(lifetimes.values(), default=0)),
        "segment_churn_ratio": float(sum(value == 0 for value in incoming.values()) / len(incoming)) if incoming else None,
    }


def trace_temporal_graph(graphs) -> dict:
    metrics = summarize_temporal_graph(graphs)
    if not graphs or not graphs[0]:
        return {"metrics": metrics, "arrays": {}}
    shape = np.asarray(graphs[0][0].data).shape
    outgoing_map = np.zeros((len(graphs), *shape), dtype=np.uint16)
    incoming_map = np.zeros((len(graphs), *shape), dtype=np.uint16)
    best_iou_map = np.zeros((len(graphs), *shape), dtype=np.float32)
    lifetime_map = np.ones((len(graphs), *shape), dtype=np.uint16)
    vertex_location = {
        id(vertex): (layer_index, vertex)
        for layer_index, layer in enumerate(graphs)
        for vertex in layer
    }
    incoming = {vertex_id: 0 for vertex_id in vertex_location}
    lifetime = {vertex_id: 1 for vertex_id in vertex_location}
    best_incoming = {vertex_id: 0.0 for vertex_id in vertex_location}
    for layer_index, layer in enumerate(graphs):
        for vertex in layer:
            mask = np.asarray(vertex.data, dtype=bool)
            outgoing_map[layer_index][mask] = len(vertex.connectivity)
            if vertex.edge_weights:
                best_iou_map[layer_index][mask] = max(vertex.edge_weights)
            for target, weight in zip(vertex.connectivity, vertex.edge_weights):
                incoming[id(target)] = incoming.get(id(target), 0) + 1
                best_incoming[id(target)] = max(best_incoming.get(id(target), 0.0), float(weight))
                lifetime[id(target)] = max(lifetime.get(id(target), 1), lifetime[id(vertex)] + 1)
    for vertex_id, (layer_index, vertex) in vertex_location.items():
        mask = np.asarray(vertex.data, dtype=bool)
        incoming_map[layer_index][mask] = incoming.get(vertex_id, 0)
        lifetime_map[layer_index][mask] = lifetime.get(vertex_id, 1)
        best_iou_map[layer_index][mask] = max(
            float(best_iou_map[layer_index][mask][0]) if mask.any() else 0.0,
            best_incoming.get(vertex_id, 0.0),
        )
    return {"metrics": metrics, "arrays": {
        "temporal_outgoing_degree_map": outgoing_map,
        "temporal_incoming_degree_map": incoming_map,
        "temporal_best_iou_map": best_iou_map,
        "temporal_lifetime_map": lifetime_map,
    }}
