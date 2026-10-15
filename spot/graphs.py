"""Directed draft graphs: representation, calibration bookkeeping, materialisation, pruning.

A draft *formula* (a graph node) is a frozenset of (i, j) pairs, where

  i = token-position rank: the rank of a still-masked slot when the slots are ordered by
      the drafting distribution's max-probability (1 = most confident), and
  j = vocabulary rank of the token chosen for that slot under the drafting distribution
      at that slot (1 = argmax).

At inference the formula is *materialised* against the live distribution to give a
concrete draft block state.  Paper: Spiffy (arXiv 2509.18085v4) Sec 4.4.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

Pair = Tuple[int, int]
Formula = FrozenSet[Pair]


@dataclass
class Node:
    formula: Formula
    level: int  # lookahead depth (number of commits ahead) at which it was observed
    count: int  # calibration frequency

    def key(self) -> str:
        return ";".join(f"{i},{j}" for i, j in sorted(self.formula))


@dataclass
class DraftGraph:
    nodes: List[Node]
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.nodes)

    def children(self, idx: int) -> List[int]:
        f = self.nodes[idx].formula
        return [k for k, n in enumerate(self.nodes) if f < n.formula]

    def children_map(self) -> Dict[int, List[int]]:
        if getattr(self, "_children", None) is None:
            self._children = {i: self.children(i) for i in range(len(self.nodes))}
        return self._children

    def max_vocab_rank(self) -> int:
        return max((j for n in self.nodes for _, j in n.formula), default=1)

    def to_json(self) -> dict:
        return {
            "nodes": [
                {"formula": sorted(map(list, n.formula)), "level": n.level, "count": n.count}
                for n in self.nodes
            ],
            "meta": self.meta,
        }

    @staticmethod
    def from_json(d: dict) -> "DraftGraph":
        return DraftGraph(
            nodes=[
                Node(frozenset((int(i), int(j)) for i, j in n["formula"]), int(n["level"]), int(n["count"]))
                for n in d["nodes"]
            ],
            meta=d.get("meta", {}))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    @staticmethod
    def load(path: str) -> "DraftGraph":
        with open(path) as f:
            return DraftGraph.from_json(json.load(f))

    def describe(self) -> str:
        out = []
        for i, n in enumerate(self.nodes):
            out.append(
                f"  [{i}] level={n.level} count={n.count} formula={{{', '.join(f'c{a}{b}' for a, b in sorted(n.formula))}}}"
            )
        return "\n".join(out)


# ----------------------------------------------------------------------------------
# Calibration: Algorithm 2 of the paper.
# ----------------------------------------------------------------------------------
def build_candidate_graph(
    sequences: Sequence[Sequence[Formula]], lookahead: int, top_k: int = 3
) -> Tuple[List[Node], Dict[Formula, int]]:
    """Part 2: per level, keep the `top_k` most frequent cumulative formulas.

    `sequences[a][k-1]` is the cumulative formula observed at anchor `a`, lookahead `k`.
    """
    counts: List[Dict[Formula, int]] = [dict() for _ in range(lookahead)]
    for seq in sequences:
        for k, formula in enumerate(seq):
            if formula is None:
                break
            counts[k][formula] = counts[k].get(formula, 0) + 1
    nodes: List[Node] = []
    seen: Dict[Formula, int] = {}
    for k in range(lookahead):
        ranked = sorted(counts[k].items(), key=lambda kv: -kv[1])[:top_k]
        for formula, c in ranked:
            if formula in seen:  # keep it at its shallowest level, accumulate the count
                nodes[seen[formula]].count += c
                continue
            seen[formula] = len(nodes)
            nodes.append(Node(formula=formula, level=k + 1, count=c))
    flat = {f: nodes[i].count for f, i in seen.items()}
    return nodes, flat


def select_best_subgraph(nodes: List[Node], budget: int) -> List[Node]:
    """Part 3: highest total count over connected size-`budget` subgraphs.

    Connected (Eq. 24-26): every chosen node is at level 1 or 2, or has a parent
    (a strict subset, per Eq. 4-6) inside the chosen set.
    """
    n = len(nodes)
    if n <= budget:
        return list(nodes)
    parents = [[p for p in range(n) if nodes[p].formula < nodes[q].formula] for q in range(n)]
    roots = [q for q in range(n) if nodes[q].level <= 2]

    best_score, best_set = -1, None
    order = sorted(range(n), key=lambda q: -nodes[q].count)
    for combo in itertools.combinations(order, budget):
        score = sum(nodes[q].count for q in combo)
        if score <= best_score:
            continue
        s = set(combo)
        if all(q in roots or any(p in s for p in parents[q]) for q in combo):
            best_score, best_set = score, combo
    if best_set is None:  # no connected subgraph of that size; fall back to greedy roots-first
        chosen = sorted(roots, key=lambda q: -nodes[q].count)[:budget]
        rest = [q for q in order if q not in chosen]
        chosen += rest[: budget - len(chosen)]
        best_set = tuple(chosen)
    return [nodes[q] for q in sorted(best_set, key=lambda q: (nodes[q].level, -nodes[q].count))]
