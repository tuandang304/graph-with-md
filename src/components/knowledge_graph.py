"""
Knowledge Graph Manager — NetworkX-based graph for structural retrieval.

Manages a directed multi-graph built from LLM-extracted triplets.
Provides multi-hop traversal, subgraph extraction, shortest path,
entity matching, and serialization to LLM-readable text.
"""
import os
import json
import difflib
import networkx as nx
from typing import Optional


class KnowledgeGraph:
    """
    Manages a NetworkX DiGraph for a corpus of papers.

    Design:
    - One merged DiGraph per dataset (all papers combined).
    - Edge attribute ``paper_id`` scopes relations to their source document.
    - Node IDs are normalized (lowercase, stripped) for deduplication;
      the original surface form is stored in the ``label`` node attribute.
    """

    def __init__(self, graph_dir: str):
        self.graph_dir = graph_dir
        self.graph = nx.DiGraph()
        # Mapping: normalized entity → set of paper_ids that mention it
        self._entity_papers: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ #
    #  Entity normalization                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_entity(entity: str) -> str:
        """Lowercase, strip whitespace, collapse internal whitespace."""
        return " ".join(entity.lower().strip().split())

    # ------------------------------------------------------------------ #
    #  Build & Persist                                                     #
    # ------------------------------------------------------------------ #

    def add_triplets(self, paper_id: str, triplets: list[dict]):
        """
        Insert a list of ``{source, target, relation}`` dicts into the graph.
        Each edge is tagged with ``paper_id`` so we can scope traversals later.
        """
        for t in triplets:
            src_raw = t.get("source", "").strip()
            tgt_raw = t.get("target", "").strip()
            rel = t.get("relation", "").strip()
            if not src_raw or not tgt_raw or not rel:
                continue

            src = self.normalize_entity(src_raw)
            tgt = self.normalize_entity(tgt_raw)

            # Add / update nodes with original label
            if src not in self.graph:
                self.graph.add_node(src, label=src_raw)
            if tgt not in self.graph:
                self.graph.add_node(tgt, label=tgt_raw)

            # Add edge (allows parallel edges via key=relation+paper)
            self.graph.add_edge(src, tgt, relation=rel, paper_id=paper_id)

            # Track entity→paper mapping
            self._entity_papers.setdefault(src, set()).add(paper_id)
            self._entity_papers.setdefault(tgt, set()).add(paper_id)

    def save(self, paper_id: str):
        """Persist the subgraph for *paper_id* as a GraphML file."""
        os.makedirs(self.graph_dir, exist_ok=True)
        subgraph = self._paper_subgraph(paper_id)
        if subgraph.number_of_nodes() == 0:
            return
        path = os.path.join(self.graph_dir, f"{paper_id}_graph.graphml")
        nx.write_graphml(subgraph, path)

    def save_all(self):
        """Persist one .graphml per paper_id found in the graph."""
        paper_ids = set()
        for _, _, d in self.graph.edges(data=True):
            paper_ids.add(d.get("paper_id", "unknown"))
        for pid in paper_ids:
            self.save(pid)

    def load(self, paper_id: str):
        """Load a single paper's .graphml and merge into the main graph."""
        path = os.path.join(self.graph_dir, f"{paper_id}_graph.graphml")
        if not os.path.exists(path):
            return
        g = nx.read_graphml(path)
        self._merge(g, paper_id)

    def load_all(self):
        """Load every .graphml in ``graph_dir`` into one merged graph."""
        if not os.path.isdir(self.graph_dir):
            return
        for fname in os.listdir(self.graph_dir):
            if fname.endswith("_graph.graphml"):
                paper_id = fname.replace("_graph.graphml", "")
                self.load(paper_id)

    def load_from_json(self, paper_id: str):
        """Fallback: load triplets from the legacy ``_graph.json`` file."""
        path = os.path.join(self.graph_dir, f"{paper_id}_graph.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "raw_output" in data:
            # Rescue JSONL output from LLM
            rescued = []
            for line in data["raw_output"].split("\n"):
                if line.strip().startswith("{"):
                    try:
                        rescued.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
            data = rescued
        if isinstance(data, list):
            self.add_triplets(paper_id, data)

    def load_all_from_json(self):
        """Fallback: build graphs from all ``_graph.json`` files."""
        if not os.path.isdir(self.graph_dir):
            return
        for fname in os.listdir(self.graph_dir):
            if fname.endswith("_graph.json"):
                paper_id = fname.replace("_graph.json", "")
                self.load_from_json(paper_id)

    # ------------------------------------------------------------------ #
    #  Query Operations                                                    #
    # ------------------------------------------------------------------ #

    def get_neighbors(self, entity: str, hops: int = 2) -> nx.DiGraph:
        """Return the ego-graph (subgraph within *hops* of *entity*)."""
        norm = self.normalize_entity(entity)
        if norm not in self.graph:
            return nx.DiGraph()
        return nx.ego_graph(self.graph, norm, radius=hops)

    def get_subgraph_for_entities(self, entities: list[str], hops: int = 2) -> nx.DiGraph:
        """
        Union of ego-graphs for each entity in *entities*.
        """
        combined_nodes: set[str] = set()
        for ent in entities:
            ego = self.get_neighbors(ent, hops=hops)
            combined_nodes.update(ego.nodes())
        if not combined_nodes:
            return nx.DiGraph()
        return self.graph.subgraph(combined_nodes).copy()

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Return the shortest path (node list) between two entities, or []."""
        src = self.normalize_entity(source)
        tgt = self.normalize_entity(target)
        try:
            return nx.shortest_path(self.graph, src, tgt)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Try undirected fallback
            try:
                return nx.shortest_path(self.graph.to_undirected(), src, tgt)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []

    def get_entity_context(self, entity: str, hops: int = 2) -> str:
        """
        Human-readable multi-hop context for a single entity.
        Used by the Embedder to create rich graph-context chunks.
        """
        ego = self.get_neighbors(entity, hops=hops)
        if ego.number_of_nodes() == 0:
            return ""
        return self.subgraph_to_text(ego, focus_entity=entity)

    def get_connected_components(self) -> list[set[str]]:
        """Return connected components of the underlying undirected graph."""
        undirected = self.graph.to_undirected()
        return [comp for comp in nx.connected_components(undirected)]

    def get_entities_for_paper(self, paper_id: str) -> list[str]:
        """Return all normalized entity names that appear in a given paper."""
        return [ent for ent, pids in self._entity_papers.items() if paper_id in pids]

    # ------------------------------------------------------------------ #
    #  Entity Matching                                                     #
    # ------------------------------------------------------------------ #

    def find_matching_entities(self, query_terms: list[str], threshold: float = 0.6) -> list[str]:
        """
        Fuzzy-match *query_terms* against all graph node IDs.
        Returns a deduplicated list of matched normalized entity names.
        """
        all_nodes = list(self.graph.nodes())
        matched: set[str] = set()
        for term in query_terms:
            norm_term = self.normalize_entity(term)
            # Exact substring match first
            for node in all_nodes:
                if norm_term in node or node in norm_term:
                    matched.add(node)
            # Fuzzy match
            close = difflib.get_close_matches(norm_term, all_nodes, n=3, cutoff=threshold)
            matched.update(close)
        return list(matched)

    def extract_query_entities(self, question: str) -> list[str]:
        """
        Simple entity extraction from a question string.
        Extracts multi-word noun-phrase candidates and matches them
        against the graph. No external NLP library required.
        """
        # Remove common question words and punctuation
        stop_words = {
            "what", "which", "who", "whom", "where", "when", "why", "how",
            "is", "are", "was", "were", "do", "does", "did", "has", "have",
            "had", "be", "been", "being", "will", "would", "could", "should",
            "may", "might", "shall", "can", "the", "a", "an", "and", "or",
            "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "as", "into", "through", "during", "before", "after", "about",
            "between", "this", "that", "these", "those", "it", "its", "they",
            "them", "their", "not", "no", "nor", "so", "if", "then", "than",
            "too", "very", "just", "also", "only", "up", "out", "all", "each",
            "any", "both", "such", "own", "same", "other", "more", "most",
            "some", "many", "much", "used", "using", "use", "based", "propose",
            "proposed", "method", "model", "approach", "paper", "study",
            "experiment", "result", "dataset", "data", "task",
        }

        # Clean question
        import re
        clean = re.sub(r"[^\w\s\-]", " ", question)
        words = clean.split()

        # Generate n-gram candidates (1 to 4 words)
        candidates: list[str] = []
        for n in range(4, 0, -1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                ngram_lower = ngram.lower().strip()
                # Skip if all words are stop words
                ngram_words = ngram_lower.split()
                if all(w in stop_words for w in ngram_words):
                    continue
                if len(ngram_lower) > 2:
                    candidates.append(ngram_lower)

        # Match against graph
        return self.find_matching_entities(candidates, threshold=0.6)

    # ------------------------------------------------------------------ #
    #  Serialization                                                       #
    # ------------------------------------------------------------------ #

    def subgraph_to_text(self, subgraph: nx.DiGraph, focus_entity: Optional[str] = None,
                         max_relations: int = 30) -> str:
        """
        Convert a subgraph to structured, LLM-readable text.

        Output format:
            Entity "BERT":
              → (is_a) → Pre-trained Language Model
              → (used_for) → Text Classification
            Entity "Text Classification":
              → (evaluated_on) → SST-2
        """
        if subgraph.number_of_nodes() == 0:
            return ""

        lines: list[str] = []
        count = 0

        # If focus entity is given, show it first
        node_order = list(subgraph.nodes())
        if focus_entity:
            norm_focus = self.normalize_entity(focus_entity)
            if norm_focus in node_order:
                node_order.remove(norm_focus)
                node_order.insert(0, norm_focus)

        for node in node_order:
            if count >= max_relations:
                break
            out_edges = list(subgraph.out_edges(node, data=True))
            in_edges = list(subgraph.in_edges(node, data=True))
            if not out_edges and not in_edges:
                continue

            label = subgraph.nodes[node].get("label", node)
            lines.append(f'Entity "{label}":')

            for _, tgt, data in out_edges:
                if count >= max_relations:
                    break
                tgt_label = subgraph.nodes[tgt].get("label", tgt)
                rel = data.get("relation", "related_to")
                lines.append(f"  → ({rel}) → {tgt_label}")
                count += 1

            for src, _, data in in_edges:
                if count >= max_relations:
                    break
                src_label = subgraph.nodes[src].get("label", src)
                rel = data.get("relation", "related_to")
                lines.append(f"  ← ({rel}) ← {src_label}")
                count += 1

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Stats                                                               #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        """Return basic graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "components": nx.number_weakly_connected_components(self.graph),
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _paper_subgraph(self, paper_id: str) -> nx.DiGraph:
        """Extract subgraph containing only edges for a given paper_id."""
        edges = [
            (u, v, d) for u, v, d in self.graph.edges(data=True)
            if d.get("paper_id") == paper_id
        ]
        sub = nx.DiGraph()
        for u, v, d in edges:
            if u not in sub:
                sub.add_node(u, **self.graph.nodes[u])
            if v not in sub:
                sub.add_node(v, **self.graph.nodes[v])
            sub.add_edge(u, v, **d)
        return sub

    def _merge(self, g: nx.DiGraph, paper_id: str):
        """Merge a loaded graph into the main graph."""
        for node, data in g.nodes(data=True):
            if node not in self.graph:
                self.graph.add_node(node, **data)
            self._entity_papers.setdefault(node, set()).add(paper_id)
        for u, v, data in g.edges(data=True):
            if "paper_id" not in data:
                data["paper_id"] = paper_id
            self.graph.add_edge(u, v, **data)
            self._entity_papers.setdefault(u, set()).add(paper_id)
            self._entity_papers.setdefault(v, set()).add(paper_id)
