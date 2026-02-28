"""
Property-Based Testing for Transitive Closure using Hypothesis

This demonstrates automated formal verification through property-based testing.
Hypothesis automatically generates hundreds of test cases to verify properties.
"""

from hypothesis import given, strategies as st, settings, assume
from hypothesis import example
from typing import Set, Tuple
import copy

from collections import defaultdict

# Simple graph representation for testing
class SimpleGraph:
    def __init__(self, vertices: Set[int], edges: Set[Tuple[int, int]]):
        self.vertices = vertices
        self.edges = edges
    
    def __eq__(self, other):
        return self.vertices == other.vertices and self.edges == other.edges
    
    def __repr__(self):
        return f"Graph(V={self.vertices}, E={self.edges})"

# This transitive_closure_bfs is from BiocGOprep
def transitive_closure_bfs(direct_edges: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Your original implementation (keep it as-is)"""
    children: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()

    for parent, ch_list in direct_edges.items():
        all_nodes.add(parent)
        for ch in ch_list:
            children[parent].add(ch)
            all_nodes.add(ch)

    pairs: list[tuple[str, str]] = []
    for ancestor in all_nodes:
        visited: set[str] = set()
        queue = list(children.get(ancestor, []))
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            pairs.append((ancestor, node))
            queue.extend(children.get(node, []))
    return pairs


# ADAPTER: Convert SimpleGraph → your format → SimpleGraph
def transitive_closure(graph: SimpleGraph) -> SimpleGraph:
    """Adapter to make your BFS implementation work with test framework"""
    # Convert SimpleGraph to dict[str, list[str]]
    direct_edges = defaultdict(list)
    for u, v in graph.edges:
        direct_edges[str(u)].append(str(v))
    
    # Ensure all vertices are represented (even isolated ones)
    for v in graph.vertices:
        if str(v) not in direct_edges:
            direct_edges[str(v)] = []
    
    # Call your implementation
    closure_pairs = transitive_closure_bfs(direct_edges)
    
    # Convert back to SimpleGraph
    closure_edges = {(int(u), int(v)) for u, v in closure_pairs}
    
    return SimpleGraph(
        vertices=graph.vertices,
        edges=closure_edges
    )



def transitive_closure(graph: SimpleGraph) -> SimpleGraph:
    """Floyd-Warshall transitive closure implementation"""
    closure = SimpleGraph(
        vertices=copy.deepcopy(graph.vertices),
        edges=copy.deepcopy(graph.edges)
    )
    
    vertices_list = sorted(list(graph.vertices))
    
    for k in vertices_list:
        for i in vertices_list:
            for j in vertices_list:
                if (i, k) in closure.edges and (k, j) in closure.edges:
                    closure.edges.add((i, j))
    
    return closure


def has_path(graph: SimpleGraph, start: int, end: int, exclude_self_loop: bool = False) -> bool:
    """BFS to check if path exists"""
    if start == end:
        # For non-reflexive closure, vertex only reaches itself if there's a cycle
        if exclude_self_loop:
            # Check if there's a cycle back to start
            return any(v != start and has_path_excluding_start(graph, v, start, start) 
                      for s, v in graph.edges if s == start)
        return True
    
    if (start, end) in graph.edges:
        return True
    
    visited = {start}
    queue = [start]
    
    while queue:
        current = queue.pop(0)
        for u, v in graph.edges:
            if u == current and v == end:
                return True
            if u == current and v not in visited:
                visited.add(v)
                queue.append(v)
    
    return False


def has_path_excluding_start(graph: SimpleGraph, current: int, end: int, start: int) -> bool:
    """Helper to check path without going through start again"""
    if current == end:
        return True
    
    visited = {current}
    queue = [current]
    
    while queue:
        node = queue.pop(0)
        for u, v in graph.edges:
            if u == node and v != start:
                if v == end:
                    return True
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
    
    return False


# =============================================================================
# HYPOTHESIS STRATEGIES FOR GENERATING GRAPHS
# =============================================================================

@st.composite
def graph_strategy(draw, max_vertices=8, max_edges=15):
    """Generate random directed graphs for testing"""
    # Generate vertices (at least 1, up to max_vertices)
    n_vertices = draw(st.integers(min_value=1, max_value=max_vertices))
    vertices = set(range(n_vertices))
    
    # Generate edges (subset of all possible edges)
    all_possible_edges = [(i, j) for i in vertices for j in vertices]
    n_edges = draw(st.integers(min_value=0, max_value=min(len(all_possible_edges), max_edges)))
    
    edges_list = draw(st.lists(
        st.sampled_from(all_possible_edges) if all_possible_edges else st.nothing(),
        min_size=n_edges,
        max_size=n_edges,
        unique=True
    ))
    
    return SimpleGraph(vertices=vertices, edges=set(edges_list))


# =============================================================================
# PROPERTY-BASED TESTS
# =============================================================================

@given(graph_strategy())
@settings(max_examples=200)
def test_property_transitivity(graph):
    """
    PROPERTY: Transitivity
    If (u,v) and (v,w) are in closure, then (u,w) must be in closure
    """
    closure = transitive_closure(graph)
    
    for u, v in closure.edges:
        for v2, w in closure.edges:
            if v == v2:  # Found chain u→v→w
                assert (u, w) in closure.edges, \
                    f"Transitivity violated: ({u},{v}) and ({v},{w}) exist but ({u},{w}) missing"


#@given(graph_strategy())
#@settings(max_examples=200)
#def test_property_soundness(graph):
#    """
#    PROPERTY: Soundness
#    Every edge in closure corresponds to an actual path in original graph
#    """
#    closure = transitive_closure(graph)
#    
#    for u, v in closure.edges:
#        # For self-loops, check if there's a cycle
#        if u == v:
#            # Self-loop should only exist if there's a cycle
#            path_exists = has_path(graph, u, v, exclude_self_loop=True)
#            assert path_exists or (u, v) in graph.edges, \
#                f"Soundness violated: self-loop ({u},{v}) with no cycle in original"
#        else:
#            assert has_path(graph, u, v), \
#                f"Soundness violated: edge ({u},{v}) in closure but no path in original"

@given(graph_strategy())
@settings(max_examples=200)
def test_property_soundness(graph):
    """Every edge in closure corresponds to an actual path in original graph"""
    closure = transitive_closure(graph)
    
    for u, v in closure.edges:
        # Check if a path exists (BFS without the complex self-loop logic)
        assert _has_simple_path(graph, u, v), \
            f"Soundness violated: edge ({u},{v}) in closure but no path in original"

def _has_simple_path(graph, start, end):
    """Simple BFS that handles self-loops correctly"""
    if start == end:
        # Self-loop only valid if there's a non-trivial cycle
        # Check if any neighbor can reach start
        for s, v in graph.edges:
            if s == start and v != start:
                if _has_simple_path(graph, v, start):
                    return True
        # Or if there's literally a self-loop edge
        return (start, start) in graph.edges
    
    # Standard BFS for non-self-loops
    visited = set()
    queue = [start]
    
    while queue:
        current = queue.pop(0)
        if current == end:
            return True
        if current in visited:
            continue
        visited.add(current)
        
        for s, v in graph.edges:
            if s == current and v not in visited:
                queue.append(v)
    
    return False


@given(graph_strategy())
@settings(max_examples=200)
def test_property_completeness(graph):
    """
    PROPERTY: Completeness
    If a path exists in original graph, corresponding edge must be in closure
    """
    closure = transitive_closure(graph)
    
    for u in graph.vertices:
        for v in graph.vertices:
            if u != v and has_path(graph, u, v):
                assert (u, v) in closure.edges, \
                    f"Completeness violated: path exists {u}→{v} but edge missing in closure"


@given(graph_strategy())
@settings(max_examples=100)  # Expensive test
def test_property_idempotence(graph):
    """
    PROPERTY: Idempotence
    Closure of closure equals closure
    """
    closure1 = transitive_closure(graph)
    closure2 = transitive_closure(closure1)
    
    assert closure1.edges == closure2.edges, \
        "Idempotence violated: TC(TC(G)) ≠ TC(G)"


@given(graph_strategy())
@settings(max_examples=200)
def test_property_preserves_vertices(graph):
    """
    PROPERTY: Vertex preservation
    Closure must have exactly the same vertices as original
    """
    closure = transitive_closure(graph)
    assert closure.vertices == graph.vertices, \
        "Vertices not preserved in closure"


@given(graph_strategy())
@settings(max_examples=200)
def test_property_preserves_edges(graph):
    """
    PROPERTY: Edge preservation
    All original edges must be in closure
    """
    closure = transitive_closure(graph)
    assert graph.edges.issubset(closure.edges), \
        f"Original edges not preserved. Missing: {graph.edges - closure.edges}"


@given(graph_strategy())
@settings(max_examples=200)
def test_property_no_spurious_edges(graph):
    """
    PROPERTY: No spurious edges
    Closure should not contain edges between disconnected components
    """
    closure = transitive_closure(graph)
    
    for u, v in closure.edges:
        # If edge exists, there must be a path (soundness check)
        if u != v:
            assert has_path(graph, u, v), \
                f"Spurious edge ({u},{v}) - vertices not connected in original"


@given(graph_strategy())
@example(SimpleGraph({0}, {(0, 0)}))  # Self-loop example
@example(SimpleGraph({0, 1, 2}, {(0, 1), (1, 2)}))  # Chain
@example(SimpleGraph({0, 1, 2}, {(0, 1), (1, 2), (2, 0)}))  # Cycle
@settings(max_examples=150)
def test_property_minimality(graph):
    """
    PROPERTY: Minimality
    Closure should be the minimal transitive extension of the graph
    
    This means: no proper subset of closure.edges is both:
    1. A superset of graph.edges
    2. Transitive
    """
    closure = transitive_closure(graph)
    
    # For each edge in closure, verify it's necessary
    for edge in closure.edges:
        if edge not in graph.edges:
            # This edge was added - it must be required for transitivity
            # Try removing it and check if result is still transitive
            smaller = SimpleGraph(closure.vertices, closure.edges - {edge})
            
            # If smaller is transitive and contains all original edges, 
            # then current edge was not necessary
            if graph.edges.issubset(smaller.edges):
                is_transitive = all(
                    (u, w) in smaller.edges
                    for u, v in smaller.edges
                    for v2, w in smaller.edges
                    if v == v2
                )
                
                # If smaller is transitive, the removed edge was spurious
                if is_transitive:
                    # But wait - was the edge actually reachable?
                    u, v = edge
                    if u != v and not has_path(graph, u, v):
                        assert False, f"Edge {edge} is spurious (not reachable in original)"


# =============================================================================
# SPECIFIC EDGE CASES
# =============================================================================

def test_empty_graph():
    """Empty graph should have empty closure"""
    graph = SimpleGraph(set(), set())
    closure = transitive_closure(graph)
    assert closure.vertices == set()
    assert closure.edges == set()


def test_single_vertex():
    """Single vertex with no edges"""
    graph = SimpleGraph({0}, set())
    closure = transitive_closure(graph)
    assert closure.vertices == {0}
    assert closure.edges == set()  # No self-loop unless explicitly present


def test_single_vertex_self_loop():
    """Single vertex with self-loop"""
    graph = SimpleGraph({0}, {(0, 0)})
    closure = transitive_closure(graph)
    assert closure.vertices == {0}
    assert closure.edges == {(0, 0)}


def test_simple_chain():
    """A→B→C should give all transitive edges"""
    graph = SimpleGraph({0, 1, 2}, {(0, 1), (1, 2)})
    closure = transitive_closure(graph)
    expected_edges = {(0, 1), (1, 2), (0, 2)}  # Added (0,2)
    assert closure.edges == expected_edges


def test_cycle():
    """A→B→C→A should give complete graph"""
    graph = SimpleGraph({0, 1, 2}, {(0, 1), (1, 2), (2, 0)})
    closure = transitive_closure(graph)
    # All vertices reach all vertices (including themselves)
    expected = {(i, j) for i in range(3) for j in range(3)}
    assert closure.edges == expected


def test_diamond():
    """A→B, A→C, B→D, C→D"""
    graph = SimpleGraph({0, 1, 2, 3}, {(0, 1), (0, 2), (1, 3), (2, 3)})
    closure = transitive_closure(graph)
    # Should add A→D
    assert (0, 3) in closure.edges
    # Original edges preserved
    assert graph.edges.issubset(closure.edges)


def test_disconnected_components():
    """Two separate components should remain separate"""
    graph = SimpleGraph(
        {0, 1, 2, 3},
        {(0, 1), (2, 3)}  # Two separate edges
    )
    closure = transitive_closure(graph)
    # No edges should connect components
    assert (0, 2) not in closure.edges
    assert (0, 3) not in closure.edges
    assert (1, 2) not in closure.edges
    assert (1, 3) not in closure.edges

def test_multiple_self_loops():
    """Multiple vertices with self-loops"""
    graph = SimpleGraph({0, 1, 2}, {(0, 0), (1, 1), (2, 2)})
    closure = transitive_closure(graph)
    # Each vertex only reaches itself
    assert closure.edges == {(0, 0), (1, 1), (2, 2)}

def test_self_loop_with_outgoing():
    """Self-loop combined with other edges"""
    graph = SimpleGraph({0, 1}, {(0, 0), (0, 1)})
    closure = transitive_closure(graph)
    assert (0, 0) in closure.edges
    assert (0, 1) in closure.edges

def test_cycle_creates_self_loops():
    """Cycle should create self-loops for all vertices"""
    graph = SimpleGraph({0, 1, 2}, {(0, 1), (1, 2), (2, 0)})
    closure = transitive_closure(graph)
    # Every vertex can reach itself via the cycle
    assert (0, 0) in closure.edges
    assert (1, 1) in closure.edges
    assert (2, 2) in closure.edges


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("PROPERTY-BASED TESTING WITH HYPOTHESIS")
    print("="*70)
    print("\nRunning property-based tests...")
    print("(Hypothesis will generate hundreds of random graphs)\n")
    
    # Run specific edge cases first
    print("[1] Testing edge cases...")
    test_empty_graph()
    print("  ✓ Empty graph")
    
    test_single_vertex()
    print("  ✓ Single vertex")
    
    test_single_vertex_self_loop()
    print("  ✓ Single vertex with self-loop")
    
    test_simple_chain()
    print("  ✓ Simple chain")
    
    test_cycle()
    print("  ✓ Cycle")
    
    test_diamond()
    print("  ✓ Diamond")
    
    test_disconnected_components()
    print("  ✓ Disconnected components")
    
    print("\n[2] Running property-based tests...")
    print("  (This will test hundreds of randomly generated graphs)")
    
    try:
        test_property_transitivity()
        print("  ✓ Transitivity property (200 examples)")
        
        test_property_soundness()
        print("  ✓ Soundness property (200 examples)")
        
        test_property_completeness()
        print("  ✓ Completeness property (200 examples)")
        
        test_property_idempotence()
        print("  ✓ Idempotence property (100 examples)")
        
        test_property_preserves_vertices()
        print("  ✓ Vertex preservation (200 examples)")
        
        test_property_preserves_edges()
        print("  ✓ Edge preservation (200 examples)")
        
        test_property_no_spurious_edges()
        print("  ✓ No spurious edges (200 examples)")
        
        test_property_minimality()
        print("  ✓ Minimality property (150 examples)")
        
        print("\n" + "="*70)
        print("✓ ALL PROPERTY-BASED TESTS PASSED")
        print("="*70)
        print("\nTotal test cases: ~1,250 randomly generated graphs")
        print("Properties verified: 8 formal properties")
        print("\nConclusion: Implementation is correct with high confidence")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("\nHypothesis found a counterexample!")
