"""
build_parquet.py
-------------------
Parse a GO OBO file and write a directory of Parquet files, one per logical
table, using the same natural-key (go_id) schema as build_go_db.py.

The OBO parsing logic (parse_obo, parse_synonym_tag, transitive_closure,
NS_MAP) is imported directly from build_go_db.py -- the two files must live
in the same directory.

Dependencies
------------
Only pyarrow is required beyond the standard library.  To avoid conflicts
with whatever scientific packages your system Python carries, run inside a
clean virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate           # Windows: .venv\\Scripts\\activate
    pip install pyarrow

Usage:
    python -m go_ontology.build_parquet go.obo output_dir/

Output
------
output_dir/
    go_ontology.parquet
    go_term.parquet
    go_obsolete.parquet
    go_synonym.parquet
    go_bp_parents.parquet   go_bp_offspring.parquet
    go_mf_parents.parquet   go_mf_offspring.parquet
    go_cc_parents.parquet   go_cc_offspring.parquet
    map_counts.parquet

Each file is written with Snappy compression.  All foreign-key relationships
use go_id (string) as the join key throughout -- no surrogate integers.

Querying
--------
Load files with pyarrow.parquet and filter with pyarrow.compute, e.g.:

    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    go_term = pq.read_table("output_dir/go_term.parquet")
    mask    = pc.equal(go_term.column("go_id"), "GO:0007005")
    row     = go_term.filter(mask)
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Reuse all parsing logic from the SQLite builder.
from .build_db import NS_MAP, parse_obo, parse_synonym_tag, transitive_closure


# ---------------------------------------------------------------------------
# Arrow schemas  (string nullable by default in Arrow; non-nullable enforced
# at write time by the builder rather than the schema itself)
# ---------------------------------------------------------------------------

GO_ONTOLOGY_SCHEMA = pa.schema([
    pa.field("ontology",  pa.string(), nullable=False),
    pa.field("term_type", pa.string(), nullable=False),
])

GO_TERM_SCHEMA = pa.schema([
    pa.field("go_id",      pa.string(), nullable=False),
    pa.field("term",       pa.string(), nullable=False),
    pa.field("ontology",   pa.string(), nullable=False),
    pa.field("definition", pa.string(), nullable=True),   # some terms have no def
])

GO_SYNONYM_SCHEMA = pa.schema([
    pa.field("go_id",      pa.string(), nullable=False),
    pa.field("synonym",    pa.string(), nullable=False),
    pa.field("secondary",  pa.string(), nullable=True),   # only set for alt_ids
    pa.field("scope",      pa.string(), nullable=True),
    pa.field("like_go_id", pa.int8(),   nullable=False),
])

PARENTS_SCHEMA = pa.schema([
    pa.field("go_id",             pa.string(), nullable=False),
    pa.field("parent_id",         pa.string(), nullable=False),
    pa.field("relationship_type", pa.string(), nullable=False),
])

OFFSPRING_SCHEMA = pa.schema([
    pa.field("go_id",        pa.string(), nullable=False),
    pa.field("offspring_id", pa.string(), nullable=False),
])

MAP_COUNTS_SCHEMA = pa.schema([
    pa.field("map_name", pa.string(), nullable=False),
    pa.field("count",    pa.int64(),  nullable=False),
])

# Ordered list used by load_tables() and tests.
TABLE_NAMES = [
    "go_ontology",
    "go_term",
    "go_obsolete",
    "go_synonym",
    "go_bp_parents",  "go_bp_offspring",
    "go_mf_parents",  "go_mf_offspring",
    "go_cc_parents",  "go_cc_offspring",
    "map_counts",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_def(raw):
    """Remove the citation block from a def: tag value."""
    if not raw:
        return None
    return re.sub(r'"\s*\[.*?\]\s*$', "", raw).lstrip('"').strip() or None


def _write(table, path):
    pq.write_table(table, path, compression="snappy")
    print(f"  {path.name}: {table.num_rows:,} rows")


def load_tables(out_dir):
    """
    Read every Parquet file produced by build() and return a dict keyed by
    table name (without the .parquet suffix).
    """
    out = Path(out_dir)
    return {name: pq.read_table(out / f"{name}.parquet") for name in TABLE_NAMES}


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(obo_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Parse OBO ---------------------------------------------------------
    terms = []
    obsolete = []
    for stanza in parse_obo(obo_path):
        go_id = stanza.get("id", "")
        if not go_id.startswith("GO:"):
            continue
        (obsolete if stanza.get("is_obsolete") == "true" else terms).append(stanza)

    print(f"Active terms  : {len(terms):>7,}")
    print(f"Obsolete terms: {len(obsolete):>7,}")

    active_ids = {t["id"] for t in terms}

    # ---- go_ontology -------------------------------------------------------
    _write(
        pa.table(
            {
                "ontology":  [short for _, (short, _) in NS_MAP.items()],
                "term_type": [long_ for _, (_, long_) in NS_MAP.items()],
            },
            schema=GO_ONTOLOGY_SCHEMA,
        ),
        out / "go_ontology.parquet",
    )

    # ---- go_term / go_obsolete ---------------------------------------------
    def _term_table(src, schema):
        ids, names, onts, defs = [], [], [], []
        for t in src:
            short, _ = NS_MAP.get(t.get("namespace", ""), ("??", "??"))
            ids.append(t["id"])
            names.append(t.get("name", ""))
            onts.append(short)
            defs.append(_strip_def(t.get("def")))
        return pa.table(
            {"go_id": ids, "term": names, "ontology": onts, "definition": defs},
            schema=schema,
        )

    _write(_term_table(terms,    GO_TERM_SCHEMA), out / "go_term.parquet")
    _write(_term_table(obsolete, GO_TERM_SCHEMA), out / "go_obsolete.parquet")

    # ---- go_synonym --------------------------------------------------------
    s_go, s_syn, s_sec, s_scope, s_lgid = [], [], [], [], []
    for t in terms:
        go_id = t["id"]
        for raw in t.get("synonym", []):
            label, scope, _ = parse_synonym_tag(raw)
            s_go.append(go_id);  s_syn.append(label)
            s_sec.append(None);  s_scope.append(scope);  s_lgid.append(0)
        for alt in t.get("alt_id", []):
            s_go.append(go_id);  s_syn.append(alt)
            s_sec.append(alt);   s_scope.append("EXACT"); s_lgid.append(1)

    _write(
        pa.table(
            {
                "go_id":      s_go,
                "synonym":    s_syn,
                "secondary":  s_sec,
                "scope":      s_scope,
                "like_go_id": pa.array(s_lgid, type=pa.int8()),
            },
            schema=GO_SYNONYM_SCHEMA,
        ),
        out / "go_synonym.parquet",
    )

    # ---- per-namespace parent and offspring tables -------------------------
    ns_map_local = [
        ("BP", "biological_process"),
        ("MF", "molecular_function"),
        ("CC", "cellular_component"),
    ]

    for ns_short, ns_key in ns_map_local:
        p_go, p_par, p_rel = [], [], []
        direct_children = defaultdict(list)

        for t in terms:
            if t.get("namespace") != ns_key:
                continue
            go_id = t["id"]
            for raw in t.get("is_a", []):
                parent_id = raw.split()[0]
                if parent_id in active_ids:
                    p_go.append(go_id); p_par.append(parent_id); p_rel.append("is_a")
                    direct_children[parent_id].append(go_id)
            for raw in t.get("relationship", []):
                parts = raw.split()
                if len(parts) >= 2:
                    rel_type, parent_id = parts[0], parts[1]
                    if parent_id in active_ids:
                        p_go.append(go_id); p_par.append(parent_id); p_rel.append(rel_type)
                        direct_children[parent_id].append(go_id)

        prefix = ns_short.lower()
        _write(
            pa.table(
                {"go_id": p_go, "parent_id": p_par, "relationship_type": p_rel},
                schema=PARENTS_SCHEMA,
            ),
            out / f"go_{prefix}_parents.parquet",
        )

        pairs  = transitive_closure(direct_children)
        o_anc  = [a for a, _ in pairs]
        o_desc = [d for _, d in pairs]
        _write(
            pa.table({"go_id": o_anc, "offspring_id": o_desc}, schema=OFFSPRING_SCHEMA),
            out / f"go_{prefix}_offspring.parquet",
        )

    # ---- map_counts --------------------------------------------------------
    map_names  = ["go_term", "go_obsolete", "go_synonym"]
    map_counts = [len(terms), len(obsolete), len(s_go)]
    _write(
        pa.table(
            {"map_name": map_names, "count": pa.array(map_counts, type=pa.int64())},
            schema=MAP_COUNTS_SCHEMA,
        ),
        out / "map_counts.parquet",
    )

    print(f"\nParquet files written to: {out}/")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m go_ontology.build_parquet <go.obo> <output_dir>")
    build(sys.argv[1], sys.argv[2])


def main():
    import sys
    if len(sys.argv) != 3:
        sys.exit("Usage: go-build-parquet <go.obo> <output_dir>")
    build(sys.argv[1], sys.argv[2])
