"""
build_db.py
--------------
Parse a GO OBO file and build a SQLite3 database using go_id as the
natural primary/foreign key throughout — no surrogate integer _id.

Usage:
    python -m go_ontology.build_db go.obo go.sqlite3

Schema highlights
-----------------
- go_term(go_id PK, ...)              active terms
- go_obsolete(go_id PK, ...)          obsolete terms
- go_synonym(go_id FK, ...)           synonyms + alt_ids
- go_bp/mf/cc_parents(go_id, parent_id, relationship_type)   direct edges
- go_bp/mf/cc_offspring(go_id, offspring_id)                 transitive closure
"""

import re
import sqlite3
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# OBO parser
# ---------------------------------------------------------------------------

def parse_obo(path: str):
    """
    Yield one dict per [Term] stanza. Keys mirror OBO tag names; multi-valued
    tags (synonym, is_a, relationship, alt_id, consider, replaced_by) are
    returned as lists.
    """
    multi = {"synonym", "is_a", "relationship", "alt_id",
             "consider", "replaced_by", "subset", "xref", "property_value"}

    with open(path, encoding="utf-8") as fh:
        stanza = None
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("[Term]"):
                if stanza is not None:
                    yield stanza
                stanza = defaultdict(list)
            elif line.startswith("[") and line.endswith("]"):
                # e.g. [Typedef] — flush current term stanza if any, skip
                if stanza is not None:
                    yield stanza
                stanza = None
            elif stanza is not None and ": " in line:
                tag, _, value = line.partition(": ")
                # strip trailing comments (! ...)
                value = re.sub(r"\s+!.*$", "", value).strip()
                if tag in multi:
                    stanza[tag].append(value)
                else:
                    stanza[tag] = value
        if stanza is not None:
            yield stanza


# ---------------------------------------------------------------------------
# Synonym / alt_id helpers
# ---------------------------------------------------------------------------

_SYNONYM_RE = re.compile(
    r'^"(.*?)"\s+(EXACT|RELATED|NARROW|BROAD|SYNONYM)?\s*(?:\[([^\]]*)\])?'
)

def parse_synonym_tag(raw: str):
    """Return (label, scope, like_go_id=0)."""
    m = _SYNONYM_RE.match(raw)
    if not m:
        return raw, "EXACT", 0
    label = m.group(1)
    scope = m.group(2) or "EXACT"
    return label, scope, 0


# ---------------------------------------------------------------------------
# Transitive closure
# ---------------------------------------------------------------------------

def transitive_closure(direct_edges: dict[str, list[str]]) -> list[tuple[str, str]]:
    """
    Given {parent: [child, ...]} direct edges, return all (ancestor, descendant)
    pairs as the transitive closure.

    The offspring tables store (go_id=ancestor, offspring_id=descendant).
    """
    # Build child→parents map (we need to traverse upward for each node)
    # Actually let's build parent→children and do BFS from each node downward.
    children: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set()

    for parent, ch_list in direct_edges.items():
        all_nodes.add(parent)
        for ch in ch_list:
            children[parent].add(ch)
            all_nodes.add(ch)

    pairs: list[tuple[str, str]] = []
    for ancestor in all_nodes:
        # BFS over all descendants
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


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS metadata (
    name  TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS go_ontology (
    ontology  TEXT PRIMARY KEY,
    term_type TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS go_term (
    go_id      CHAR(10) PRIMARY KEY,
    term       TEXT NOT NULL,
    ontology   TEXT NOT NULL REFERENCES go_ontology(ontology),
    definition TEXT
);
CREATE INDEX IF NOT EXISTS idx_go_term_ontology ON go_term(ontology);

CREATE TABLE IF NOT EXISTS go_obsolete (
    go_id      CHAR(10) PRIMARY KEY,
    term       TEXT NOT NULL,
    ontology   TEXT NOT NULL REFERENCES go_ontology(ontology),
    definition TEXT
);
CREATE INDEX IF NOT EXISTS idx_go_obsolete_ontology ON go_obsolete(ontology);

CREATE TABLE IF NOT EXISTS go_synonym (
    go_id      CHAR(10) NOT NULL REFERENCES go_term(go_id),
    synonym    TEXT NOT NULL,
    secondary  CHAR(10),          -- populated when synonym is an alt_id
    scope      TEXT,              -- EXACT | RELATED | NARROW | BROAD
    like_go_id SMALLINT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_go_synonym_go_id  ON go_synonym(go_id);
CREATE INDEX IF NOT EXISTS idx_go_synonym_syn    ON go_synonym(synonym);

-- ---- BP parents / offspring ------------------------------------------------

CREATE TABLE IF NOT EXISTS go_bp_parents (
    go_id             CHAR(10) NOT NULL REFERENCES go_term(go_id),
    parent_id         CHAR(10) NOT NULL REFERENCES go_term(go_id),
    relationship_type TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bp_parents_child  ON go_bp_parents(go_id);
CREATE INDEX IF NOT EXISTS idx_bp_parents_parent ON go_bp_parents(parent_id);

CREATE TABLE IF NOT EXISTS go_bp_offspring (
    go_id        CHAR(10) NOT NULL REFERENCES go_term(go_id),
    offspring_id CHAR(10) NOT NULL REFERENCES go_term(go_id)
);
CREATE INDEX IF NOT EXISTS idx_bp_offspring_anc  ON go_bp_offspring(go_id);
CREATE INDEX IF NOT EXISTS idx_bp_offspring_desc ON go_bp_offspring(offspring_id);

-- ---- MF parents / offspring ------------------------------------------------

CREATE TABLE IF NOT EXISTS go_mf_parents (
    go_id             CHAR(10) NOT NULL REFERENCES go_term(go_id),
    parent_id         CHAR(10) NOT NULL REFERENCES go_term(go_id),
    relationship_type TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mf_parents_child  ON go_mf_parents(go_id);
CREATE INDEX IF NOT EXISTS idx_mf_parents_parent ON go_mf_parents(parent_id);

CREATE TABLE IF NOT EXISTS go_mf_offspring (
    go_id        CHAR(10) NOT NULL REFERENCES go_term(go_id),
    offspring_id CHAR(10) NOT NULL REFERENCES go_term(go_id)
);
CREATE INDEX IF NOT EXISTS idx_mf_offspring_anc  ON go_mf_offspring(go_id);
CREATE INDEX IF NOT EXISTS idx_mf_offspring_desc ON go_mf_offspring(offspring_id);

-- ---- CC parents / offspring ------------------------------------------------

CREATE TABLE IF NOT EXISTS go_cc_parents (
    go_id             CHAR(10) NOT NULL REFERENCES go_term(go_id),
    parent_id         CHAR(10) NOT NULL REFERENCES go_term(go_id),
    relationship_type TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cc_parents_child  ON go_cc_parents(go_id);
CREATE INDEX IF NOT EXISTS idx_cc_parents_parent ON go_cc_parents(parent_id);

CREATE TABLE IF NOT EXISTS go_cc_offspring (
    go_id        CHAR(10) NOT NULL REFERENCES go_term(go_id),
    offspring_id CHAR(10) NOT NULL REFERENCES go_term(go_id)
);
CREATE INDEX IF NOT EXISTS idx_cc_offspring_anc  ON go_cc_offspring(go_id);
CREATE INDEX IF NOT EXISTS idx_cc_offspring_desc ON go_cc_offspring(offspring_id);

CREATE TABLE IF NOT EXISTS map_metadata (
    map_name    TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    source_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS map_counts (
    map_name TEXT PRIMARY KEY,
    count    INTEGER NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Namespace → short label
# ---------------------------------------------------------------------------

NS_MAP = {
    "biological_process": ("BP", "biological process"),
    "molecular_function": ("MF", "molecular function"),
    "cellular_component": ("CC", "cellular component"),
}


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(obo_path: str, db_path: str):
    terms: list[dict] = []
    obsolete: list[dict] = []

    for stanza in parse_obo(obo_path):
        go_id = stanza.get("id", "")
        if not go_id.startswith("GO:"):
            continue
        if stanza.get("is_obsolete") == "true":
            obsolete.append(stanza)
        else:
            terms.append(stanza)

    print(f"Active terms  : {len(terms):>7,}")
    print(f"Obsolete terms: {len(obsolete):>7,}")

    # --- build a set of active go_ids for FK validation ---
    active_ids: set[str] = {t["id"] for t in terms}

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = OFF")   # load order flexibility
    con.execute("PRAGMA journal_mode = WAL")
    con.executescript(DDL)

    # ---- go_ontology -------------------------------------------------------
    con.executemany(
        "INSERT OR IGNORE INTO go_ontology VALUES (?,?)",
        [(short, long_) for _, (short, long_) in NS_MAP.items()]
    )

    # ---- go_term -----------------------------------------------------------
    term_rows = []
    for t in terms:
        ns = t.get("namespace", "")
        short, _ = NS_MAP.get(ns, ("??", "??"))
        defn = t.get("def", None)
        if defn:
            # strip leading/trailing quotes and citation block
            defn = re.sub(r'"\s*\[.*?\]\s*$', '', defn).lstrip('"').strip()
        term_rows.append((t["id"], t.get("name", ""), short, defn))

    con.executemany(
        "INSERT OR IGNORE INTO go_term(go_id, term, ontology, definition) VALUES (?,?,?,?)",
        term_rows,
    )

    # ---- go_obsolete -------------------------------------------------------
    obs_rows = []
    for t in obsolete:
        ns = t.get("namespace", "")
        short, _ = NS_MAP.get(ns, ("??", "??"))
        defn = t.get("def", None)
        if defn:
            defn = re.sub(r'"\s*\[.*?\]\s*$', '', defn).lstrip('"').strip()
        obs_rows.append((t["id"], t.get("name", ""), short, defn))

    con.executemany(
        "INSERT OR IGNORE INTO go_obsolete(go_id, term, ontology, definition) VALUES (?,?,?,?)",
        obs_rows,
    )

    # ---- go_synonym (text synonyms + alt_ids) ------------------------------
    syn_rows = []
    for t in terms:
        go_id = t["id"]

        # text synonyms
        for raw in t.get("synonym", []):
            label, scope, _ = parse_synonym_tag(raw)
            syn_rows.append((go_id, label, None, scope, 0))

        # alt_ids: these ARE go ids, so like_go_id=1 and secondary is set
        for alt in t.get("alt_id", []):
            syn_rows.append((go_id, alt, alt, "EXACT", 1))

    con.executemany(
        "INSERT INTO go_synonym(go_id, synonym, secondary, scope, like_go_id) VALUES (?,?,?,?,?)",
        syn_rows,
    )

    # ---- parent tables + collect edges for offspring -----------------------
    # direct_children[ns][parent_go_id] = [child_go_id, ...]
    direct_children: dict[str, dict[str, list[str]]] = {
        "BP": defaultdict(list),
        "MF": defaultdict(list),
        "CC": defaultdict(list),
    }

    parent_rows: dict[str, list[tuple]] = {"BP": [], "MF": [], "CC": []}

    for t in terms:
        go_id = t["id"]
        ns = t.get("namespace", "")
        short, _ = NS_MAP.get(ns, ("??", "??"))
        if short not in parent_rows:
            continue

        # is_a relationships
        for raw in t.get("is_a", []):
            parent_id = raw.split()[0]
            if parent_id in active_ids:
                parent_rows[short].append((go_id, parent_id, "is_a"))
                direct_children[short][parent_id].append(go_id)

        # other relationships (part_of, regulates, etc.)
        for raw in t.get("relationship", []):
            parts = raw.split()
            if len(parts) >= 2:
                rel_type, parent_id = parts[0], parts[1]
                if parent_id in active_ids:
                    parent_rows[short].append((go_id, parent_id, rel_type))
                    direct_children[short][parent_id].append(go_id)

    ns_tables = {
        "BP": ("go_bp_parents", "go_bp_offspring"),
        "MF": ("go_mf_parents", "go_mf_offspring"),
        "CC": ("go_cc_parents", "go_cc_offspring"),
    }

    for ns, (p_table, o_table) in ns_tables.items():
        con.executemany(
            f"INSERT INTO {p_table}(go_id, parent_id, relationship_type) VALUES (?,?,?)",
            parent_rows[ns],
        )
        print(f"  {p_table}: {len(parent_rows[ns]):,} rows")

        offspring = transitive_closure(direct_children[ns])
        con.executemany(
            f"INSERT INTO {o_table}(go_id, offspring_id) VALUES (?,?)",
            offspring,
        )
        print(f"  {o_table}: {len(offspring):,} rows")

    # ---- map_counts --------------------------------------------------------
    counts = [
        ("go_term",     len(terms)),
        ("go_obsolete", len(obsolete)),
        ("go_synonym",  len(syn_rows)),
    ]
    con.executemany("INSERT OR REPLACE INTO map_counts VALUES (?,?)", counts)

    con.commit()
    con.execute("PRAGMA foreign_keys = ON")

    # quick sanity check
    cur = con.execute("SELECT COUNT(*) FROM go_term")
    print(f"\ngo_term rows in DB: {cur.fetchone()[0]:,}")
    cur = con.execute("SELECT COUNT(*) FROM go_obsolete")
    print(f"go_obsolete rows  : {cur.fetchone()[0]:,}")

    con.close()
    print(f"\nDatabase written to: {db_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python -m go_ontology.build_db <go.obo> <output.sqlite3>")
    build(sys.argv[1], sys.argv[2])


def main():
    import sys
    if len(sys.argv) != 3:
        sys.exit("Usage: go-build-db <go.obo> <output.sqlite3>")
    build(sys.argv[1], sys.argv[2])
