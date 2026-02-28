# BiocGOprep/go-ontology-tools

This software was developed primarily by Claude, with prompting and
revisions by Vince Carey.  The purpose of this work is to factor the production
of Bioconductor's GO.db AnnotationPackage away from the BioconductorAnnotationPipeline
system.


Main objective: Parse the [Gene Ontology](http://geneontology.org/) OBO file and build either
a **SQLite** database or a directory of **Parquet** files, using `go_id` as
the natural primary/foreign key throughout — no surrogate integers.

Python tooling is used to help create this Bioconductor resource
at this time.  A benefit is direct usage of the [hypothesis](https://hypothesis.readthedocs.io/en/latest/)
property-based testing system, applied to (admittedly simple) graph operations used to build the database tables.


## Outputs

Both builders produce the same logical tables:

| Table | Contents |
|---|---|
| `go_term` | Active terms (go_id, name, ontology, definition) |
| `go_obsolete` | Obsolete terms |
| `go_synonym` | Text synonyms and alt_ids |
| `go_bp/mf/cc_parents` | Direct parent edges per namespace |
| `go_bp/mf/cc_offspring` | Transitive closure per namespace |
| `go_ontology` | Ontology code → term_type mapping |
| `map_counts` | Row counts for key tables |

## Requirements

- Python 3.10+
- `pyarrow >= 14.0` (Parquet builder only; not needed for SQLite)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# SQLite builder only:
pip install -e .

# SQLite + Parquet:
pip install -e ".[parquet]"
```

## Usage

Download the latest OBO file from https://current.geneontology.org/ontology/go.obo

```bash
# Build SQLite database
go-build-db go.obo go.sqlite3

# Build Parquet dataset
go-build-parquet go.obo parquet_out/
```

Or call the builders directly as modules:

```bash
python -m go_ontology.build_db      go.obo go.sqlite3
python -m go_ontology.build_parquet go.obo parquet_out/
```

## Querying the Parquet output

```python
import pyarrow.parquet as pq
import pyarrow.compute as pc

go_term  = pq.read_table("parquet_out/go_term.parquet")
mask     = pc.equal(go_term.column("go_id"), "GO:0007005")
row      = go_term.filter(mask)
```

## Running the tests

```bash
pip install -e ".[test]"
python -m pytest tests/ -v
```

To run the full-dataset integration tests, supply a path to `go.obo`:

```bash
python -m pytest tests/ -v --obo /path/to/go.obo
```

## License

Apache 2.0
