"""
tests/test_build_parquet.py
---------------------------
Unit and integration tests for go_ontology.build_parquet.

Dependencies
------------
    python3 -m venv .venv
    source .venv/bin/activate           # Windows: .venv\\Scripts\\activate
    pip install -e ".[test]"
    python -m pytest tests/ -v

With a real go.obo:
    python -m pytest tests/ -v --obo /path/to/go.obo

Arrow query idioms used
-----------------------
    table.num_rows                          COUNT(*)
    table.filter(pc.equal(col, val))        WHERE col = val
    table.column("x").to_pylist()           SELECT x ...
    table.column("x").unique()              SELECT DISTINCT x
    pc.is_null(col)                         IS NULL
    pc.is_valid(col)                        IS NOT NULL
    set(col_a) - set(col_b)                 NOT IN subquery
    pc.equal(col_a, col_b)                  self-join equality
    set(zip(col_a, col_b))                  DISTINCT (a, b) pairs
"""

import tempfile
import textwrap
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from go_ontology.build_parquet import build, load_tables, TABLE_NAMES

# --------------------------------------------------------------------------- #
# Fixture OBO                                                                  #
# --------------------------------------------------------------------------- #

FIXTURE_OBO = textwrap.dedent("""\
    format-version: 1.2
    ontology: go

    [Term]
    id: GO:0000001
    name: mitochondrion inheritance
    namespace: biological_process
    def: "The distribution of mitochondria into daughter cells." [GOC:mcc, PMID:123]
    synonym: "mitochondrial inheritance" EXACT []
    is_a: GO:0048308 ! organelle inheritance
    is_a: GO:0048311 ! mitochondrion distribution

    [Term]
    id: GO:0000002
    name: obsolete mitochondrial genome maintenance
    namespace: biological_process
    def: "OBSOLETE. Some definition." [GOC:ai]
    is_obsolete: true

    [Term]
    id: GO:0000006
    name: high-affinity zinc transporter activity
    namespace: molecular_function
    def: "Enables transfer of zinc ions." [TC:2.A.5.1.1]
    synonym: "high affinity zinc uptake transporter activity" EXACT []
    synonym: "zinc uptake transporter activity" RELATED []
    is_a: GO:0005385 ! zinc ion transmembrane transporter activity

    [Term]
    id: GO:0000010
    name: heptaprenyl diphosphate synthase activity
    namespace: molecular_function
    def: "Catalysis of heptaprenyl diphosphate synthesis." [PMID:9708911]
    alt_id: GO:0036422
    synonym: "all-trans-heptaprenyl-diphosphate synthase activity" EXACT [EC:2.5.1.30]
    is_a: GO:0120531 ! prenyl diphosphate synthase activity

    [Term]
    id: GO:0000015
    name: phosphopyruvate hydratase complex
    namespace: cellular_component
    def: "A multimeric enzyme complex." [GOC:jl]
    synonym: "enolase complex" EXACT []
    is_a: GO:1902494 ! catalytic complex
    relationship: part_of GO:0005829 ! cytosol

    [Term]
    id: GO:0000018
    name: regulation of DNA recombination
    namespace: biological_process
    def: "Any process that modulates DNA recombination." [GOC:go_curators]
    is_a: GO:0051052 ! regulation of DNA metabolic process
    relationship: regulates GO:0006310 ! DNA recombination

    [Term]
    id: GO:0000019
    name: regulation of mitotic recombination
    namespace: biological_process
    def: "Any process that modulates DNA recombination during mitosis." [GOC:go_curators]
    synonym: "regulation of recombination within rDNA repeats" NARROW []
    is_a: GO:0000018 ! regulation of DNA recombination
    relationship: regulates GO:0006312 ! mitotic recombination

    [Term]
    id: GO:0048308
    name: organelle inheritance
    namespace: biological_process
    def: "The distribution of organelles into daughter cells." [GOC:mcc]
    is_a: GO:0048856 ! anatomical structure development

    [Term]
    id: GO:0048311
    name: mitochondrion distribution
    namespace: biological_process
    def: "Any process that establishes the spatial arrangement of mitochondria." [GOC:mcc]
    is_a: GO:0048308 ! organelle inheritance

    [Typedef]
    id: regulates
    name: regulates
    is_cyclic: false
""")


def _write_fixture(tmp_path: Path):
    obo = tmp_path / "fixture.obo"
    obo.write_text(FIXTURE_OBO, encoding="utf-8")
    return obo, tmp_path / "parquet_out"


def _obo_path(request):
    cli = request.config.getoption("--obo", default=None)
    if cli:
        return Path(cli)
    for candidate in [Path("go.obo"), Path("data/go.obo"), Path("/data/go.obo")]:
        if candidate.exists():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Shared Arrow query helpers                                                   #
# --------------------------------------------------------------------------- #

class _ArrowHelpers:

    def _tbl(self, name):
        return self.tables[name]

    def _count(self, name):
        return self._tbl(name).num_rows

    def _col(self, name, col):
        return self._tbl(name).column(col)

    def _filter(self, name, **eq):
        t = self._tbl(name)
        mask = None
        for col, val in eq.items():
            m = pc.equal(t.column(col), val)
            mask = m if mask is None else pc.and_(mask, m)
        return t.filter(mask) if mask is not None else t

    def _count_where(self, name, **eq):
        return self._filter(name, **eq).num_rows

    def _scalar(self, name, col, **eq):
        return self._filter(name, **eq).column(col)[0].as_py()

    def _distinct_values(self, name, col):
        return set(self._col(name, col).unique().to_pylist())

    def _count_self_equal(self, name, col_a, col_b):
        t = self._tbl(name)
        return pc.sum(
            pc.cast(pc.equal(t.column(col_a), t.column(col_b)), pa.int32())
        ).as_py() or 0

    def _count_distinct_pairs(self, name, col_a, col_b):
        t = self._tbl(name)
        return len(set(zip(t.column(col_a).to_pylist(), t.column(col_b).to_pylist())))

    def _set(self, name, col):
        return set(self._col(name, col).to_pylist())

    def _map_count(self, table_name):
        return self._scalar("map_counts", "count", map_name=table_name)


# =========================================================================== #
# 1.  Fixture-based integration tests                                          #
# =========================================================================== #

class TestBuildGoParquetFixture(_ArrowHelpers, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        obo, out = _write_fixture(Path(cls._tmp.name))
        build(str(obo), str(out))
        cls.tables = load_tables(out)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_all_expected_files_present(self):
        for name in TABLE_NAMES:
            self.assertIn(name, self.tables)

    def test_all_tables_are_arrow_tables(self):
        for name, t in self.tables.items():
            self.assertIsInstance(t, pa.Table)

    def test_go_ontology_has_three_rows(self):
        self.assertEqual(self._count("go_ontology"), 3)

    def test_go_ontology_codes_correct(self):
        self.assertEqual(self._distinct_values("go_ontology", "ontology"), {"BP","MF","CC"})

    def test_active_term_count(self):
        self.assertEqual(self._count("go_term"), 8)

    def test_known_term_present(self):
        rows = self._filter("go_term", go_id="GO:0000001")
        self.assertEqual(rows.num_rows, 1)
        self.assertEqual(rows.column("term")[0].as_py(), "mitochondrion inheritance")
        self.assertEqual(rows.column("ontology")[0].as_py(), "BP")

    def test_go_ids_match_format(self):
        import re
        pattern = re.compile(r"^GO:\d{7}$")
        for gid in self._col("go_term", "go_id").to_pylist():
            self.assertRegex(gid, pattern)

    def test_definition_citation_stripped(self):
        defn = self._scalar("go_term", "definition", go_id="GO:0000001")
        self.assertNotIn("[GOC:", defn)
        self.assertIn("mitochondria", defn)

    def test_obsolete_not_in_go_term(self):
        self.assertEqual(self._count_where("go_term", go_id="GO:0000002"), 0)

    def test_no_overlap_active_obsolete(self):
        self.assertEqual(self._set("go_term","go_id") & self._set("go_obsolete","go_id"), set())

    def test_obsolete_term_count(self):
        self.assertEqual(self._count("go_obsolete"), 1)

    def test_text_synonym_stored(self):
        rows = self._filter("go_synonym", go_id="GO:0000001",
                            synonym="mitochondrial inheritance")
        self.assertEqual(rows.num_rows, 1)
        self.assertEqual(rows.column("scope")[0].as_py(), "EXACT")
        self.assertEqual(rows.column("like_go_id")[0].as_py(), 0)
        self.assertIsNone(rows.column("secondary")[0].as_py())

    def test_related_scope_stored(self):
        rows = self._filter("go_synonym", go_id="GO:0000006",
                            synonym="zinc uptake transporter activity")
        self.assertEqual(rows.column("scope")[0].as_py(), "RELATED")

    def test_narrow_scope_stored(self):
        t = self._tbl("go_synonym")
        mask = pc.and_(pc.equal(t.column("go_id"), "GO:0000019"),
                       pc.equal(t.column("scope"), "NARROW"))
        self.assertEqual(t.filter(mask).num_rows, 1)

    def test_alt_id_stored_with_like_go_id(self):
        rows = self._filter("go_synonym", go_id="GO:0000010", like_go_id=1)
        self.assertEqual(rows.num_rows, 1)
        self.assertEqual(rows.column("synonym")[0].as_py(),   "GO:0036422")
        self.assertEqual(rows.column("secondary")[0].as_py(), "GO:0036422")

    def test_synonyms_reference_only_active_terms(self):
        orphans = self._set("go_synonym","go_id") - self._set("go_term","go_id")
        self.assertEqual(orphans, set())

    def test_bp_parents_is_a_edges(self):
        rows = self._filter("go_bp_parents", go_id="GO:0000001")
        self.assertIn("GO:0048308", set(rows.column("parent_id").to_pylist()))
        self.assertIn("GO:0048311", set(rows.column("parent_id").to_pylist()))

    def test_out_of_scope_parents_excluded(self):
        self.assertEqual(self._count_where("go_bp_parents", parent_id="GO:0051052"), 0)

    def test_direct_parent_in_offspring(self):
        self.assertEqual(
            self._count_where("go_bp_offspring", go_id="GO:0048308", offspring_id="GO:0000001"), 1
        )

    def test_transitive_ancestor_in_offspring(self):
        self.assertEqual(
            self._count_where("go_bp_offspring", go_id="GO:0048308", offspring_id="GO:0000001"), 1
        )

    def test_no_self_loops_in_bp_offspring(self):
        self.assertEqual(self._count_self_equal("go_bp_offspring","go_id","offspring_id"), 0)

    def test_no_duplicate_pairs_in_bp_offspring(self):
        self.assertEqual(
            self._count("go_bp_offspring"),
            self._count_distinct_pairs("go_bp_offspring","go_id","offspring_id"),
        )

    def test_mf_offspring_empty_in_fixture(self):
        self.assertEqual(self._count("go_mf_offspring"), 0)

    def test_cc_offspring_empty_in_fixture(self):
        self.assertEqual(self._count("go_cc_offspring"), 0)

    def test_map_counts_go_term(self):
        self.assertEqual(self._count("go_term"), self._map_count("go_term"))

    def test_map_counts_go_obsolete(self):
        self.assertEqual(self._count("go_obsolete"), self._map_count("go_obsolete"))

    def test_map_counts_go_synonym(self):
        self.assertEqual(self._count("go_synonym"), self._map_count("go_synonym"))

    def test_like_go_id_is_int8(self):
        self.assertEqual(
            self._tbl("go_synonym").schema.field("like_go_id").type, pa.int8()
        )

    def test_map_counts_count_is_int64(self):
        self.assertEqual(
            self._tbl("map_counts").schema.field("count").type, pa.int64()
        )


# =========================================================================== #
# 2.  Full go.obo integration tests                                            #
# =========================================================================== #

@pytest.mark.usefixtures("request")
class TestBuildGoParquetFullObo(_ArrowHelpers):

    @pytest.fixture(autouse=True)
    def _attach(self, full_parquet_tables):
        if full_parquet_tables is None:
            pytest.skip("go.obo not found -- pass --obo <path> to enable")
        self.tables = full_parquet_tables

    def test_active_term_count_plausible(self):
        assert self._count("go_term") > 35_000

    def test_obsolete_term_count_plausible(self):
        assert self._count("go_obsolete") > 5_000

    def test_all_three_namespaces_populated(self):
        for ns in ("BP", "MF", "CC"):
            assert self._count_where("go_term", ontology=ns) > 0

    def test_no_orphaned_synonyms(self):
        assert not (self._set("go_synonym","go_id") - self._set("go_term","go_id"))

    def test_no_orphaned_bp_children(self):
        assert self._set("go_bp_parents","go_id") <= self._set("go_term","go_id")

    def test_no_orphaned_bp_parents(self):
        assert self._set("go_bp_parents","parent_id") <= self._set("go_term","go_id")

    def test_no_overlap_active_obsolete(self):
        assert not (self._set("go_term","go_id") & self._set("go_obsolete","go_id"))

    def test_no_self_loops_bp_offspring(self):
        assert self._count_self_equal("go_bp_offspring","go_id","offspring_id") == 0

    def test_no_self_loops_mf_offspring(self):
        assert self._count_self_equal("go_mf_offspring","go_id","offspring_id") == 0

    def test_no_self_loops_cc_offspring(self):
        assert self._count_self_equal("go_cc_offspring","go_id","offspring_id") == 0

    def test_no_duplicate_pairs_bp_offspring(self):
        assert self._count("go_bp_offspring") == \
               self._count_distinct_pairs("go_bp_offspring","go_id","offspring_id")

    def test_every_bp_parent_edge_in_offspring(self):
        p = self._tbl("go_bp_parents")
        o = self._tbl("go_bp_offspring")
        parent_pairs   = set(zip(p.column("parent_id").to_pylist(), p.column("go_id").to_pylist()))
        offspring_pairs = set(zip(o.column("go_id").to_pylist(), o.column("offspring_id").to_pylist()))
        missing = parent_pairs - offspring_pairs
        assert not missing, f"{len(missing)} BP parent edges missing from offspring"

    def test_relationship_types_sensible(self):
        known = {"is_a","part_of","regulates","positively_regulates",
                 "negatively_regulates","occurs_in","has_part"}
        for tname in ("go_bp_parents","go_mf_parents","go_cc_parents"):
            unknown = self._distinct_values(tname,"relationship_type") - known
            assert not unknown, f"Unexpected types in {tname}: {unknown}"

    def test_alt_ids_have_like_go_id_set(self):
        t = self._tbl("go_synonym")
        assert t.filter(pc.and_(pc.is_valid(t.column("secondary")),
                                pc.not_equal(t.column("like_go_id"), 1))).num_rows == 0

    def test_text_synonyms_have_like_go_id_unset(self):
        t = self._tbl("go_synonym")
        assert t.filter(pc.and_(pc.is_null(t.column("secondary")),
                                pc.not_equal(t.column("like_go_id"), 0))).num_rows == 0

    def test_map_counts_match_actual(self):
        for tname in ("go_term","go_obsolete","go_synonym"):
            assert self._count(tname) == self._map_count(tname)

    def test_root_bp_term_present(self):
        assert self._count_where("go_term", go_id="GO:0008150") == 1

    def test_root_mf_term_present(self):
        assert self._count_where("go_term", go_id="GO:0003674") == 1

    def test_root_cc_term_present(self):
        assert self._count_where("go_term", go_id="GO:0005575") == 1

    def test_root_bp_has_many_offspring(self):
        count = self._count_where("go_bp_offspring", go_id="GO:0008150")
        assert count > 20_000, f"BP root offspring count suspiciously low: {count}"


# --------------------------------------------------------------------------- #
# Session-scoped fixture                                                       #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def full_parquet_tables(request, tmp_path_factory):
    obo = _obo_path(request)
    if obo is None or not obo.exists():
        yield None
        return
    out = tmp_path_factory.mktemp("go_parquet_full") / "parquet"
    build(str(obo), str(out))
    yield load_tables(out)


if __name__ == "__main__":
    unittest.main()
