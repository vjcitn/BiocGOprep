"""
tests/test_build_db.py
----------------------
Unit and integration tests for go_ontology.build_db.

Dependencies
------------
    python3 -m venv .venv
    source .venv/bin/activate           # Windows: .venv\\Scripts\\activate
    pip install -e ".[test]"
    python -m pytest tests/ -v

With a real go.obo:
    python -m pytest tests/ -v --obo /path/to/go.obo

Test organisation
-----------------
TestParseSynonymTag     -- pure-function tests, no I/O
TestTransitiveClosure   -- pure-function tests, no I/O
TestParseObo            -- parser tests over synthetic OBO text written to tmp files
TestBuildFixture        -- integration tests against a small self-contained fixture OBO
TestBuildFullObo        -- whole-database sanity checks against a real go.obo
                           (skipped automatically when go.obo is not present)
"""

import os
import re
import sqlite3
import tempfile
import textwrap
import unittest
from collections import defaultdict
from pathlib import Path

import pytest

from go_ontology.build_db import (
    build,
    parse_obo,
    parse_synonym_tag,
    transitive_closure,
)

# --------------------------------------------------------------------------- #
# Shared fixture OBO text                                                      #
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


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    obo = tmp_path / "fixture.obo"
    obo.write_text(FIXTURE_OBO, encoding="utf-8")
    db = tmp_path / "fixture.sqlite3"
    return obo, db


def _open_db(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _obo_path(request):
    cli = request.config.getoption("--obo", default=None)
    if cli:
        return Path(cli)
    for candidate in [Path("go.obo"), Path("data/go.obo"), Path("/data/go.obo")]:
        if candidate.exists():
            return candidate
    return None


# =========================================================================== #
# 1.  parse_synonym_tag                                                        #
# =========================================================================== #

class TestParseSynonymTag(unittest.TestCase):

    def _call(self, raw):
        return parse_synonym_tag(raw)

    def test_exact_scope(self):
        label, scope, lgid = self._call('"mitochondrial inheritance" EXACT []')
        self.assertEqual(label, "mitochondrial inheritance")
        self.assertEqual(scope, "EXACT")
        self.assertEqual(lgid, 0)

    def test_related_scope(self):
        label, scope, _ = self._call('"zinc uptake transporter activity" RELATED []')
        self.assertEqual(scope, "RELATED")

    def test_narrow_scope(self):
        label, scope, _ = self._call('"50S subunit assembly" NARROW [GOC:mah]')
        self.assertEqual(label, "50S subunit assembly")
        self.assertEqual(scope, "NARROW")

    def test_broad_scope(self):
        _, scope, _ = self._call('"acyl-CoA or acyl binding" BROAD []')
        self.assertEqual(scope, "BROAD")

    def test_citation_in_brackets_preserved_in_label(self):
        label, scope, _ = self._call(
            '"all-trans-heptaprenyl-diphosphate synthase activity" EXACT [EC:2.5.1.30]'
        )
        self.assertNotIn("EC:", label)
        self.assertEqual(scope, "EXACT")

    def test_missing_scope_defaults_to_exact(self):
        label, scope, _ = self._call('"some synonym" []')
        self.assertEqual(scope, "EXACT")

    def test_label_with_internal_quotes(self):
        label, scope, _ = self._call('"alpha-(1->6) linkage" EXACT []')
        self.assertEqual(label, 'alpha-(1->6) linkage')

    def test_malformed_falls_back_gracefully(self):
        raw = "not a proper synonym"
        label, scope, _ = self._call(raw)
        self.assertEqual(label, raw)
        self.assertEqual(scope, "EXACT")


# =========================================================================== #
# 2.  transitive_closure                                                       #
# =========================================================================== #

class TestTransitiveClosure(unittest.TestCase):

    def _pairs(self, edges):
        return set(transitive_closure(edges))

    def test_empty_graph(self):
        self.assertEqual(self._pairs({}), set())

    def test_single_edge(self):
        self.assertIn(("A", "B"), self._pairs({"A": ["B"]}))

    def test_linear_chain(self):
        pairs = self._pairs({"A": ["B"], "B": ["C"]})
        self.assertIn(("A", "B"), pairs)
        self.assertIn(("A", "C"), pairs)
        self.assertIn(("B", "C"), pairs)
        self.assertNotIn(("C", "A"), pairs)

    def test_diamond(self):
        edges = {"A": ["B", "C"], "B": ["D"], "C": ["D"]}
        pairs = self._pairs(edges)
        for expected in [("A","B"),("A","C"),("A","D"),("B","D"),("C","D")]:
            self.assertIn(expected, pairs)

    def test_diamond_no_duplicates(self):
        edges = {"A": ["B", "C"], "B": ["D"], "C": ["D"]}
        counts = defaultdict(int)
        for p in transitive_closure(edges):
            counts[p] += 1
        self.assertEqual(counts[("A", "D")], 1)

    def test_no_self_loops(self):
        for anc, desc in transitive_closure({"A": ["B"], "B": ["C"]}):
            self.assertNotEqual(anc, desc)

    def test_disconnected_components(self):
        pairs = self._pairs({"A": ["B"], "X": ["Y"]})
        self.assertIn(("A", "B"), pairs)
        self.assertIn(("X", "Y"), pairs)
        self.assertNotIn(("A", "Y"), pairs)

    def test_deep_chain_length(self):
        edges = {"R": ["A"], "A": ["B"], "B": ["C"], "C": ["D"]}
        pairs = self._pairs(edges)
        self.assertIn(("R", "D"), pairs)
        self.assertIn(("A", "D"), pairs)

    def test_multiple_children(self):
        pairs = self._pairs({"A": ["B", "C", "D"]})
        for child in ["B", "C", "D"]:
            self.assertIn(("A", child), pairs)
        self.assertNotIn(("B", "C"), pairs)


# =========================================================================== #
# 3.  parse_obo                                                                #
# =========================================================================== #

class TestParseObo(unittest.TestCase):

    def _parse(self, text: str):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".obo", encoding="utf-8", delete=False
        ) as f:
            f.write(text)
            name = f.name
        try:
            return list(parse_obo(name))
        finally:
            os.unlink(name)

    def test_single_term_round_trip(self):
        stanzas = self._parse(textwrap.dedent("""\
            [Term]
            id: GO:0000001
            name: mitochondrion inheritance
            namespace: biological_process
        """))
        self.assertEqual(len(stanzas), 1)
        t = stanzas[0]
        self.assertEqual(t["id"], "GO:0000001")
        self.assertEqual(t["name"], "mitochondrion inheritance")

    def test_multi_valued_tags_are_lists(self):
        stanzas = self._parse(textwrap.dedent("""\
            [Term]
            id: GO:0000001
            name: test
            namespace: biological_process
            is_a: GO:0048308 ! organelle inheritance
            is_a: GO:0048311 ! mitochondrion distribution
            synonym: "syn1" EXACT []
            synonym: "syn2" RELATED []
        """))
        self.assertIsInstance(stanzas[0]["is_a"], list)
        self.assertEqual(len(stanzas[0]["is_a"]), 2)
        self.assertEqual(len(stanzas[0]["synonym"]), 2)

    def test_inline_comment_stripped(self):
        stanzas = self._parse(textwrap.dedent("""\
            [Term]
            id: GO:0000001
            name: test
            namespace: biological_process
            is_a: GO:0048308 ! organelle inheritance
        """))
        self.assertEqual(stanzas[0]["is_a"][0], "GO:0048308")

    def test_obsolete_flag_captured(self):
        stanzas = self._parse(textwrap.dedent("""\
            [Term]
            id: GO:0000002
            name: obsolete term
            namespace: biological_process
            is_obsolete: true
        """))
        self.assertEqual(stanzas[0]["is_obsolete"], "true")

    def test_typedef_stanza_excluded(self):
        stanzas = self._parse(textwrap.dedent("""\
            [Term]
            id: GO:0000001
            name: real term
            namespace: biological_process

            [Typedef]
            id: regulates
            name: regulates
        """))
        self.assertEqual(len(stanzas), 1)

    def test_multiple_terms_all_yielded(self):
        stanzas = self._parse(FIXTURE_OBO)
        go_ids = [s["id"] for s in stanzas if s.get("id","").startswith("GO:")]
        self.assertEqual(len(go_ids), 9)

    def test_alt_id_collected_as_list(self):
        stanzas = self._parse(textwrap.dedent("""\
            [Term]
            id: GO:0000010
            name: test
            namespace: molecular_function
            alt_id: GO:0036422
        """))
        self.assertIn("GO:0036422", stanzas[0]["alt_id"])

    def test_relationship_tag_collected(self):
        stanzas = self._parse(textwrap.dedent("""\
            [Term]
            id: GO:0000015
            name: test
            namespace: cellular_component
            relationship: part_of GO:0005829 ! cytosol
        """))
        self.assertEqual(stanzas[0]["relationship"][0], "part_of GO:0005829")

    def test_empty_file_yields_nothing(self):
        self.assertEqual(self._parse(""), [])

    def test_last_stanza_not_lost(self):
        stanzas = self._parse(
            "[Term]\nid: GO:9999999\nname: last term\nnamespace: biological_process"
        )
        self.assertEqual(len(stanzas), 1)

    def test_definition_tag_captured(self):
        stanzas = self._parse(textwrap.dedent("""\
            [Term]
            id: GO:0000001
            name: test
            namespace: biological_process
            def: "Some definition text." [GOC:mcc, PMID:123]
        """))
        self.assertIn("Some definition text", stanzas[0]["def"])


# =========================================================================== #
# 4.  build() -- fixture-based integration tests                               #
# =========================================================================== #

class TestBuildFixture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        obo, db = _write_fixture(Path(cls._tmp.name))
        build(str(obo), str(db))
        cls.con = _open_db(db)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls._tmp.cleanup()

    def _scalar(self, sql, *args):
        return self.con.execute(sql, args).fetchone()[0]

    def _rows(self, sql, *args):
        return self.con.execute(sql, args).fetchall()

    def test_all_expected_tables_present(self):
        expected = {
            "metadata", "go_ontology", "go_term", "go_obsolete", "go_synonym",
            "go_bp_parents", "go_bp_offspring", "go_mf_parents", "go_mf_offspring",
            "go_cc_parents", "go_cc_offspring", "map_metadata", "map_counts",
        }
        actual = {r[0] for r in self.con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        self.assertTrue(expected.issubset(actual))

    def test_go_ontology_has_exactly_three_rows(self):
        self.assertEqual(self._scalar("SELECT COUNT(*) FROM go_ontology"), 3)

    def test_active_term_count(self):
        self.assertEqual(self._scalar("SELECT COUNT(*) FROM go_term"), 8)

    def test_known_term_present(self):
        row = self._rows("SELECT term, ontology FROM go_term WHERE go_id=?", "GO:0000001")
        self.assertEqual(row[0]["term"], "mitochondrion inheritance")
        self.assertEqual(row[0]["ontology"], "BP")

    def test_go_ids_have_correct_format(self):
        for row in self._rows("SELECT go_id FROM go_term"):
            self.assertRegex(row[0], r"^GO:\d{7}$")

    def test_definition_citation_stripped(self):
        defn = self._scalar("SELECT definition FROM go_term WHERE go_id=?", "GO:0000001")
        self.assertNotIn("[GOC:", defn)
        self.assertIn("mitochondria", defn)

    def test_obsolete_term_not_in_go_term(self):
        self.assertEqual(
            self._scalar("SELECT COUNT(*) FROM go_term WHERE go_id=?", "GO:0000002"), 0
        )

    def test_no_overlap_between_active_and_obsolete(self):
        self.assertEqual(
            self._scalar("SELECT COUNT(*) FROM go_term JOIN go_obsolete USING (go_id)"), 0
        )

    def test_obsolete_term_count(self):
        self.assertEqual(self._scalar("SELECT COUNT(*) FROM go_obsolete"), 1)

    def test_text_synonym_stored(self):
        rows = self._rows(
            "SELECT synonym, scope, like_go_id, secondary "
            "FROM go_synonym WHERE go_id=? AND synonym=?",
            "GO:0000001", "mitochondrial inheritance",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scope"], "EXACT")
        self.assertEqual(rows[0]["like_go_id"], 0)
        self.assertIsNone(rows[0]["secondary"])

    def test_alt_id_stored_as_synonym_with_like_go_id(self):
        rows = self._rows(
            "SELECT synonym, secondary, like_go_id FROM go_synonym "
            "WHERE go_id=? AND like_go_id=1",
            "GO:0000010",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["synonym"], "GO:0036422")

    def test_bp_parent_is_a_recorded(self):
        rows = self._rows("SELECT parent_id FROM go_bp_parents WHERE go_id=?", "GO:0000001")
        parent_ids = {r["parent_id"] for r in rows}
        self.assertIn("GO:0048308", parent_ids)
        self.assertIn("GO:0048311", parent_ids)

    def test_out_of_scope_parents_excluded(self):
        self.assertEqual(
            self._scalar("SELECT COUNT(*) FROM go_bp_parents WHERE parent_id=?", "GO:0051052"), 0
        )

    def test_direct_parent_implies_offspring_entry(self):
        self.assertEqual(
            self._scalar(
                "SELECT COUNT(*) FROM go_bp_offspring WHERE go_id=? AND offspring_id=?",
                "GO:0048308", "GO:0000001",
            ), 1
        )

    def test_transitive_ancestor_present(self):
        self.assertEqual(
            self._scalar(
                "SELECT COUNT(*) FROM go_bp_offspring WHERE go_id=? AND offspring_id=?",
                "GO:0048308", "GO:0000001",
            ), 1
        )

    def test_no_self_loops_in_offspring(self):
        self.assertEqual(
            self._scalar("SELECT COUNT(*) FROM go_bp_offspring WHERE go_id = offspring_id"), 0
        )

    def test_no_duplicate_pairs_in_offspring(self):
        self.assertEqual(
            self._scalar("SELECT COUNT(*) FROM go_bp_offspring"),
            self._scalar("SELECT COUNT(*) FROM (SELECT DISTINCT go_id, offspring_id FROM go_bp_offspring)"),
        )

    def test_map_counts_go_term(self):
        self.assertEqual(
            self._scalar("SELECT COUNT(*) FROM go_term"),
            self._scalar("SELECT count FROM map_counts WHERE map_name=?", "go_term"),
        )

    def test_map_counts_go_obsolete(self):
        self.assertEqual(
            self._scalar("SELECT COUNT(*) FROM go_obsolete"),
            self._scalar("SELECT count FROM map_counts WHERE map_name=?", "go_obsolete"),
        )

    def test_map_counts_go_synonym(self):
        self.assertEqual(
            self._scalar("SELECT COUNT(*) FROM go_synonym"),
            self._scalar("SELECT count FROM map_counts WHERE map_name=?", "go_synonym"),
        )


# =========================================================================== #
# 5.  Full go.obo integration tests                                            #
# =========================================================================== #

@pytest.mark.usefixtures("request")
class TestBuildFullObo:

    @pytest.fixture(autouse=True)
    def _attach_db(self, full_db):
        if full_db is None:
            pytest.skip("go.obo not found -- pass --obo <path> to enable")
        self.con = full_db

    def _scalar(self, sql, *args):
        return self.con.execute(sql, args).fetchone()[0]

    def _rows(self, sql, *args):
        return self.con.execute(sql, args).fetchall()

    def test_active_term_count_plausible(self):
        count = self._scalar("SELECT COUNT(*) FROM go_term")
        assert count > 35_000, f"go_term count suspiciously low: {count}"

    def test_obsolete_term_count_plausible(self):
        assert self._scalar("SELECT COUNT(*) FROM go_obsolete") > 5_000

    def test_all_three_namespaces_populated(self):
        for ns in ("BP", "MF", "CC"):
            assert self._scalar("SELECT COUNT(*) FROM go_term WHERE ontology=?", ns) > 0

    def test_no_orphaned_synonyms(self):
        assert self._scalar("""
            SELECT COUNT(*) FROM go_synonym
            WHERE go_id NOT IN (SELECT go_id FROM go_term)
        """) == 0

    def test_no_orphaned_bp_parent_child(self):
        assert self._scalar("""
            SELECT COUNT(*) FROM go_bp_parents
            WHERE go_id NOT IN (SELECT go_id FROM go_term)
        """) == 0
        assert self._scalar("""
            SELECT COUNT(*) FROM go_bp_parents
            WHERE parent_id NOT IN (SELECT go_id FROM go_term)
        """) == 0

    def test_no_overlap_active_obsolete(self):
        assert self._scalar(
            "SELECT COUNT(*) FROM go_term JOIN go_obsolete USING (go_id)"
        ) == 0

    def test_no_self_loops_in_bp_offspring(self):
        assert self._scalar(
            "SELECT COUNT(*) FROM go_bp_offspring WHERE go_id = offspring_id"
        ) == 0

    def test_no_duplicate_offspring_pairs_bp(self):
        assert (
            self._scalar("SELECT COUNT(*) FROM go_bp_offspring")
            == self._scalar("SELECT COUNT(*) FROM (SELECT DISTINCT go_id, offspring_id FROM go_bp_offspring)")
        )

    def test_every_bp_direct_parent_edge_in_offspring(self):
        assert self._scalar("""
            SELECT COUNT(*) FROM go_bp_parents p
            WHERE NOT EXISTS (
                SELECT 1 FROM go_bp_offspring o
                WHERE o.go_id = p.parent_id AND o.offspring_id = p.go_id
            )
        """) == 0

    def test_relationship_types_sensible(self):
        known = {"is_a","part_of","regulates","positively_regulates",
                 "negatively_regulates","occurs_in","has_part"}
        for table in ("go_bp_parents", "go_mf_parents", "go_cc_parents"):
            types_in_db = {
                r[0] for r in self._rows(f"SELECT DISTINCT relationship_type FROM {table}")
            }
            assert not (types_in_db - known), \
                f"Unexpected relationship_type(s) in {table}: {types_in_db - known}"

    def test_map_counts_match_actual_counts(self):
        for table in ("go_term", "go_obsolete", "go_synonym"):
            actual   = self._scalar(f"SELECT COUNT(*) FROM {table}")
            recorded = self._scalar("SELECT count FROM map_counts WHERE map_name=?", table)
            assert actual == recorded

    def test_root_bp_term_present(self):
        assert self._scalar("SELECT COUNT(*) FROM go_term WHERE go_id=?", "GO:0008150") == 1

    def test_root_mf_term_present(self):
        assert self._scalar("SELECT COUNT(*) FROM go_term WHERE go_id=?", "GO:0003674") == 1

    def test_root_cc_term_present(self):
        assert self._scalar("SELECT COUNT(*) FROM go_term WHERE go_id=?", "GO:0005575") == 1

    def test_root_bp_has_many_offspring(self):
        count = self._scalar(
            "SELECT COUNT(*) FROM go_bp_offspring WHERE go_id=?", "GO:0008150"
        )
        assert count > 20_000, f"BP root offspring count suspiciously low: {count}"


# --------------------------------------------------------------------------- #
# Session-scoped fixture                                                       #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def full_db(request, tmp_path_factory):
    obo = _obo_path(request)
    if obo is None or not obo.exists():
        yield None
        return
    db_path = tmp_path_factory.mktemp("go_full") / "go.sqlite3"
    build(str(obo), str(db_path))
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    yield con
    con.close()


if __name__ == "__main__":
    unittest.main()
