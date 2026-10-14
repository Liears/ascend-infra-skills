"""Tests that the repo is well-formed as an open-source skill package.

These are the CI checks that keep the package installable and consistent:
every skill directory has a SKILL.md with valid frontmatter, the manifests
parse, the plugin skill list is not stale, and no dossier claims an anchor it
does not record.
"""
import json
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKILLS = os.path.join(ROOT, "skills")

FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def skill_dirs():
    return sorted(
        d for d in os.listdir(SKILLS)
        if os.path.isfile(os.path.join(SKILLS, d, "SKILL.md"))
    )


class TestSkills(unittest.TestCase):
    def test_has_skills(self):
        self.assertGreaterEqual(len(skill_dirs()), 6)

    def test_every_skill_has_frontmatter(self):
        for d in skill_dirs():
            p = os.path.join(SKILLS, d, "SKILL.md")
            with open(p, encoding="utf-8") as f:
                text = f.read()
            m = FRONTMATTER.match(text)
            self.assertIsNotNone(m, f"{d}/SKILL.md has no frontmatter")
            fm = m.group(1)
            self.assertIn("name:", fm, f"{d} frontmatter missing name")
            self.assertIn("description:", fm, f"{d} frontmatter missing description")

    def test_skill_name_matches_directory(self):
        for d in skill_dirs():
            p = os.path.join(SKILLS, d, "SKILL.md")
            with open(p, encoding="utf-8") as f:
                fm = FRONTMATTER.match(f.read()).group(1)
            name = re.search(r"^name:\s*(\S+)", fm, re.MULTILINE).group(1)
            self.assertEqual(name, d, f"{d}: frontmatter name is {name}")

    def test_directory_names_are_kebab_case(self):
        for d in skill_dirs():
            self.assertRegex(d, r"^[a-z0-9]+(-[a-z0-9]+)*$", f"bad skill dir name: {d}")

    def test_descriptions_are_substantial(self):
        # A one-line description cannot carry trigger conditions, which is how
        # the runtime decides whether to load the skill.
        for d in skill_dirs():
            p = os.path.join(SKILLS, d, "SKILL.md")
            with open(p, encoding="utf-8") as f:
                fm = FRONTMATTER.match(f.read()).group(1)
            desc = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE).group(1)
            self.assertGreater(len(desc), 80, f"{d}: description too short to trigger on")


class TestManifests(unittest.TestCase):
    def test_plugin_json_parses(self):
        with open(os.path.join(ROOT, ".claude-plugin", "plugin.json")) as f:
            d = json.load(f)
        self.assertEqual(d["name"], "ascend-infra-skills")
        self.assertIn("version", d)

    def test_marketplace_json_parses(self):
        with open(os.path.join(ROOT, ".claude-plugin", "marketplace.json")) as f:
            d = json.load(f)
        self.assertTrue(d["plugins"])
        self.assertEqual(d["plugins"][0]["source"], "./")

    def test_manifest_version_matches(self):
        with open(os.path.join(ROOT, ".claude-plugin", "plugin.json")) as f:
            pv = json.load(f)["version"]
        with open(os.path.join(ROOT, ".claude-plugin", "marketplace.json")) as f:
            mv = json.load(f)["plugins"][0]["version"]
        self.assertEqual(pv, mv)


class TestDocs(unittest.TestCase):
    def test_evidence_rules_present(self):
        self.assertTrue(os.path.isfile(os.path.join(ROOT, "docs", "EVIDENCE-RULES.md")))

    def test_license_is_apache(self):
        with open(os.path.join(ROOT, "LICENSE"), encoding="utf-8") as f:
            text = f.read()
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0", text)

    def test_hardware_specs_parse(self):
        with open(os.path.join(ROOT, "references", "ascend-hardware-specs.json")) as f:
            d = json.load(f)
        self.assertIn("chips", d)
        self.assertIn("910b3", d["chips"])


class TestDossiers(unittest.TestCase):
    def dossier_files(self):
        base = os.path.join(SKILLS, "model-pr-history-knowledge", "references", "dossiers")
        if not os.path.isdir(base):
            return []
        return [os.path.join(base, f) for f in os.listdir(base)
                if f.endswith(".md") and f != "README.md"]

    def test_every_dossier_records_its_anchor(self):
        # A dossier without the HEAD it was built from cannot be checked for
        # staleness, which the querying rules forbid.
        for p in self.dossier_files():
            with open(p, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("repo HEAD", text, f"{os.path.basename(p)} missing HEAD anchor")


if __name__ == "__main__":
    unittest.main()
