#!/usr/bin/env python3
"""
Ascend Open-Source Finder — search multiple code/model hosts for existing
implementations of a (model, hardware) pair, and rate each hit on evidence.

Core rule: a hit is only as good as its evidence. A repo that says "supports
Ascend" in its README but ships no patch, no run log, and no weights is rated
`claimed`, not `verified`.

Usage:
    python3 find_open_source.py --model "DeepSeek-V4.1-Flash" --hardware 910B
    python3 find_open_source.py --model deepseek-v4.1 --hardware 910B --json
    python3 find_open_source.py --model GLM-5.3 --hardware "A2" --max 20

Sources:
    - GitHub  (REST search, unauthenticated works, GITHUB_TOKEN raises limits)
    - GitCode (REST search, needs GITCODE_TOKEN — a private-token header)
    - ModelScope (model page probing; the search API endpoint has moved)

Only the standard library is used.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Iterable

UA = "ascend-open-source-finder/0.1 (+https://github.com/)"
TIMEOUT = 25


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def _get(url: str, headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001 - network errors are expected here
        return 0, str(e).encode()


def _get_json(url: str, headers: dict | None = None) -> tuple[int, object | None]:
    status, body = _get(url, headers)
    if status != 200:
        return status, None
    try:
        return status, json.loads(body)
    except Exception:  # noqa: BLE001
        return status, None


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------

# Hardware synonyms. A model ported for 910B may say "A2", "Atlas 800 A2",
# "910B", "910B3" or "ascend910b". The finder must try all of them.
HARDWARE_SYNONYMS: dict[str, list[str]] = {
    "910b": ["910B", "910B3", "910b", "A2", "Atlas 800 A2", "ascend910b"],
    "910b3": ["910B3", "910B", "A2", "Atlas 800 A2"],
    "a2": ["A2", "Atlas 800 A2", "910B", "910B3"],
    "910c": ["910C", "A3", "Atlas 800 A3"],
    "a3": ["A3", "Atlas 800 A3", "910C"],
    "950": ["950", "950DT", "A5", "A5x"],
    "a5": ["A5", "950", "950DT"],
}

# Model name variants. V4.1 vs V4_1 vs V41 vs V4-1 all appear in the wild.
def model_variants(model: str) -> list[str]:
    """Return name spellings that appear in the wild, in a deterministic order.

    A `set` here would make the query list (and therefore the report) vary
    between runs for no reason; keep it a list with an explicit order.
    """
    base = model.strip()
    raw = [
        base,                                   # as typed
        base.lower(),                           # lowercased
        base.replace(".", "_"),                 # DeepSeek-V4_1-Flash
        base.replace(".", "-"),                 # DeepSeek-V4-1-Flash
        re.sub(r"[.\-_]", "", base),            # DeepSeekV41Flash
    ]
    seen: set[str] = set()
    return [v for v in raw if v and not (v in seen or seen.add(v))]


def build_queries(model: str, hardware: str) -> list[str]:
    """Build a bounded but complete query set.

    Two cross-products matter and both must be covered without exploding the
    query count (each query is an API call; unauthenticated GitHub search allows
    only ~10/minute):

      - the canonical model name across *every* hardware synonym, so an
        `A2`/`Atlas 800 A2`-named repo is not missed by an over-eager slice;
      - each model spelling across the *primary* hardware term.

    An earlier version sliced the hardware synonyms to the first three and so
    never issued an `A2` query at all.
    """
    hw_key = hardware.strip().lower().replace(" ", "")
    hw_terms = HARDWARE_SYNONYMS.get(hw_key, [hardware])
    primary_hw = hw_terms[0]
    variants = model_variants(model)
    canonical = variants[0]

    queries: list[str] = []
    # Canonical name x all hardware aliases.
    for hw in hw_terms:
        queries.append(f"{canonical} {hw}")
    # Each spelling x the primary hardware term.
    for mv in variants[1:]:
        queries.append(f"{mv} {primary_hw}")
    # A generic "ascend" query catches repos naming neither the chip nor A2/A3.
    queries.append(f"{canonical} ascend")

    seen: set[str] = set()
    return [q for q in queries if not (q in seen or seen.add(q))]


# --------------------------------------------------------------------------
# Evidence scoring
# --------------------------------------------------------------------------

# Signals that a repo contains a real implementation rather than a claim.
EVIDENCE_PATTERNS: list[tuple[str, str, int]] = [
    (r"\b0\d{3}-.*\.patch|\.patch$", "patch files", 3),
    (r"\bREPRO\.md|reproduc|复现", "repro doc", 2),
    (r"\bEXPECTED_PERF|CORRECTNESS_STATUS|ACCEPTANCE", "acceptance/perf doc", 3),
    (r"\bevidence/|reports?/|logs?_meta/", "evidence dir", 3),
    (r"\bserve_.*\.sh|launch.*\.sh|run.*\.sh", "launch script", 2),
    (r"\bDockerfile|quay\.|docker pull", "container recipe", 2),
    (r"\bW4A8|W8A8|W4A16|quant", "quantization material", 1),
    (r"\bCANN|torch_npu|vllm[-_]ascend|mindie", "ascend stack reference", 2),
]

LOW_TRUST = [
    (r"\bprivate\b", "marked private", -2),
    (r"\btest|toy|demo|hello", "looks like a toy", -1),
]


@dataclass
class Candidate:
    source: str
    full_name: str
    url: str
    description: str = ""
    language: str = ""
    stars: int = 0
    pushed_at: str = ""
    created_at: str = ""
    license: str = ""
    topics: list[str] = field(default_factory=list)
    evidence_signals: list[str] = field(default_factory=list)
    doubts: list[str] = field(default_factory=list)
    score: int = 0
    rating: str = "claimed"     # claimed | partial | verified
    notes: list[str] = field(default_factory=list)


def score_candidate(c: Candidate) -> None:
    """Rate evidence from metadata alone. This is a triage score, not a verdict.

    Metadata cannot prove a repo runs. It can only tell you which repos are worth
    a manual look. The rating reflects that limit.
    """
    blob = " ".join([c.description or "", " ".join(c.topics or []), c.full_name]).lower()
    score = 0
    for pat, label, weight in EVIDENCE_PATTERNS:
        if re.search(pat, blob, re.IGNORECASE):
            c.evidence_signals.append(label)
            score += weight
    for pat, label, weight in LOW_TRUST:
        if re.search(pat, blob, re.IGNORECASE):
            c.doubts.append(label)
            score += weight

    # Star count is a weak proxy for presence but a strong one for "someone else
    # has looked at this". Weight it lightly and never let it dominate evidence.
    if c.stars >= 20:
        score += 2
    elif c.stars >= 5:
        score += 1

    # Extremely fresh or extremely stale repos both deserve a flag.
    if not c.pushed_at:
        c.notes.append("no push date visible")
    c.score = score

    if score >= 6 and not c.doubts:
        c.rating = "verified"
    elif score >= 3:
        c.rating = "partial"
    else:
        c.rating = "claimed"


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------

def search_github(query: str, limit: int = 10) -> list[Candidate]:
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = (
        "https://api.github.com/search/repositories?"
        + urllib.parse.urlencode({"q": query, "sort": "stars", "order": "desc", "per_page": limit})
    )
    status, data = _get_json(url, headers)
    if status != 200 or not isinstance(data, dict):
        return []
    out: list[Candidate] = []
    for r in data.get("items", [])[:limit]:
        lic = (r.get("license") or {}).get("spdx_id") or ""
        out.append(
            Candidate(
                source="github",
                full_name=r.get("full_name", ""),
                url=r.get("html_url", ""),
                description=r.get("description") or "",
                language=r.get("language") or "",
                stars=r.get("stargazers_count", 0),
                pushed_at=(r.get("pushed_at") or "")[:10],
                created_at=(r.get("created_at") or "")[:10],
                license=lic,
                topics=r.get("topics") or [],
            )
        )
    return out


def search_gitcode(query: str, limit: int = 10) -> list[Candidate]:
    """GitCode search needs a token; without one we return a clear, non-fatal note."""
    token = os.environ.get("GITCODE_TOKEN")
    if not token:
        return []
    url = "https://gitcode.com/api/v5/search/repositories?" + urllib.parse.urlencode(
        {"q": query, "per_page": limit}
    )
    status, data = _get_json(url, {"private-token": token})
    if status != 200 or not isinstance(data, list):
        return []
    out: list[Candidate] = []
    for r in data[:limit]:
        out.append(
            Candidate(
                source="gitcode",
                full_name=r.get("full_name") or r.get("path_with_namespace", ""),
                url=r.get("html_url", ""),
                description=r.get("description") or "",
                language=r.get("language") or "",
                stars=r.get("stargazers_count", 0),
                pushed_at=(r.get("updated_at") or "")[:10],
            )
        )
    return out


def _modelscope_model_info(model_id: str) -> dict | None:
    """Fetch ModelScope model metadata, or None if the model does not exist.

    The HTML page is a single-page app that returns 200 for missing models and
    echoes the URL path, so a page check yields false positives for every name
    guessed. The JSON API is authoritative:

      GET /api/v1/models/{owner}/{name}
        200 -> {"Code":200,"Data":{...}}   model exists
        404 -> {"Code":10010205001,"Message":"...record not found"}

    Verified on 2026-10-14 with DeepSeek-V4.1-Flash-w4a8-Ascend (200) and a
    deliberately nonexistent id (404).
    """
    url = f"https://modelscope.cn/api/v1/models/{model_id}"
    status, data = _get_json(url)
    if status != 200 or not isinstance(data, dict):
        return None
    if data.get("Code") != 200:
        return None
    return data.get("Data") or {}


def _modelscope_page_exists(model_id: str) -> bool:
    """Kept as a thin wrapper: existence is decided by the JSON API, not the page."""
    return _modelscope_model_info(model_id) is not None


def probe_modelscope(model: str, hardware: str, owners: list[str] | None = None) -> list[Candidate]:
    """Probe likely ModelScope model ids by name convention.

    ModelScope's search API endpoint has moved and returns 404 for the legacy
    paths, so we cannot enumerate. We probe the conventional naming patterns and
    confirm existence with `_modelscope_page_exists`, which does not trust the
    HTTP status code alone.
    """
    variants = model_variants(model)
    hw_hint = hardware.strip()
    patterns: list[str] = []
    for mv in variants[:2]:
        patterns += [
            f"{mv}-w4a8-{hw_hint}",
            f"{mv}-W4A8-{hw_hint}",
            f"{mv}-{hw_hint}",
            f"{mv}-w8a8-{hw_hint}",
        ]
    out: list[Candidate] = []
    for owner in (owners or ["chiro2001"]):
        for pat in patterns[:6]:
            mid = f"{owner}/{pat}"
            info = _modelscope_model_info(mid)
            if info is None:
                continue
            desc = (info.get("Description") or "").strip().replace("\n", " ")
            if not desc:
                desc = (info.get("ChineseName") or info.get("Name") or "").strip()
            out.append(
                Candidate(
                    source="modelscope",
                    full_name=mid,
                    url=f"https://modelscope.cn/models/{mid}",
                    description=f"ModelScope weights. {desc}"[:300],
                    pushed_at=(info.get("LastUpdatedTime") or "")[:10],
                    created_at=(info.get("CreatedTime") or "")[:10],
                    stars=info.get("Stars") or 0,
                    license=(info.get("License") or ""),
                    notes=["existence confirmed via ModelScope JSON API"],
                )
            )
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def render(cands: list[Candidate], model: str, hardware: str, queries: list[str]) -> str:
    lines: list[str] = []
    lines.append(f"# Open-source implementations: {model} on {hardware}")
    lines.append("")
    lines.append(f"Queries issued ({len(queries)}): " + " | ".join(queries))
    lines.append("")
    if not cands:
        lines.append("No candidates found. Widen the hardware term or try another source.")
        return "\n".join(lines)

    order = {"verified": 0, "partial": 1, "claimed": 2}
    cands = sorted(cands, key=lambda c: (order.get(c.rating, 3), -c.score, -c.stars))

    lines.append(f"{len(cands)} candidate(s), rated by evidence in metadata:")
    lines.append("")
    for c in cands:
        lines.append(f"## [{c.rating}] {c.full_name}  (score {c.score})")
        lines.append(f"- url: {c.url}")
        lines.append(f"- source: {c.source} | lang: {c.language or '-'} | stars: {c.stars} | license: {c.license or '-'}")
        lines.append(f"- pushed: {c.pushed_at or '-'} | created: {c.created_at or '-'}")
        if c.description:
            lines.append(f"- desc: {c.description[:200]}")
        if c.evidence_signals:
            lines.append(f"- evidence signals: {', '.join(sorted(set(c.evidence_signals)))}")
        if c.doubts:
            lines.append(f"- doubts: {', '.join(sorted(set(c.doubts)))}")
        for n in c.notes:
            lines.append(f"- note: {n}")
        lines.append("")
        lines.append(
            "  -> `"
            + {"verified": "verify by reading the patch set and the acceptance doc",
               "partial": "read REPRO/scripts before trusting",
               "claimed": "metadata shows no implementation evidence; treat as a lead only"}
            [c.rating]
            + "`"
        )
        lines.append("")

    lines.append("---")
    lines.append("Rating is metadata-only triage. `verified` here means the repo advertises")
    lines.append("implementation evidence (patches/acceptance/repro), NOT that it was run.")
    lines.append("Audit the top candidate before citing it:")
    lines.append("")
    lines.append("    git clone --depth 1 <url> /tmp/cand && \\")
    lines.append("        python3 scripts/audit_repo.py /tmp/cand")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Find open-source implementations of a model on Ascend hardware.")
    ap.add_argument("--model", required=True, help='e.g. "DeepSeek-V4.1-Flash"')
    ap.add_argument("--hardware", required=True, help='e.g. "910B", "A2", "910C", "A3"')
    ap.add_argument("--max", type=int, default=10, help="results per query (default 10)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    ap.add_argument("--sources", default="github,gitcode,modelscope",
                    help="comma list of sources to query")
    args = ap.parse_args(list(argv) if argv is not None else None)

    sources = {s.strip() for s in args.sources.split(",") if s.strip()}
    queries = build_queries(args.model, args.hardware)

    seen: dict[str, Candidate] = {}
    for q in queries:
        if "github" in sources:
            for c in search_github(q, args.max):
                seen.setdefault(c.full_name, c)
        if "gitcode" in sources:
            for c in search_gitcode(q, args.max):
                seen.setdefault(c.full_name, c)
        time.sleep(0.4)  # be a good citizen with unauthenticated search

    if "modelscope" in sources:
        # Probe under the owners we actually discovered, so the guess is grounded
        # in a repo that exists rather than a made-up namespace.
        owners = sorted({c.full_name.split("/", 1)[0] for c in seen.values() if "/" in c.full_name})
        owners = owners[:5] or ["chiro2001"]
        for c in probe_modelscope(args.model, args.hardware, owners=owners):
            seen.setdefault(c.full_name, c)

    cands = list(seen.values())
    for c in cands:
        score_candidate(c)

    if args.json:
        print(json.dumps(
            {"model": args.model, "hardware": args.hardware, "queries": queries,
             "candidates": [asdict(c) for c in cands]},
            indent=2, ensure_ascii=False,
        ))
    else:
        print(render(cands, args.model, args.hardware, queries))

    if not os.environ.get("GITHUB_TOKEN"):
        print("\n[hint] set GITHUB_TOKEN to raise the search rate limit.", file=sys.stderr)
    if not os.environ.get("GITCODE_TOKEN"):
        print("[hint] set GITCODE_TOKEN to enable GitCode search.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
