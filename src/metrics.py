
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


FIELDS = ["company", "date", "address", "total"]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_CURRENCY = re.compile(r"[$£€₹]|\b(rm|usd|inr|myr)\b", re.IGNORECASE)
_PUNCT_EDGE = re.compile(r"^[^\w]+|[^\w]+$")


def normalize(value: Any, *, field_name: str | None = None) -> str:
    """
    Canonicalise a field value before comparison.

    DECISION POINT — every choice here inflates or deflates your scores, and an
    interviewer may ask you to justify any of them:

      - casefold          : "TESCO" == "Tesco". Almost always right for OCR.
      - unicode NFKC      : collapses fullwidth/ligature variants.
      - collapse ws       : "12  Jalan" == "12 Jalan". Right for multi-line
                            addresses where line breaks are arbitrary.
      - strip currency    : "RM 25.00" == "25.00" for the `total` field.
      - strip edge punct  : trailing periods/commas from OCR noise.

    What it deliberately does NOT do: reorder tokens, correct spelling, or
    normalise date formats. Date normalisation in particular is tempting and
    dangerous — "01/02/2019" vs "02/01/2019" are different dates, and a
    normaliser that treats them as equal is hiding a real failure.
    """
    if value is None:
        return ""
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    if field_name == "total":
        text = _CURRENCY.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    text = _PUNCT_EDGE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Levenshtein
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Iterative two-row Levenshtein distance. O(len(a) * len(b)) time, O(len(b)) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    """Levenshtein distance scaled to [0, 1] by the longer string."""
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def anls_score(pred: str, gold: str, threshold: float = 0.5) -> float:
    """
    ANLS for a single field.

    similarity = 1 - normalized_levenshtein
    If similarity < threshold, return 0.0 — the standard ANLS convention, which
    stops near-random guesses from accruing partial credit.
    """
    similarity = 1.0 - normalized_levenshtein(pred, gold)
    return similarity if similarity >= threshold else 0.0


# ---------------------------------------------------------------------------
# Prediction parsing
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_prediction(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Extract a JSON object from raw model output.

    Returns (parsed_dict, failure_reason). On success failure_reason is None.

    IMPORTANT: a parse failure is a RESULT, not a bug to paper over. The
    zero-shot base model will fail to produce valid JSON reasonably often, and
    the drop in that rate after fine-tuning is one of the more honest things
    you can report. Log it; never silently retry or repair.
    """
    if not raw or not raw.strip():
        return None, "empty_output"

    text = raw.strip()

    # Strip markdown fences if the model wrapped its answer.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        return (obj, None) if isinstance(obj, dict) else (None, "not_an_object")
    except json.JSONDecodeError:
        pass

    # Fall back to the first {...} span. This is a concession to reality, not
    # repair: models often prepend "Here is the extracted data:".
    match = _JSON_BLOCK.search(text)
    if match:
        try:
            obj = json.loads(match.group(0))
            return (obj, None) if isinstance(obj, dict) else (None, "not_an_object")
        except json.JSONDecodeError:
            return None, "malformed_json"

    return None, "no_json_found"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class Scores:
    """Aggregate results over a split."""

    n: int = 0
    parse_failures: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    exact_hits: dict[str, int] = field(default_factory=lambda: {f: 0 for f in FIELDS})
    anls_sum: dict[str, float] = field(default_factory=lambda: {f: 0.0 for f in FIELDS})

    def summary(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n": 0}
        per_field_em = {f: self.exact_hits[f] / self.n for f in FIELDS}
        per_field_anls = {f: self.anls_sum[f] / self.n for f in FIELDS}
        return {
            "n": self.n,
            "parse_failure_rate": self.parse_failures / self.n,
            "failure_reasons": dict(self.failure_reasons),
            "exact_match": {
                "overall": sum(per_field_em.values()) / len(FIELDS),
                "per_field": per_field_em,
            },
            "anls": {
                "overall": sum(per_field_anls.values()) / len(FIELDS),
                "per_field": per_field_anls,
            },
        }

    def pretty(self) -> str:
        s = self.summary()
        if s["n"] == 0:
            return "no samples scored"
        lines = [
            f"n = {s['n']}   parse failures = {s['parse_failure_rate']:.1%}",
            "",
            f"{'field':<12}{'exact match':>14}{'ANLS':>10}",
            "-" * 36,
        ]
        for f in FIELDS:
            lines.append(
                f"{f:<12}{s['exact_match']['per_field'][f]:>13.1%}"
                f"{s['anls']['per_field'][f]:>10.3f}"
            )
        lines += [
            "-" * 36,
            f"{'OVERALL':<12}{s['exact_match']['overall']:>13.1%}"
            f"{s['anls']['overall']:>10.3f}",
        ]
        if s["failure_reasons"]:
            lines += ["", "parse failures by reason:"]
            for reason, count in sorted(s["failure_reasons"].items(), key=lambda kv: -kv[1]):
                lines.append(f"  {reason:<20}{count:>5}")
        return "\n".join(lines)


def score_sample(scores: Scores, raw_prediction: str, gold: dict[str, Any]) -> dict[str, Any]:
    """
    Score one prediction against ground truth and fold it into `scores`.

    Returns a per-sample record, so you can dump every result to disk and do
    failure analysis later without re-running inference. Do dump it — on Day 5
    you will want to grep this file, not regenerate it.
    """
    scores.n += 1
    parsed, reason = parse_prediction(raw_prediction)

    if parsed is None:
        # Total miss on every field. Do not award partial credit for unparseable
        # output — a downstream system could not use it at all.
        scores.parse_failures += 1
        scores.failure_reasons[reason] = scores.failure_reasons.get(reason, 0) + 1
        return {
            "parse_ok": False,
            "parse_failure_reason": reason,
            "raw": raw_prediction,
            "fields": {f: {"pred": None, "gold": gold.get(f), "em": 0, "anls": 0.0} for f in FIELDS},
        }

    record: dict[str, Any] = {"parse_ok": True, "raw": raw_prediction, "fields": {}}
    for f in FIELDS:
        pred_norm = normalize(parsed.get(f), field_name=f)
        gold_norm = normalize(gold.get(f), field_name=f)

        em = int(pred_norm == gold_norm)
        anls = 1.0 if (not pred_norm and not gold_norm) else anls_score(pred_norm, gold_norm)

        scores.exact_hits[f] += em
        scores.anls_sum[f] += anls
        record["fields"][f] = {
            "pred": parsed.get(f),
            "gold": gold.get(f),
            "pred_norm": pred_norm,
            "gold_norm": gold_norm,
            "em": em,
            "anls": round(anls, 4),
        }
    return record


if __name__ == "__main__":
    # Sanity checks. Run `python src/metrics.py` — everything should pass.
    assert levenshtein("kitten", "sitting") == 3
    assert normalize("  RM 25.00 ", field_name="total") == "25.00"
    assert normalize("TESCO STORES.") == "tesco stores"
    assert anls_score("tesco", "tesco") == 1.0
    assert anls_score("tesco", "xxxxx") == 0.0

    parsed, reason = parse_prediction('```json\n{"company": "Tesco"}\n```')
    assert parsed == {"company": "Tesco"} and reason is None

    parsed, reason = parse_prediction("I could not read this receipt.")
    assert parsed is None and reason == "no_json_found"

    print("metrics.py — all sanity checks passed")
