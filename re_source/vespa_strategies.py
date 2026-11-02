"""Vespa-backed strategies."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langfuse import observe

from .config import LLM_VALIDATION_MAX_RESULTS
from . import lane_union
from .search_types import SearchResult
from .vespa_chunks import (
    CHUNK_FIELDS,
    best_score_by_parent,
    normalize_full_row,
    quote_yql_string,
    read_chunk_response,
)
from .vespa_client import VespaClient

logger = logging.getLogger(__name__)


def _escape_embed_text(text: str) -> str:
    """Make `text` safe to interpolate inside `embed(id, "...")`. Vespa's
    YQL string literal accepts `\\` for backslash, `\\"` for quote, `\\n`
    for newline. Whitespace collapsed because long-form queries with hard
    wraps don't help embedding quality anyway."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
    )


@dataclass
class StrategyResult:
    """Result from running a Vespa-backed strategy."""
    results: list[SearchResult]
    elapsed_seconds: float
    approach_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Content strategies exclude the AI-generated author_profile chunk so they match
# on what the tip SAYS, not on who the synthetic profile claims submitted it.
# Only the Author Profile strategy opts back in (exclude_author_profile=False).
_EXCLUDE_AUTHOR_PROFILE = '!(column_key contains "author_profile")'

# YQL is built per-call in AtomicVespaStrategy.execute: ANN candidates
# ({targetHits:N}nearestNeighbor) OR'd with lexical userInput(@query) hits,
# ranked by the `hybrid` profile (ColBERT MaxSim in second phase). targetHits
# is parameterized so the find-all author-profile strategy can retrieve deep.


# userInput() parses to weakAnd, an APPROXIMATE top-k operator: it keeps a
# running score threshold and skips documents whose upper bound can't beat it.
# The surviving set is not reproducible, and the harder it prunes the more it
# misses. Pruning gets worse as the query gets longer, which is exactly the
# shape of a real reporter query.
#
# Measured on a static 696k-chunk index, under the real `hybrid` profile with
# the ANN leg OR'd in, for a 15-term query, 3 runs each:
#
#   lexical            p50     totalCount        reproducible
#   default            129ms   4707..4726        no
#   {targetHits:5000}   98ms   33783..33823      no
#   {targetHits:50000} 145ms   194211..194825    yes
#   {grammar:"any"}    150ms   498870 (exact)    yes
#
# So the default was matching ~4.7k of 498k eligible chunks — under 1% — and a
# different 1% each run. A plain OR is exhaustive, deterministic, and costs
# ~20ms more than the broken default; weakAnd needs targetHits ~50000 to reach
# the same stability and is no cheaper. Coverage was 100% and non-degraded for
# every variant, so this is pruning, not timeouts.
#
# Deliberately not a tuned targetHits: the value that works scales with query
# term count, and a constant fitted to one query length was already wrong twice.
# Function words carry essentially no BM25 weight — IDF for "the" over a 700k
# chunk corpus is ~0.07 against ~25 for a rare term — but under an exhaustive OR
# they are what makes a 15-word query match ~500,000 chunks, roughly 72% of the
# index. Dropping them from the LEXICAL leg only shrinks the candidate set; it
# cannot lose a document that would have ranked, because those terms were not
# what ranked it.
#
# This matters beyond cost. BM25's unbounded scale is currently the only thing
# keeping filler-only matches out of contention: normalise it (see the rejected
# hybrid_sat in chunk.sd) and 500k chunks that matched on "the" start competing
# on dense similarity alone. Shrinking the candidate set is the durable fix; the
# rank formula was the symptom.
#
# The dense and ColBERT legs still see the ORIGINAL query — this only affects
# what goes to userInput(). Conservative list: unambiguous function words only,
# and if stripping would empty the query the original is kept.
_LEXICAL_STOPWORDS = frozenset("""
a an the and or but if then than that this these those
of in on at to for from by with without into onto about as
is are was were be been being am do does did doing done
have has had having will would shall should can could may might must
i me my we us our you your he him his she her it its they them their
who whom whose which what when where why how
there here not no nor so such own same too very just also
any some all each every other another more most much many few
anything anyone someone something
""".split())


# Toggle so stripping can be read side by side against not stripping. It is a
# relevance change with a large measured effect — det.bm25 overlap@50 fell as
# low as 0.010, i.e. a near-total reshuffle of the top 50 — and churn that size
# proves the change matters, not that it helps.
STRIP_LEXICAL_STOPWORDS = os.getenv("VESPA_LEXICAL_STOPWORDS", "1").lower() not in ("0", "false", "no")


def _lexical_text(query: str) -> str:
    """The query as sent to userInput().

    Returns:
        The stripped text, or the original if stripping leaves nothing.
    """
    if not STRIP_LEXICAL_STOPWORDS:
        return query
    kept = [w for w in query.split() if w.strip(".,;:!?\"'()[]").lower() not in _LEXICAL_STOPWORDS]
    return " ".join(kept) if kept else query


# Web-syntax operators: a quote, a +/-/( starting a term, or a capital OR between
# terms. Hyphens inside a word (h-e-b, 555-1234) are not operators.
_OPERATOR = re.compile(r'"|(?:^|\s)[-+(]|\sOR\s')


def _has_operators(query: str) -> bool:
    return bool(_OPERATOR.search(query))


_LEXICAL_CLAUSE = '({grammar: "any"}userInput(@query))'
# "web" is "all" plus `a OR b`: every term required, "quoted phrases" and -excluded
# terms honored, and +term matched as typed (no stemming).
_KEYWORD_CLAUSE = '({grammar: "web", defaultIndex: "keyword_text"}userInput(@query))'


# Which rank profile the hybrid retrieval path uses. hybrid_sat normalises BM25
# into [0,1) so it is comparable with closeness; `hybrid` keeps the original
# unbounded sum and stays available for comparison. See vespa/schemas/chunk.sd.
HYBRID_PROFILE = os.getenv("VESPA_HYBRID_PROFILE", "hybrid_sat")

# k in hybrid_sat's bm25/(k + bm25) — the BM25 value that maps to 0.5. Unset
# means the schema default (20). Set to sweep without redeploying.
BM25_K = os.getenv("VESPA_BM25_K")


def _profile_inputs() -> dict:
    """Extra rank-profile inputs for the hybrid path.

    Returns:
        Only bm25_k, and only when overridden.
    """
    if BM25_K and HYBRID_PROFILE == "hybrid_sat":
        return {"bm25_k": BM25_K}
    return {}


# How deep retrieval goes before anything is validated. Users don't tune this:
# the adaptive relevance cutoff (find_relevance_cutoff) and LLM validation
# decide how many results come back, so exposing a Top-K knob just adds a
# confusing dial. Set generous for a recall-first pool — the strategies'
# max_parents (200) and the LLM cap bound the actual result count, not this.
DEFAULT_TOP_K = 500

# Pool candidates validated per page: the first page, and each "Load more".
DEFAULT_PAGE_SIZE = 50


class Paged:
    """Declares that a strategy validates its candidate pool a page at a time."""

    PAGE_SIZE = DEFAULT_PAGE_SIZE
    paginates = True


class AtomicVespaStrategy:
    """Port of re_source/strategies.py:AtomicStrategy on Vespa."""

    name = "Atomic (Vespa)"

    def __init__(self, client: VespaClient):
        self.client = client

    @observe(name="Atomic (Vespa)")
    def execute(
        self,
        query: str,
        top_k: int = 100,
        retrieval_query: str | None = None,
        extra_where: str | None = None,
        target_hits: int | None = None,
        apply_cutoff: bool = True,
        max_parents: int = 200,
        exclude_author_profile: bool = True,
    ) -> StrategyResult:
        """Run the Atomic pipeline.

        Args:
            query: the user's query. ColBERT MaxSim reranking always uses this.
            top_k: max chunk hits to pull from Vespa.
            retrieval_query: if set, this text drives dense ANN retrieval
                and BM25 lexical match instead of `query`. Used by HyDE
                (retrieval_query=hypothetical document) and QueryExpansion
                (retrieval_query=expanded query) so rerank still grounds
                on what the user actually asked.
            extra_where: optional YQL clause AND-ed onto retrieval (e.g.
                'column_key contains "author_profile"', which the Author Profile
                strategy passes). Trusted input — don't pass user strings.
            exclude_author_profile: when True (default), drop the synthetic
                author_profile chunk from retrieval so content strategies match
                real submission text. The Author Profile strategy passes False.
        """
        start = time.time()
        retrieval_text = retrieval_query or query
        # Dense ANN depth follows retrieval breadth so a large top_k deepens
        # BOTH legs (dense + lexical), not just the lexical one.
        if target_hits is None:
            target_hits = top_k

        nn = f"({{targetHits: {target_hits}}}nearestNeighbor(dense, q_dense))"
        where = f"({nn} or {_LEXICAL_CLAUSE})"
        if extra_where:
            where += f" and {extra_where}"
        if exclude_author_profile:
            where += f" and {_EXCLUDE_AUTHOR_PROFILE}"
        yql = f"select {CHUNK_FIELDS} from chunk where {where}"
        # e5-large-v2 requires a "query: " prefix at query time;
        # documents were embedded with "passage: " baked into the
        # indexing statement (see vespa/schemas/chunk.sd).
        # Sanitize text going into the YQL embed("...") literal: real
        # queries can contain newlines, quotes, and backslashes that
        # would otherwise break Vespa's string parser.
        ret_safe = _escape_embed_text(retrieval_text)
        q_safe = _escape_embed_text(query)
        response = self.client.query(
            yql=yql,
            ranking=HYBRID_PROFILE,
            hits=top_k,
            timeout="30s",
            # userInput() only; the dense and ColBERT legs below still embed
            # the full text.
            query=_lexical_text(retrieval_text),
            q_dense=f'embed(e5, "query: {ret_safe}")',
            q_colbert=f'embed(colbert, "{q_safe}")',
            **_profile_inputs(),
        )

        # Vespa already sorted by ColBERT (second-phase) score, so that is the
        # feature the cutoff logic has to see. An empty response needs no
        # special case: the reader short-circuits without touching the client.
        readout = read_chunk_response(
            self.client,
            response,
            score_feature="secondPhase",
            apply_cutoff=apply_cutoff,
            max_parents=max_parents,
        )

        return StrategyResult(
            results=readout.results,
            elapsed_seconds=time.time() - start,
            approach_name=self.name,
            metadata={
                "num_candidates": readout.num_candidates,
                "cutoff": readout.cutoff,
                "num_parents": len(readout.parents),
            },
        )



# ----- BM25 ------------------------------------------------------------


class BM25VespaStrategy:
    """BM25 keyword search over complete rows, with no embedding fields."""

    name = "BM25 (Vespa)"

    def __init__(self, client: VespaClient):
        self.client = client

    @observe(name="BM25 (Vespa)")
    def execute(
        self,
        query: str,
        top_k: int = 100,
    ) -> StrategyResult:
        """Search every stored field on parent rows using lexical BM25 only."""
        start = time.time()
        yql = (
            "select parent_id, source_file, full_row from parent where "
            f"{_KEYWORD_CLAUSE}"
        )
        response = self.client.query(
            yql=yql,
            ranking="keyword",
            hits=top_k,
            timeout="15s",
            # Stripping stopwords would drop OR and split "quoted phrases".
            query=query if _has_operators(query) else _lexical_text(query),
        )
        results = [_keyword_result(hit, query) for hit in _response_hits(response)]

        return StrategyResult(
            results=results,
            elapsed_seconds=time.time() - start,
            approach_name=self.name,
            metadata={"num_parents": len(results)},
        )


def _response_hits(response: dict) -> list[dict]:
    return response.get("root", {}).get("children", []) or []


# One web-syntax term: an optional +/- then a "phrase", a (group), or a word. An
# unclosed quote or parenthesis runs to the end, as Vespa reads it.
_SYNTAX_TERM = re.compile(r'([-+]?)(?:"([^"]*)"?|\(([^)]*)\)?|(\S+))')


def _word_and_pieces(word: str) -> list[str]:
    return [word, *re.findall(r"[\w]+", word, flags=re.UNICODE)]


def _keyword_phrases(query: str) -> list[str]:
    """Useful display highlights, preserving an email or identifier intact.

    With operators: phrases stay whole, and excluded terms and OR are skipped.
    """
    whole = query.strip()
    if not whole:
        return []
    if not _has_operators(whole):
        return list(dict.fromkeys(_word_and_pieces(whole)))
    phrases = []
    for m in _SYNTAX_TERM.finditer(whole):
        sign, quoted, group, word = m.groups()
        if sign == "-":
            continue
        if quoted is not None:
            phrases.append(quoted.strip())
        elif group is not None:
            # One level, no operators inside: every word is a wanted term.
            for inner in _SYNTAX_TERM.finditer(group):
                phrases.extend([inner.group(2).strip()] if inner.group(2) is not None
                               else _word_and_pieces(inner.group(4) or ""))
        elif word != "OR":
            phrases.extend(_word_and_pieces(word))
    return [p for p in dict.fromkeys(phrases) if p]


def _matching_keyword_chunks(parent_id: str, fields: dict, query: str, score: float) -> list[dict]:
    phrases = [phrase.casefold() for phrase in _keyword_phrases(query)]
    chunks = []
    for index, (name, value) in enumerate(fields.items()):
        if value is None or not str(value).strip():
            continue
        searchable = f"{name}: {value}".casefold()
        if not any(phrase in searchable for phrase in phrases):
            continue
        chunks.append({
            "chunk_id": f"{parent_id}_keyword_{index}",
            "column_name": str(name),
            "column_key": re.sub(r"\W+", "_", str(name).casefold()).strip("_"),
            "text": str(value),
            "score": score,
        })
    return chunks


def _keyword_result(hit: dict, query: str) -> SearchResult:
    raw = hit.get("fields", {})
    fields = normalize_full_row(raw.get("full_row"))
    score = raw.get("matchfeatures", {}).get(
        "bm25(keyword_text)", hit.get("relevance", 0.0)
    )
    parent_id = raw.get("parent_id", "")
    chunks = _matching_keyword_chunks(parent_id, fields, query, score)
    return SearchResult(
        doc_id=parent_id,
        text="\n".join(chunk["text"] for chunk in chunks[:3]),
        source_file=raw.get("source_file", ""),
        fields=fields,
        score=score,
        payload={
            "parent_id": parent_id,
            "matching_chunks": chunks,
            "num_matching_chunks": len(chunks),
            "highlights": _keyword_phrases(query),
        },
    )


# ----- LLM-validated wrappers ------------------------------------------


def cited_evidence_view(fields: dict, evidence) -> tuple[list[dict], str]:
    """What a result should show, given the field/quote pairs a validator cited
    for it. Returns (corroborated evidence, body text).

    Only evidence the row itself corroborates. The citations are model output,
    and they become both the evidence chips and the result body — so an
    uncited field or an invented quote would be rendered and exported as the
    tipster's own words, with the real text gone and nothing left to contradict
    it. `corroborated_evidence` keeps a pair only when the field exists on this
    row and the quote occurs in it.

    An empty result is the same state as a validator that cited nothing: the
    chip renders "—" and the body falls back to `llm_reason`, so the
    AI-written author_profile never reaches the screen as the tipster's prose.

    Pure, and module-level, because this is the decision a reporter actually
    sees. It used to be a mutation loop inside `validate_page`, reachable only
    with a VespaClient and a live LLM — so the corroboration rule was tested
    thoroughly on its own and the code applying it had no test at all.
    """
    # Imported here rather than at module scope, like the other llm_utils uses
    # in this file — importing it eagerly pulls the provider SDKs onto the
    # search path.
    from .llm_utils import corroborated_evidence

    kept = corroborated_evidence(fields, evidence)
    text = "\n".join(f"{ev['field']}: {ev['quote']}" for ev in kept)
    return kept, text


def apply_cited_evidence(result: SearchResult) -> None:
    """Rewrite one result to show its corroborated evidence.

    The corroborated pairs replace the raw model output in `payload["evidence"]`,
    so the payload never carries an unchecked answer that looks authoritative.
    """
    payload = result.payload if result.payload is not None else {}
    evidence, text = cited_evidence_view(result.fields, payload.get("evidence"))
    payload["evidence"] = evidence
    # matching_chunks means what retrieval matched, which here was only the
    # generated author_profile; the cited quotes live in `evidence` instead.
    payload["matching_chunks"] = []
    payload["num_matching_chunks"] = 0
    result.payload = payload
    result.text = text


def _author_profile_verify_text(items: list[SearchResult]) -> list[str]:
    """Verification text for the author-profile strategy: the full submission EXCLUDING the
    LLM-generated author_profile field.
    """
    from .db_loader import extract_text
    from .llm_utils import GENERATED_FIELDS

    # The same set corroborated_evidence refuses to accept citations against.
    # These two rules are one rule — "the validator neither sees nor may cite
    # text this system wrote" — and they were separate literals, which is how
    # the citation side came to miss author_profile while this side hid it.
    return [
        extract_text({
            k: v for k, v in (r.fields or {}).items()
            if str(k).strip().casefold() not in GENERATED_FIELDS
        })
        for r in items
    ]


def _llm_validate_results(
    query: str,
    results: list[SearchResult],
    label: str,
    *,
    batch_size: int,
    max_doc_chars: int,
    max_results: int | None = LLM_VALIDATION_MAX_RESULTS,
    text_fn=None,
    max_workers: int = 1,
    validate_fn=None,
) -> tuple[list[SearchResult], list[float]]:
    """Run LLM validation.

    Returns:
        (kept_results, batch_times). Each kept result gets `llm_reason` on its payload.
    """
    from .llm_utils import run_llm_validation

    if text_fn is None:
        def text_fn(items: list[SearchResult]) -> list[str]:
            return [r.text or "" for r in items]

    validated_pairs, batch_times = run_llm_validation(
        query, results, text_fn=text_fn,
        batch_size=batch_size, max_results=max_results, label=label,
        max_workers=max_workers, max_doc_chars=max_doc_chars,
        validate_fn=validate_fn,
    )

    kept: list[SearchResult] = []
    for result, validation in validated_pairs:
        if result.payload is None:
            result.payload = {}
        result.payload["llm_reason"] = validation.get("reason", "")
        # Real-row evidence cited by the validator (author-profile strategy);
        # None for topical validators that don't emit it.
        result.payload["evidence"] = validation.get("evidence")
        # Verbatim query-relevant phrases the validator copied from the doc, for
        # in-place UI highlighting; [] for validators that don't emit them.
        result.payload["highlights"] = validation.get("highlights") or []
        kept.append(result)
    return kept, batch_times


class AuthorProfileLLMVespaStrategy(Paged):
    """Search ONLY the LLM-derived `author_profile` chunks (who the tipster
    is), then LLM-verify each candidate against the FULL parent row.
    """

    name = "Author Profile + LLM (Vespa)"

    # `top_k` is the recall knob: how many author_profile matches are retrieved
    # into the candidate pool. Verification cost is paged (PAGE_SIZE) in the UI and
    # capped by `llm_cap` in execute(), so a deep pool no longer means a deep LLM
    # sweep up front.
    # maxHits in vespa/search/query-profiles/default.xml caps how deep top_k can go.
    VALIDATION_WORKERS = 8      # concurrent LLM validation batches
    VALIDATION_BATCH_SIZE = 12  # balances request count against context size
    VALIDATION_MAX_DOC_CHARS = 20000  # bounds unusually large rows per batch

    def __init__(self, client: VespaClient):
        self.client = client
        self._atomic = AtomicVespaStrategy(client)

    @observe(name="Author Profile: retrieve")
    def retrieve_pool(self, query: str, top_k: int = 100) -> tuple[list[SearchResult], dict]:
        """Retrieve the author_profile candidate pool, UNVALIDATED."""
        retrieval = self._atomic.execute(
            query, top_k=top_k, target_hits=top_k, apply_cutoff=False, max_parents=top_k,
            extra_where='column_key contains "author_profile"',
            exclude_author_profile=False,  # this strategy matches ON the profile
        )
        return retrieval.results, {**retrieval.metadata, "pool_size": len(retrieval.results)}

    @observe(name="Author Profile: validate")
    def validate_page(self, query: str, pool_slice: list[SearchResult]) -> list[SearchResult]:
        """LLM-verify one slice of the pool against the real row content
        (excluding the LLM's own author_profile), parallelized.
        validate_author_profile_batch judges whether the SUBMITTER matches the
        query description, not just whether the topic is mentioned."""
        if not pool_slice:
            return []
        from .llm_utils import validate_author_profile_batch
        kept, _batch_times = _llm_validate_results(
            query, pool_slice, label="AuthorProfile",
            text_fn=_author_profile_verify_text,
            max_results=None, max_workers=self.VALIDATION_WORKERS,
            batch_size=self.VALIDATION_BATCH_SIZE,
            max_doc_chars=self.VALIDATION_MAX_DOC_CHARS,
            validate_fn=validate_author_profile_batch,
        )

        # Surface REAL row evidence instead of the AI-generated profile: the
        # field/quote pairs the validator cited from the actual submission, in
        # `evidence`. matching_chunks held only the author_profile chunk (retrieval
        # was column-restricted to it), so it is emptied. When the LLM cites
        # nothing, the chip renders "—" and the body falls back to llm_reason
        # (no synthetic-profile leak).
        for r in kept:
            apply_cited_evidence(r)

        return kept

    @observe(name="Author Profile + LLM (Vespa)")
    def execute(
        self, query: str, top_k: int = 100, llm_cap: int | None = None,
    ) -> StrategyResult:
        """One-shot: retrieve the pool and verify it (eval/API path). llm_cap=None verifies
        every candidate — the exhaustive find-all sweep this strategy is for.
        """
        start = time.time()
        pool, meta = self.retrieve_pool(query, top_k=top_k)
        sent = pool if llm_cap is None else pool[:llm_cap]
        kept = self.validate_page(query, sent)
        return StrategyResult(
            results=kept,
            elapsed_seconds=time.time() - start,
            approach_name=self.name,
            metadata={
                **meta,
                "before_llm": len(sent),
                "sent_to_llm": len(sent),
                "validated_count": len(kept),
            },
        )


# ----- HyDE / QueryExpansion (query-rewrite strategies) ----------------


class HyDEVespaStrategy:
    """Generate a hypothetical answer via LLM, retrieve with its embedding, rerank ColBERT
    against the original query.
    """

    name = "HyDE (Vespa)"

    def __init__(self, client: VespaClient):
        self.client = client
        self._atomic = AtomicVespaStrategy(client)

    @observe(name="HyDE (Vespa)")
    def execute(self, query: str, top_k: int = 100) -> StrategyResult:
        from .llm_utils import generate_hyde_document
        start = time.time()

        hyde_doc = generate_hyde_document(query)
        inner = self._atomic.execute(query, top_k=top_k, retrieval_query=hyde_doc)

        return StrategyResult(
            results=inner.results,
            elapsed_seconds=time.time() - start,
            approach_name=self.name,
            metadata={**inner.metadata, "hyde_document": hyde_doc},
        )


class QueryExpansionVespaStrategy:
    """Expand the query with synonyms via LLM, retrieve with the expanded form, rerank
    ColBERT against the original.
    """

    name = "Query Expansion (Vespa)"

    def __init__(self, client: VespaClient):
        self.client = client
        self._atomic = AtomicVespaStrategy(client)

    @observe(name="Query Expansion (Vespa)")
    def execute(self, query: str, top_k: int = 100) -> StrategyResult:
        from .llm_utils import expand_query
        start = time.time()

        expanded = expand_query(query)
        inner = self._atomic.execute(query, top_k=top_k, retrieval_query=expanded)

        return StrategyResult(
            results=inner.results,
            elapsed_seconds=time.time() - start,
            approach_name=self.name,
            metadata={**inner.metadata, "original_query": query, "expanded_query": expanded},
        )


# ----- Decomposition strategies ---------------------------------------


def _merge_by_best_score(
    per_query_results: list[list[SearchResult]],
) -> list[SearchResult]:
    """Merge results from N sub-queries: dedupe by doc_id (parent_id), keep the
    SearchResult with the highest score per parent, sort descending.
    """
    best: dict[str, SearchResult] = {}
    for results in per_query_results:
        for r in results:
            existing = best.get(r.doc_id)
            if existing is None or r.score > existing.score:
                best[r.doc_id] = r
    return sorted(best.values(), key=lambda r: r.score, reverse=True)


class AtomicDecompVespaStrategy:
    """Decompose query, run AtomicVespa on each sub-query, merge by best score."""

    name = "Atomic + Decomp (Vespa)"

    def __init__(self, client: VespaClient):
        self.client = client
        self._atomic = AtomicVespaStrategy(client)

    @observe(name="Atomic + Decomp (Vespa)")
    def execute(self, query: str, top_k: int = 100) -> StrategyResult:
        from .llm_utils import decompose_query
        start = time.time()

        sub_queries = decompose_query(query)
        logger.info(f"[{self.name}] decomposed into {len(sub_queries)} sub-queries: {sub_queries}")

        per_query: list[list[SearchResult]] = []
        per_query_meta = []
        for sq in sub_queries:
            r = self._atomic.execute(sq, top_k=top_k)
            per_query.append(r.results)
            per_query_meta.append({"sub_query": sq, "count": len(r.results)})

        merged = _merge_by_best_score(per_query)

        return StrategyResult(
            results=merged,
            elapsed_seconds=time.time() - start,
            approach_name=self.name,
            metadata={
                "sub_queries": sub_queries,
                "per_query": per_query_meta,
                "total_unique": len(merged),
            },
        )


# ----- Tiered ----------------------------------------------------------


# ----- Combined: union of retrieval lanes → ColBERT rerank → LLM -------------


# Upper bound on the chunk ids sent to one tail-rerank query. Hit often enough
# on find-all sweeps that truncation is logged rather than silent — a silent cap
# reads as full coverage.
_TAIL_RERANK_MAX_IDS = 1000


def _colbert_scores_for_chunks(
    client: VespaClient, query: str, chunk_ids: list[str]
) -> dict[str, float]:
    """Best ColBERT-MaxSim(original query) score per parent, for a specific set of chunk
    ids.
    """
    if not chunk_ids:
        return {}
    if len(chunk_ids) > _TAIL_RERANK_MAX_IDS:
        logger.warning(
            f"[Content] tail rerank capped at {_TAIL_RERANK_MAX_IDS} of "
            f"{len(chunk_ids)} chunk ids; {len(chunk_ids) - _TAIL_RERANK_MAX_IDS} "
            "will fall back to the comparable-score floor"
        )
        chunk_ids = chunk_ids[:_TAIL_RERANK_MAX_IDS]
    try:
        ids = ", ".join(f'"{quote_yql_string(c)}"' for c in chunk_ids)
        yql = f"select chunk_id, parent_id, column_key from chunk where chunk_id in ({ids})"
        q_safe = _escape_embed_text(query)
        # colbert_by_id scores MaxSim in FIRST phase. Under the `hybrid` profile
        # this query scored every hit identically (no userInput means bm25 is 0
        # and there is no ANN leg), so second-phase rerank-count reranked an
        # arbitrary subset and the rest carried a meaningless score. No q_dense:
        # this profile has no dense term, so embedding the query twice was waste.
        resp = client.query(
            yql=yql, ranking="colbert_by_id", hits=len(chunk_ids), timeout="20s",
            q_colbert=f'embed(colbert, "{q_safe}")',
        )
    except Exception as e:
        logger.warning(f"[Content] tail ColBERT rerank failed: {e}")
        return {}
    # colbert_by_id scores MaxSim in the FIRST phase, so that is the feature
    # to read here — the narrower reader, because this response carries ids
    # and scores and no text to build chunks from.
    return best_score_by_parent(resp, score_feature="firstPhase")


class CombinedVespaStrategy(Paged):
    """Recall-first CONTENT search (for "what the tip is about" queries — the
    author/submitter case has its own strategy, AuthorProfileLLMVespaStrategy).
    """

    name = "Content (Vespa)"

    # Lanes whose per-parent score is ColBERT-MaxSim vs the ORIGINAL query,
    # hence comparable across lanes and reusable without reranking.
    _COMPARABLE_LANES = ("atomic", "hyde", "query_expansion")

    VALIDATION_WORKERS = 8
    VALIDATION_BATCH_SIZE = 12
    VALIDATION_MAX_DOC_CHARS = 20000

    def __init__(self, client: VespaClient):
        self.client = client
        # Retrieval-only content lanes. Author Profile is intentionally excluded
        # — it's the separate submitter-intent strategy, not a content lane.
        # No bm25 lane. Measured over 21 real queries: it appeared 616 times in
        # the LLM's first page but was the ONLY lane to find just 4 results —
        # 0.19 per query. Every lane below already ORs userInput() with the ANN
        # leg, so lexical matching is not lost; what the bm25 lane added was
        # results ranked by BM25 alone, and _colbert_scores_for_chunks re-scored
        # those onto the ColBERT scale anyway, sinking them.
        #
        # Its remaining justification was catching matches outside ColBERT's
        # 256-token window on long chunks. Token-aware chunking removes that.
        #
        # BM25VespaStrategy itself stays — it is a separate strategy in the
        # dashboard for keyword-intent searches, which is a different job.
        self._lanes = {
            "atomic": AtomicVespaStrategy(client),
            "hyde": HyDEVespaStrategy(client),
            "query_expansion": QueryExpansionVespaStrategy(client),
            "atomic_decomp": AtomicDecompVespaStrategy(client),
        }

    @observe(name="Content: retrieve")
    def retrieve_pool(self, query: str, top_k: int = 100) -> tuple[list[SearchResult], dict]:
        """Run all lanes, union to parent level, put everything on one comparable
        ColBERT-vs-original scale, drop content-less noise, and return the ranked but
        UNVALIDATED candidate pool plus metadata.
        """
        from concurrent.futures import ThreadPoolExecutor
        from opentelemetry import context as _otel_ctx

        # Attach the current span context in each worker so lane @observe spans
        # nest under this trace (worker threads don't inherit contextvars).
        _parent_ctx = _otel_ctx.get_current()

        def _run(lane_name: str):
            token = _otel_ctx.attach(_parent_ctx)
            try:
                res = self._lanes[lane_name].execute(query=query, top_k=top_k)
                return lane_name, res.results
            except Exception as e:
                logger.warning(f"[Content] lane {lane_name} failed: {e}")
                return lane_name, []
            finally:
                _otel_ctx.detach(token)

        with ThreadPoolExecutor(max_workers=len(self._lanes)) as ex:
            lane_results = dict(ex.map(_run, list(self._lanes.keys())))

        pool = lane_union.rank(
            lane_results,
            comparable_lanes=self._COMPARABLE_LANES,
            rescore=lambda chunk_ids: _colbert_scores_for_chunks(
                self.client, query, chunk_ids
            ),
        )
        return pool.results, pool.metadata

    @observe(name="Content: validate")
    def validate_page(self, query: str, pool_slice: list[SearchResult]) -> list[SearchResult]:
        """LLM-validate one slice of the retrieved pool for precision.

        Returns:
            The kept results, most-confident first.
        """
        if not pool_slice:
            return []
        kept, _batch_times = _llm_validate_results(
            query, pool_slice, label="Content",
            max_results=None, max_workers=self.VALIDATION_WORKERS,
            batch_size=self.VALIDATION_BATCH_SIZE,
            max_doc_chars=self.VALIDATION_MAX_DOC_CHARS,
        )
        # Validation filters; it does not reorder. The pool arrives on the
        # comparable ColBERT scale (see retrieve_pool) and keeps that order.
        #
        # There used to be a confidence-first sort here, but no prompt ever
        # asked the model for a confidence, so every row tied at 0.0 and the
        # sort was already this. Restoring it means adding the field to the
        # validator specs AND measuring the reordering — a relevance change,
        # which this repo measures rather than assumes.
        return kept

    @observe(name="Content (Vespa)")
    def execute(
        self, query: str, top_k: int = 100,
        llm_cap: int = LLM_VALIDATION_MAX_RESULTS,
    ) -> StrategyResult:
        """One-shot: retrieve the pool and validate its top `llm_cap` (eval/API path)."""
        start = time.time()
        pool, meta = self.retrieve_pool(query, top_k=top_k)
        kept = self.validate_page(query, pool[:llm_cap])
        return StrategyResult(
            results=kept,
            elapsed_seconds=time.time() - start,
            approach_name=self.name,
            metadata={
                **meta,
                "sent_to_llm": min(len(pool), llm_cap),
                "validated_count": len(kept),
            },
        )


# --- Strategy registry -------------------------------------------------------
# The single source of truth for the selectable strategies, shared by the
# Streamlit UI (re_source/search_ui.py) and the FastAPI runner (re_source/search_runner.py).
# All entries construct as `Cls(client)` and run as `.execute(query=, top_k=)`.
VESPA_STRATEGIES: dict[str, type] = {
    "Content": CombinedVespaStrategy,
    "BM25": BM25VespaStrategy,
    "Author Profile + LLM": AuthorProfileLLMVespaStrategy,
}

# The strategy a caller gets when it doesn't pick one. Derived rather than spelled
# out a second time: the API's default used to be a literal, and it named a
# strategy the registry had stopped holding, so every request that omitted the
# field got a 400.
DEFAULT_STRATEGY = next(iter(VESPA_STRATEGIES))
