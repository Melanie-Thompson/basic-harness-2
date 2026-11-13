"""Tests for TDA-based context management.

Exercises clustering, scoring, budget packing, archival, and rehydration
using a synthetic multi-topic conversation corpus against real Redis.
"""

import random
from collections import Counter

import numpy as np
import pytest
from langchain_core.messages import HumanMessage

from AI import redis_store
from AI.context import ContextManager, REHYDRATION_THRESHOLD
from AI.embeddings import embed, semantic_similarity
from AI.tokenizer import count_tokens, count_messages


# ── Corpus generator ────────────────────────────────────────────────────────

CLUSTERS = {
    "backend": [
        "We need to optimize the FastAPI async dependency injection loop.",
        "Pydantic v2 validation models are throwing serialization overhead warnings.",
        "Let's move the MongoDB connection pooling logic to motor async client.",
        "Are we handling CORS middleware headers correctly in the gateway?",
        "The Redis cache TTL needs to be bumped to 3600 seconds for user sessions.",
    ],
    "frontend": [
        "The service worker isn't caching the PWA asset manifest properly offline.",
        "IndexedDB outbox sync is failing when network reconnects abruptly.",
        "We should use Zustand for client-side state management instead of Redux.",
        "Tailwind responsive grid classes are breaking on mobile viewports.",
        "Did you test the Radix UI accessibility attributes on the modal dialog?",
    ],
    "art": [
        "Adjusting the vector line weight curve in Clip Studio makes ink lines much more organic.",
        "How do you handle screentone halftoning in digital manga panels?",
        "The 3D mannequin posing tool is saving so much time on perspective foreshortening.",
        "Exporting layers to PSD is messing up the blending mode opacity.",
        "Working on the cyberpunk character design sketch tonight.",
    ],
    "cooking": [
        "Need to check the pantry inventory for chickpea flour and coconut milk.",
        "The sourdough starter is bubbling nicely after day four of feeding.",
        "Let's make roasted sweet potato and peanut butter curry for dinner.",
        "Did we run out of olive oil and flakey sea salt?",
        "Meal prep containers are washed and ready for the week.",
    ],
    "noise": [
        "Did anyone catch the football match last night?",
        "The weather forecast says heavy rain all weekend.",
        "Coffee machine in the breakroom is acting up again.",
        "Has anyone seen where my wireless mouse dongle went?",
        "That new sci-fi series trailer dropped on YouTube.",
    ],
}


def generate_corpus(
    total: int = 200, seed: int = 42
) -> tuple[list[HumanMessage], list[str]]:
    rng = random.Random(seed)
    cluster_keys = list(CLUSTERS.keys())
    messages: list[HumanMessage] = []
    labels: list[str] = []
    active = rng.choice(cluster_keys)

    for i in range(total):
        if i % rng.randint(3, 8) == 0:
            active = rng.choice(cluster_keys)
        elif rng.random() > 0.85:
            active = rng.choice(cluster_keys)

        template = rng.choice(CLUSTERS[active])
        text = f"{template} (Ref: {i + 1})"
        messages.append(HumanMessage(content=text))
        labels.append(active)

    return messages, labels


# ── Redis fixture ──────────────────────────────────────────────────────────

TEST_SESSION = "test-context-run"


@pytest.fixture(autouse=True)
async def redis():
    await redis_store.connect()
    print("\n  [redis] connected")
    await redis_store.set_clusters(TEST_SESSION, "[]")
    yield
    await redis_store.close()
    print("  [redis] disconnected")


@pytest.fixture(scope="module")
def corpus():
    msgs, labels = generate_corpus(200)
    dist = Counter(labels)
    print(f"\n  [corpus] generated 200 messages: {dict(dist)}")
    return msgs, labels


@pytest.fixture(scope="module")
def small_corpus():
    msgs, labels = generate_corpus(30, seed=99)
    print(f"\n  [corpus] generated 30 messages (small)")
    return msgs, labels


# ── Embedding tests ─────────────────────────────────────────────────────────

class TestEmbeddings:
    def test_embed_returns_correct_shape(self):
        vecs = embed(["hello world", "foo bar"])
        print(f"    embed shape: {vecs.shape}")
        assert vecs.shape == (2, 384)

    def test_same_text_gives_high_similarity(self):
        score = semantic_similarity(
            "The sourdough starter is ready",
            "The sourdough starter is ready",
        )
        print(f"    identical text similarity: {score:.4f}")
        assert score > 0.99

    def test_same_topic_scores_higher_than_cross_topic(self):
        backend_a = "FastAPI async dependency injection optimization"
        backend_b = "Pydantic v2 serialization overhead warnings"
        cooking = "Sourdough starter bubbling after day four"

        same = semantic_similarity(backend_a, backend_b)
        cross = semantic_similarity(backend_a, cooking)
        print(f"    same-topic (backend/backend): {same:.4f}")
        print(f"    cross-topic (backend/cooking): {cross:.4f}")
        assert same > cross


# ── Tokenizer tests ─────────────────────────────────────────────────────────

class TestTokenizer:
    def test_count_tokens_nonzero(self):
        n = count_tokens("Hello, world!")
        print(f"    'Hello, world!' = {n} tokens")
        assert n > 0

    def test_count_messages(self, small_corpus):
        msgs, _ = small_corpus
        total = count_messages(msgs)
        print(f"    30 messages = {total} tokens")
        assert total > 0
        assert total > len(msgs)


# ── Clustering tests ────────────────────────────────────────────────────────

class TestClustering:
    def test_produces_multiple_clusters(self, corpus):
        msgs, _ = corpus
        vecs = embed([m.content for m in msgs])

        cm = ContextManager.__new__(ContextManager)
        clusters = cm._cluster(vecs)

        sizes = [len(c) for c in clusters]
        print(f"    found {len(clusters)} clusters, sizes: {sorted(sizes, reverse=True)}")
        assert len(clusters) > 1
        assert sum(sizes) == len(msgs)

    def test_cluster_purity(self, corpus):
        msgs, labels = corpus
        vecs = embed([m.content for m in msgs])

        cm = ContextManager.__new__(ContextManager)
        clusters = cm._cluster(vecs)

        purities = []
        for i, members in enumerate(clusters):
            if len(members) < 3:
                continue
            member_labels = [labels[j] for j in members]
            dist = Counter(member_labels)
            dominant_label, dominant_count = dist.most_common(1)[0]
            purity = dominant_count / len(members)
            purities.append(purity)
            print(f"    cluster {i}: {len(members)} msgs, dominant='{dominant_label}' ({purity:.0%}), {dict(dist)}")

        avg_purity = sum(purities) / len(purities) if purities else 0
        print(f"    average purity: {avg_purity:.2%}")
        assert avg_purity > 0.4

    def test_single_message_returns_one_cluster(self):
        vecs = embed(["just one message"])
        cm = ContextManager.__new__(ContextManager)
        clusters = cm._cluster(vecs)
        print(f"    single message -> {len(clusters)} cluster(s)")
        assert clusters == [[0]]


# ── Scoring tests ───────────────────────────────────────────────────────────

class TestScoring:
    def test_relevant_cluster_scores_highest(self, corpus):
        msgs, labels = corpus
        vecs = embed([m.content for m in msgs])

        cm = ContextManager.__new__(ContextManager)
        clusters = cm._cluster(vecs)

        query = "What's the status of the FastAPI backend optimization?"
        query_vec = embed([query])[0]
        scored = cm._score_clusters(clusters, vecs, query_vec)

        print(f"    query: '{query}'")
        for rank, (score, members) in enumerate(scored[:5]):
            member_labels = [labels[i] for i in members]
            dominant = Counter(member_labels).most_common(1)[0][0]
            print(f"    #{rank + 1} score={score:.4f} dominant='{dominant}' ({len(members)} msgs)")

        top_labels = [labels[i] for i in scored[0][1]]
        dominant = Counter(top_labels).most_common(1)[0][0]
        assert dominant == "backend"

    def test_scores_are_sorted_descending(self, corpus):
        msgs, _ = corpus
        vecs = embed([m.content for m in msgs])
        cm = ContextManager.__new__(ContextManager)
        clusters = cm._cluster(vecs)
        query_vec = embed(["meal prep and cooking"])[0]
        scored = cm._score_clusters(clusters, vecs, query_vec)
        scores = [s for s, _ in scored]
        print(f"    score range: {scores[0]:.4f} -> {scores[-1]:.4f}")
        assert scores == sorted(scores, reverse=True)


# ── Budget packing tests ───────────────────────────────────────────────────

class TestPacking:
    def test_packing_respects_budget(self, small_corpus):
        msgs, _ = small_corpus
        vecs = embed([m.content for m in msgs])

        cm = ContextManager.__new__(ContextManager)
        cm.max_tokens = 200
        clusters = cm._cluster(vecs)
        query_vec = embed(["backend"])[0]
        scored = cm._score_clusters(clusters, vecs, query_vec)

        selected, overflow = cm._pack(scored, msgs)
        sel_tokens = count_messages(selected)

        print(f"    budget: 200 tokens")
        print(f"    selected: {len(selected)} msgs ({sel_tokens} tokens)")
        print(f"    overflow: {len(overflow)} clusters")
        assert len(selected) < len(msgs)
        assert len(overflow) > 0
        assert sel_tokens <= 200

    def test_large_budget_keeps_everything(self, small_corpus):
        msgs, _ = small_corpus
        vecs = embed([m.content for m in msgs])

        cm = ContextManager.__new__(ContextManager)
        cm.max_tokens = 999_999
        clusters = cm._cluster(vecs)
        query_vec = embed(["anything"])[0]
        scored = cm._score_clusters(clusters, vecs, query_vec)

        selected, overflow = cm._pack(scored, msgs)
        print(f"    budget: 999999 -> selected all {len(selected)} msgs, {len(overflow)} overflow")
        assert len(selected) == len(msgs)
        assert len(overflow) == 0


# ── Integration: prepare() with real Redis ─────────────────────────────────

class TestPrepareIntegration:

    async def test_under_budget_returns_unchanged(self, redis, small_corpus):
        msgs, _ = small_corpus
        cm = ContextManager(TEST_SESSION, max_tokens=999_999)
        result = await cm.prepare(msgs, "anything")
        total = count_messages(msgs)
        print(f"    {total} tokens vs 999999 budget -> returned unchanged (identity={result is msgs})")
        assert result is msgs

    async def test_over_budget_prunes_and_archives(self, redis):
        # Clear any prior state
        await redis_store.set_clusters(TEST_SESSION, "[]")

        msgs, labels = generate_corpus(200, seed=77)
        total_tokens = count_messages(msgs)
        print(f"    input: {len(msgs)} messages, {total_tokens} tokens")
        print(f"    budget: 500 tokens")

        cm = ContextManager(TEST_SESSION, max_tokens=500)
        result = await cm.prepare(msgs, "Tell me about the backend API")

        result_tokens = count_messages(result)
        print(f"    result: {len(result)} messages, {result_tokens} tokens")

        # Check what's in Redis
        raw = await redis_store.get_clusters(TEST_SESSION)
        assert raw is not None, "overflow should be archived to Redis"

        import json
        archived = json.loads(raw)
        total_archived_msgs = sum(len(c["messages"]) for c in archived)
        print(f"    [redis] archived: {len(archived)} clusters, {total_archived_msgs} messages")
        for i, c in enumerate(archived):
            print(f"      cluster {i}: {len(c['messages'])} msgs, {c['token_count']} tokens")

        assert len(result) < len(msgs)
        assert result_tokens <= 500

    async def test_rehydration_pulls_relevant_archived_cluster(self, redis):
        # Clear state
        await redis_store.set_clusters(TEST_SESSION, "[]")

        msgs, labels = generate_corpus(200, seed=55)
        total_tokens = count_messages(msgs)

        # Pass 1: cooking query — archives most clusters
        print(f"\n    === Pass 1: cooking query ===")
        print(f"    input: {len(msgs)} messages, {total_tokens} tokens, budget=500")
        cm = ContextManager(TEST_SESSION, max_tokens=500)
        result1 = await cm.prepare(msgs, "How is the sourdough starter doing?")

        result1_labels = []
        for m in result1:
            if m in msgs:
                result1_labels.append(labels[msgs.index(m)])
        print(f"    selected: {len(result1)} messages — {dict(Counter(result1_labels))}")

        raw = await redis_store.get_clusters(TEST_SESSION)
        import json
        archived = json.loads(raw)
        print(f"    [redis] archived: {len(archived)} clusters")

        # Pass 2: feed full corpus again with backend query
        print(f"\n    === Pass 2: backend query (should rehydrate) ===")
        cm2 = ContextManager(TEST_SESSION, max_tokens=500)
        result2 = await cm2.prepare(msgs, "Tell me about FastAPI dependency injection")

        result2_labels = []
        for m in result2:
            if m in msgs:
                result2_labels.append(labels[msgs.index(m)])
        print(f"    selected: {len(result2)} messages — {dict(Counter(result2_labels))}")

        result2_text = " ".join(m.content for m in result2)
        has_backend = any(
            kw in result2_text.lower()
            for kw in ["fastapi", "pydantic", "mongodb", "cors", "redis cache"]
        )
        print(f"    contains backend content: {has_backend}")

        # Check Redis state after rehydration
        raw2 = await redis_store.get_clusters(TEST_SESSION)
        archived2 = json.loads(raw2)
        print(f"    [redis] remaining archived: {len(archived2)} clusters")

        assert has_backend

    async def test_empty_history_returns_empty(self, redis):
        cm = ContextManager(TEST_SESSION)
        result = await cm.prepare([], "hello")
        print(f"    empty input -> {len(result)} messages returned")
        assert result == []

    async def test_message_order_preserved(self, redis):
        await redis_store.set_clusters(TEST_SESSION, "[]")

        msgs, _ = generate_corpus(100, seed=33)
        cm = ContextManager(TEST_SESSION, max_tokens=800)
        result = await cm.prepare(msgs, "backend optimization")

        original_indices = []
        for m in result:
            if m in msgs:
                original_indices.append(msgs.index(m))

        is_sorted = original_indices == sorted(original_indices)
        print(f"    {len(result)} messages selected, order preserved: {is_sorted}")
        print(f"    index sample (first 10): {original_indices[:10]}")
        assert is_sorted
