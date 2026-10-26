import pytest

from opspilot.retrieval import EmbeddingRetriever, load_policy_chunks, tokenize


def test_policies_are_split_into_sections(settings):
    chunks = load_policy_chunks(settings.policies_dir)

    assert len(chunks) >= 25
    sources = {c.source for c in chunks}
    assert {"leave-policy.md", "it-security-policy.md", "expenses-policy.md"} <= sources
    assert all(c.text and c.section for c in chunks)


def test_tokenize_drops_stopwords():
    assert tokenize("How many days of Annual Leave do I get?") == ["many", "days", "annual", "leave", "get"]


@pytest.mark.parametrize(
    ("query", "expected_section"),
    [
        ("How many days of annual leave do I get?", "Annual leave entitlement"),
        ("Can I carry over unused holiday to next year?", "Carrying over leave"),
        ("What is the minimum password length?", "Passwords"),
        ("I lost my laptop, what should I do?", "Lost or stolen devices"),
        ("How quickly will IT respond to a P2 ticket?", "Ticket priorities and response times"),
        ("What is the mileage rate for using my own car?", "Mileage"),
    ],
)
def test_bm25_finds_the_right_section(retriever, query, expected_section):
    results = retriever.search(query, k=3)

    assert expected_section in [r.chunk.section for r in results]


def test_embedding_retriever_ranks_by_cosine_similarity(settings):
    chunks = load_policy_chunks(settings.policies_dir)[:3]
    vectors = {c.text: [float(i == j) for j in range(3)] for i, c in enumerate(chunks)}
    retriever = EmbeddingRetriever(
        chunks,
        embed_documents=lambda texts: [vectors[t] for t in texts],
        embed_query=lambda texts: [[0.0, 1.0, 0.1]],
    )

    assert retriever.search("anything", k=1)[0].chunk == chunks[1]
