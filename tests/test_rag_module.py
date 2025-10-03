from src.rag import example_queries


def test_example_queries_length():
    queries = example_queries()
    assert isinstance(queries, list)
    assert len(queries) >= 5
