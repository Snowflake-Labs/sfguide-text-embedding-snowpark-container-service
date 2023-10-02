import pytest
from simple_lru_cache import LRUCache


def test():
    cache = LRUCache(maxsize=2)
    assert len(cache) == 0
    assert "a" not in cache

    # Add one item.
    cache["a"] = 1
    assert "a" in cache
    assert cache["a"] == 1
    assert len(cache) == 1
    with pytest.raises(KeyError):
        cache["b"]

    # Add a second item.
    cache["b"] = 2
    assert len(cache) == 2
    assert cache["a"] == 1
    assert cache["b"] == 2

    # Add a third item, which should delete the first.
    cache["c"] = 3
    assert len(cache) == 2
    with pytest.raises(KeyError):
        cache["a"]
    assert cache["b"] == 2
    assert cache["c"] == 3

    # Test `.get()`.
    assert cache.get("a") is None
    assert cache.get("b") == 2

    # Test clear.
    cache.clear()
    assert len(cache) == 0
    assert cache.get("a") is None
    assert cache.get("b") is None
    assert cache.get("c") is None


def test_updates():
    cache = LRUCache(maxsize=2)

    # Updates should update position to most recently touched.
    cache["a"] = True
    cache["b"] = True
    cache["a"] = True
    cache["c"] = True
    assert "a" in cache
    assert "b" not in cache

    # Updates should affect value.
    cache["a"] = False
    assert cache.get("a") == False
