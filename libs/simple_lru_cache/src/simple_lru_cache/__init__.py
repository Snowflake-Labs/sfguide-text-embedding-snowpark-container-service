# Adapted from the source code of `functools.lru_cache`.
from threading import RLock
from typing import Any
from typing import Dict
from typing import Hashable
from typing import List

# Python typing doesn't really support type annotating this kind of specialized
# use of a list yet, so the best we have is `List[Any]` with a bunch of `type: ignore`
# flags below.
# The ideal would be something like a non-tuple version of:
# `LinkedListNode = Tuple["LinkedListNode", "LinkedListNode", Hashable, Any]`
LinkedListNode = List[Any]


# Names for the link fields in the doubly-linked list structure.
PREV, NEXT, KEY, VALUE = 0, 1, 2, 3


class LRUCache:
    def __init__(self, maxsize: int) -> None:
        assert maxsize > 0
        self.maxsize = maxsize
        self.full = False
        self.lock = RLock()  # because linkedlist updates aren't threadsafe
        self.cache: Dict[Hashable, LinkedListNode] = {}
        self.root: LinkedListNode = []  # root of the circular doubly linked list
        # initialize by pointing to self
        self.root[:] = [self.root, self.root, None, None]

    def _bump_link(self, link: LinkedListNode):
        """Moves a particular link to the most-recently-used position."""
        link_prev, link_next, _key, _result = link
        link_prev[NEXT] = link_next
        link_next[PREV] = link_prev
        last = self.root[PREV]
        last[NEXT] = self.root[PREV] = link  # type: ignore
        link[PREV] = last
        link[NEXT] = self.root

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, key: Hashable) -> bool:
        return key in self.cache

    def __getitem__(self, key: Hashable) -> Any:
        with self.lock:
            link = self.cache.get(key)

            # Check if key not in the cache.
            if link is None:
                raise KeyError(key)

            # Move the link to the front of the circular queue
            self._bump_link(link)

            # Return the result.
            return link[VALUE]

    def __setitem__(self, key: Hashable, value: Any) -> None:
        with self.lock:
            link = self.cache.get(key)
            # Key in cache, perform an update.
            if link is not None:
                # Move item to the top of the queue
                self._bump_link(link)

                # Update link value.
                link[VALUE] = value
            # Key not in cache, perform an insert.
            else:
                if self.full:
                    # Use the old root to store the new key and result.
                    oldroot = self.root
                    oldroot[KEY] = key  # type: ignore
                    oldroot[VALUE] = value  # type: ignore

                    # Empty the oldest link and make it the new root.
                    # Keep a reference to the old key and old result to
                    # prevent their ref counts from going to zero during the
                    # update. That will prevent potentially arbitrary object
                    # clean-up code (i.e. __del__) from running while we're
                    # still adjusting the links.
                    root = oldroot[NEXT]
                    oldkey = root[KEY]
                    root[VALUE]
                    root[KEY] = root[VALUE] = None

                    # Now update the cache dictionary.
                    del self.cache[oldkey]

                    # Save the potentially reentrant cache[key] assignment
                    # for last, after the root and links have been put in
                    # a consistent state.
                    self.cache[key] = oldroot
                else:
                    # Put result in a new link at the front of the queue.
                    last = self.root[PREV]
                    link = [last, self.root, key, value]
                    last[NEXT] = self.root[PREV] = self.cache[key] = link
                    self.full = len(self.cache) >= self.maxsize

    def __delitem__(self, key: Hashable) -> None:
        raise NotImplementedError

    def get(self, key: Hashable, default=None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.root[:] = [self.root, self.root, None, None]
            self.full = False
