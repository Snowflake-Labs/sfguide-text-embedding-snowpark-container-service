import pytest
from _utils import _ensure_semaphore_not_allocated
from lodis import librt_semaphore


def test_create_runtime_error():
    with pytest.raises(RuntimeError):
        librt_semaphore.allocate("/invalid/name/with/slashes")


def test_open_key_error():
    with pytest.raises(KeyError):
        librt_semaphore.open("/nonexistent")


def test_create_and_open():
    _ensure_semaphore_not_allocated("/my_lock")
    librt_semaphore.allocate("/my_lock")
    s = librt_semaphore.open("/my_lock")

    # Re-creating an existing lock gives an error.
    with pytest.raises(RuntimeError):
        librt_semaphore.allocate("/my_lock")

    assert librt_semaphore.getvalue(s) == 1
    s2 = librt_semaphore.open("/my_lock")
    assert librt_semaphore.getvalue(s) == 1

    # Check pointer equality.
    assert s[0] == s2[0]

    # Tear down.
    librt_semaphore.close(s)
    librt_semaphore.close(s2)
    librt_semaphore.unlink("/my_lock")


def test_as_lock():
    _ensure_semaphore_not_allocated("/my_lock")
    librt_semaphore.allocate("/my_lock")
    s = librt_semaphore.open("/my_lock")
    s2 = librt_semaphore.open("/my_lock")

    # Acquire the lock and assert other acquisition attempts fail.
    with librt_semaphore.acquire_context(s):
        s_acq_attempt_result = librt_semaphore.acquire(s, blocking=False)
        s2_acq_attempt_result = librt_semaphore.acquire(s2, blocking=False)
        assert not s_acq_attempt_result
        assert not s2_acq_attempt_result
        assert librt_semaphore.getvalue(s) == 0
        assert librt_semaphore.getvalue(s2) == 0

    # Acquire out of the context to ensure the lock has been released.
    assert librt_semaphore.getvalue(s) == 1
    s2_acq_attempt_result = librt_semaphore.acquire(s2, blocking=False)
    assert librt_semaphore.getvalue(s) == 0
    assert s2_acq_attempt_result

    # Assert acquisition now fails for the s.
    s_acq_attempt_result = librt_semaphore.acquire(s, blocking=False)
    assert not s_acq_attempt_result

    # Ensure releasing works.
    librt_semaphore.release(s)
    s_acq_attempt_result = librt_semaphore.acquire(s, blocking=False)
    assert s_acq_attempt_result

    # Tear down.
    librt_semaphore.close(s)
    librt_semaphore.close(s2)
    librt_semaphore.unlink("/my_lock")
