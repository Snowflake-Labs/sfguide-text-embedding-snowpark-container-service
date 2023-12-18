# Attempting to use Linux libc semaphores as a multi-process lock.
from contextlib import contextmanager
from ctypes import c_int
from ctypes import c_long
from ctypes import c_uint
from ctypes import c_void_p
from ctypes import cast
from ctypes import cdll
from ctypes import create_string_buffer
from ctypes import POINTER
from ctypes import pointer
from typing import Iterator
from typing import NewType

# HACK: We call it a pointer to an int, but in reality it's a pointer to a struct.
SemaphoreMemoryAddress = NewType(  # type: ignore
    "SemaphoreMemoryAddress", POINTER(c_int)
)

LIBRT = None
for librt_name in ("librt.so", "librt.so.1", "libc.so"):
    try:
        LIBRT = cdll.LoadLibrary(librt_name)
        break
    except OSError:
        pass
if LIBRT is None:
    raise RuntimeError(
        "Could not find the `librt` shared library. Maybe you're trying to run on a non-Linux OS? "
        "If so, unfortunately `lodis` does not support that yet."
    )
LIBRT.sem_open.restype = POINTER(c_int)  # Cannot be `SemaphoreMemoryAddress`
LIBRT.sem_trywait.restype = c_int
LIBRT.sem_wait.restype = c_int
LIBRT.sem_post.restype = c_int
LIBRT.sem_close.restype = c_int
LIBRT.sem_unlink.restype = c_int
LIBRT.sem_getvalue.restype = c_int

# https://github.com/torvalds/linux/blob/master/include/uapi/asm-generic/fcntl.h
O_CREAT = c_long(0o00000100)
O_EXCL = c_long(0o00000200)
O_CREAT_AND_EXCL = c_long(O_CREAT.value | O_EXCL.value)
O_EMPTY_FLAG = c_long(0o00000000)
PERMISSION_MODE = c_int(0o0644)


def _get_name_arg(name: str):
    if not name.startswith("/"):
        raise ValueError(f"Invalid name `{name}`. Semaphore names start with a slash.")
    return create_string_buffer(bytes(name, encoding="utf-8"))


def allocate(name: str, value: int = 1) -> None:
    memory_address_pointer = LIBRT.sem_open(
        _get_name_arg(name), O_CREAT_AND_EXCL, PERMISSION_MODE, c_uint(value)
    )
    is_null_pointer = cast(memory_address_pointer, c_void_p).value is None
    if is_null_pointer:
        raise RuntimeError(
            f"Unable to create semaphore {name}. Perhaps it already exists?"
        )
    # To separate allocation from opening, we close the semaphore here.
    close(memory_address_pointer)


def open(name: str) -> SemaphoreMemoryAddress:
    memory_address = LIBRT.sem_open(_get_name_arg(name), O_EMPTY_FLAG)
    if not memory_address:
        raise KeyError(name)
    return memory_address


def acquire(semaphore: SemaphoreMemoryAddress, blocking: bool = True) -> bool:
    fn = LIBRT.sem_wait if blocking else LIBRT.sem_trywait
    result = fn(semaphore)
    return result == 0


def release(semaphore: SemaphoreMemoryAddress) -> None:
    result = LIBRT.sem_post(semaphore)
    if result != 0:
        raise RuntimeError("Release failed")


def getvalue(semaphore: SemaphoreMemoryAddress) -> int:
    value_c_int = c_int()
    result = LIBRT.sem_getvalue(semaphore, pointer(value_c_int))
    if result != 0:
        raise RuntimeError("Getvalue failed")
    return value_c_int.value


def close(semaphore: SemaphoreMemoryAddress) -> None:
    result = LIBRT.sem_close(semaphore)
    if result != 0:
        raise RuntimeError("Close failed")


def unlink(name: str) -> None:
    result = LIBRT.sem_unlink(_get_name_arg(name))
    if result != 0:
        raise RuntimeError("Unlink failed")


@contextmanager
def acquire_context(semaphore: SemaphoreMemoryAddress) -> Iterator[None]:
    acquire(semaphore=semaphore, blocking=True)
    try:
        yield
    finally:
        release(semaphore=semaphore)
