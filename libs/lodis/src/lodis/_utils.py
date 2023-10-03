from multiprocessing import resource_tracker


def unregister_shared_memory_tracking(memory_name: str) -> None:
    """
    Tell Python to not be so clever about automatically deallocating this memory.
    See: https://stackoverflow.com/a/73159334
    """
    resource_tracker.unregister(name=f"/{memory_name}", rtype="shared_memory")
