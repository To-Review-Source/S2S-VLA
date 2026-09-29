from s2s_vla.interfaces import Runtime


def create_runtime(method, **kwargs) -> Runtime:
    raise NotImplementedError("Provide a frozen VLA policy adapter and a platform environment adapter. See docs/ADAPTERS.md.")
