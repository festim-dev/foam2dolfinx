from importlib import metadata

try:
    __version__ = metadata.version("foam2dolfinx")
except Exception:
    __version__ = "unknown"


from .open_foam_reader import OpenFOAMReader, find_closest_value

__all__ = ["OpenFOAMReader, find_closest_value"]
