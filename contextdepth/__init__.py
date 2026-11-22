"""ContextDepth: generated context and reference-guided candidate selection."""

__version__ = "0.1.0"


def __getattr__(name):
    if name in {"ContextDepthPipeline", "ContextDepthResult"}:
        from . import pipeline
        return getattr(pipeline, name)
    if name in {"Outpainting", "OutpaintingBaseline"}:
        from . import generation
        return getattr(generation, name)
    raise AttributeError(name)


__all__ = ["ContextDepthPipeline", "ContextDepthResult", "Outpainting", "OutpaintingBaseline"]
