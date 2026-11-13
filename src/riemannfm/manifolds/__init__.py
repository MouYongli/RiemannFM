"""Product manifold embedding spaces: Hyperbolic x Spherical x Euclidean."""

__all__ = [
    "EuclideanManifold",
    "LorentzManifold",
    "RiemannFMManifold",
    "RiemannFMProductManifold",
    "SphericalManifold",
]


def __getattr__(name: str):
    if name == "EuclideanManifold":
        from riemannfm.manifolds.euclidean import EuclideanManifold

        return EuclideanManifold
    if name == "LorentzManifold":
        from riemannfm.manifolds.lorentz import LorentzManifold

        return LorentzManifold
    if name == "RiemannFMManifold":
        from riemannfm.manifolds.base import RiemannFMManifold

        return RiemannFMManifold
    if name == "RiemannFMProductManifold":
        from riemannfm.manifolds.product import RiemannFMProductManifold

        return RiemannFMProductManifold
    if name == "SphericalManifold":
        from riemannfm.manifolds.spherical import SphericalManifold

        return SphericalManifold
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
