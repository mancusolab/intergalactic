# pattern: Functional Core

"""Matrix-free same-haplotype cis-regulatory interaction kernels."""

from importlib.metadata import PackageNotFoundError, version

from .likelihood import (
    covariance_operator as covariance_operator,
    gaussian_log_likelihood as gaussian_log_likelihood,
    GaussianLogLikelihood as GaussianLogLikelihood,
    LikelihoodMethod as LikelihoodMethod,
    optimize_variance_components as optimize_variance_components,
    VarianceComponentFit as VarianceComponentFit,
    VarianceComponentOperator as VarianceComponentOperator,
    VarianceComponents as VarianceComponents,
)
from .operators import (
    AdditiveHaplotypeKernel as AdditiveHaplotypeKernel,
    DiagonalNormalizer as DiagonalNormalizer,
    DiploidHaplotypeMap as DiploidHaplotypeMap,
    InteractionMode as InteractionMode,
    SameHaplotypeInteractionKernel as SameHaplotypeInteractionKernel,
    TraceNormalizer as TraceNormalizer,
    VarianceComponentKernel as VarianceComponentKernel,
)

try:
    __version__ = version("intergalactic")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del PackageNotFoundError, version
