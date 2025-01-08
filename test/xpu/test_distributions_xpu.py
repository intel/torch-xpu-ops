# Owner(s): ["module: intel"]


from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport
with XPUPatchForImport(False):
    from test_distributions import (
        _get_examples,
        pairwise,
        TestAgainstScipy,
        TestConstraints,
        TestDistributions,
        TestDistributionShapes,
        TestFunctors,
        TestJit,
        TestKL,
        TestLazyLogitsInitialization,
        TestNumericalStability,
        TestRsample,
        TestValidation,
    )

from itertools import product

import numpy as np
import scipy
import torch
from packaging import version
from torch.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Cauchy,
    Chi2,
    ContinuousBernoulli,
    Dirichlet,
    Exponential,
    FisherSnedecor,
    Gamma,
    Geometric,
    Gumbel,
    HalfCauchy,
    HalfNormal,
    Independent,
    InverseGamma,
    Laplace,
    LogNormal,
    LowRankMultivariateNormal,
    Multinomial,
    MultivariateNormal,
    Normal,
    OneHotCategorical,
    Pareto,
    Poisson,
    StudentT,
    Uniform,
    VonMises,
    Weibull,
    Wishart,
)
from torch.nn.functional import softmax
from torch.testing._internal.common_utils import set_rng_seed


def _test_beta_underflow_gpu(self):
    set_rng_seed(1)
    num_samples = 50000
    conc = torch.tensor(1e-2, dtype=torch.float64).xpu()
    beta_samples = Beta(conc, conc).sample([num_samples])
    self.assertEqual((beta_samples == 0).sum(), 0)
    self.assertEqual((beta_samples == 1).sum(), 0)
    # assert support is concentrated around 0 and 1
    frac_zeros = float((beta_samples < 0.1).sum()) / num_samples
    frac_ones = float((beta_samples > 0.9).sum()) / num_samples
    # TODO: increase precision once imbalance on GPU is fixed.
    self.assertEqual(frac_zeros, 0.5, atol=0.12, rtol=0)
    self.assertEqual(frac_ones, 0.5, atol=0.12, rtol=0)


def _test_zero_excluded_binomial(self):
    vals = Binomial(
        total_count=torch.tensor(1.0).xpu(), probs=torch.tensor(0.9).xpu()
    ).sample(torch.Size((100000000,)))
    self.assertTrue((vals >= 0).all())
    vals = Binomial(
        total_count=torch.tensor(1.0).xpu(), probs=torch.tensor(0.1).xpu()
    ).sample(torch.Size((100000000,)))
    self.assertTrue((vals < 2).all())
    vals = Binomial(
        total_count=torch.tensor(1.0).xpu(), probs=torch.tensor(0.5).xpu()
    ).sample(torch.Size((10000,)))
    # vals should be roughly half zeroes, half ones
    assert (vals == 0.0).sum() > 4000
    assert (vals == 1.0).sum() > 4000


def _test_gamma_gpu_sample(self):
    set_rng_seed(0)
    for alpha, beta in product([0.1, 1.0, 5.0], [0.1, 1.0, 10.0]):
        a, b = torch.tensor([alpha]).xpu(), torch.tensor([beta]).xpu()
        self._check_sampler_sampler(
            Gamma(a, b),
            scipy.stats.gamma(alpha, scale=1.0 / beta),
            f"Gamma(alpha={alpha}, beta={beta})",
            failure_rate=1e-4,
        )


def _test_gamma_gpu_shape(self):
    alpha = torch.randn(2, 3).xpu().exp().requires_grad_()
    beta = torch.randn(2, 3).xpu().exp().requires_grad_()
    alpha_1d = torch.randn(1).xpu().exp().requires_grad_()
    beta_1d = torch.randn(1).xpu().exp().requires_grad_()
    self.assertEqual(Gamma(alpha, beta).sample().size(), (2, 3))
    self.assertEqual(Gamma(alpha, beta).sample((5,)).size(), (5, 2, 3))
    self.assertEqual(Gamma(alpha_1d, beta_1d).sample((1,)).size(), (1, 1))
    self.assertEqual(Gamma(alpha_1d, beta_1d).sample().size(), (1,))
    self.assertEqual(Gamma(0.5, 0.5).sample().size(), ())
    self.assertEqual(Gamma(0.5, 0.5).sample((1,)).size(), (1,))

    def ref_log_prob(idx, x, log_prob):
        a = alpha.view(-1)[idx].detach().cpu()
        b = beta.view(-1)[idx].detach().cpu()
        expected = scipy.stats.gamma.logpdf(x.cpu(), a, scale=1 / b)
        self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

    self._check_log_prob(Gamma(alpha, beta), ref_log_prob)


def _test_poisson_gpu_sample(self):
    set_rng_seed(1)
    for rate in [0.12, 0.9, 4.0]:
        self._check_sampler_discrete(
            Poisson(torch.tensor([rate]).xpu()),
            scipy.stats.poisson(rate),
            f"Poisson(lambda={rate}, xpu)",
            failure_rate=1e-3,
        )


TestDistributions.test_beta_underflow_gpu = _test_beta_underflow_gpu
TestDistributions.test_zero_excluded_binomial = _test_zero_excluded_binomial
TestDistributions.test_gamma_gpu_sample = _test_gamma_gpu_sample
TestDistributions.test_gamma_gpu_shape = _test_gamma_gpu_shape
TestDistributions.test_poisson_gpu_sample = _test_poisson_gpu_sample
instantiate_device_type_tests(
    TestDistributions, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(TestRsample, globals(), only_for="xpu", allow_xpu=True)


def setup_TestDistributionShapes(self):
    self.scalar_sample = 1
    self.tensor_sample_1 = torch.ones(3, 2)
    self.tensor_sample_2 = torch.ones(3, 2, 3)


TestDistributionShapes.setUp = setup_TestDistributionShapes
instantiate_device_type_tests(
    TestDistributionShapes, globals(), only_for="xpu", allow_xpu=True
)


def setup_TestKL(self):
    class Binomial30(Binomial):
        def __init__(self, probs):
            super().__init__(30, probs)

    # These are pairs of distributions with 4 x 4 parameters as specified.
    # The first of the pair e.g. bernoulli[0] varies column-wise and the second
    # e.g. bernoulli[1] varies row-wise; that way we test all param pairs.
    bernoulli = pairwise(Bernoulli, [0.1, 0.2, 0.6, 0.9])
    binomial30 = pairwise(Binomial30, [0.1, 0.2, 0.6, 0.9])
    binomial_vectorized_count = (
        Binomial(torch.tensor([3, 4]), torch.tensor([0.4, 0.6])),
        Binomial(torch.tensor([3, 4]), torch.tensor([0.5, 0.8])),
    )
    beta = pairwise(Beta, [1.0, 2.5, 1.0, 2.5], [1.5, 1.5, 3.5, 3.5])
    categorical = pairwise(
        Categorical,
        [[0.4, 0.3, 0.3], [0.2, 0.7, 0.1], [0.33, 0.33, 0.34], [0.2, 0.2, 0.6]],
    )
    cauchy = pairwise(Cauchy, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
    chi2 = pairwise(Chi2, [1.0, 2.0, 2.5, 5.0])
    dirichlet = pairwise(
        Dirichlet,
        [[0.1, 0.2, 0.7], [0.5, 0.4, 0.1], [0.33, 0.33, 0.34], [0.2, 0.2, 0.4]],
    )
    exponential = pairwise(Exponential, [1.0, 2.5, 5.0, 10.0])
    gamma = pairwise(Gamma, [1.0, 2.5, 1.0, 2.5], [1.5, 1.5, 3.5, 3.5])
    gumbel = pairwise(Gumbel, [-2.0, 4.0, -3.0, 6.0], [1.0, 2.5, 1.0, 2.5])
    halfnormal = pairwise(HalfNormal, [1.0, 2.0, 1.0, 2.0])
    inversegamma = pairwise(InverseGamma, [1.0, 2.5, 1.0, 2.5], [1.5, 1.5, 3.5, 3.5])
    laplace = pairwise(Laplace, [-2.0, 4.0, -3.0, 6.0], [1.0, 2.5, 1.0, 2.5])
    lognormal = pairwise(LogNormal, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
    normal = pairwise(Normal, [-2.0, 2.0, -3.0, 3.0], [1.0, 2.0, 1.0, 2.0])
    independent = (Independent(normal[0], 1), Independent(normal[1], 1))
    onehotcategorical = pairwise(
        OneHotCategorical,
        [[0.4, 0.3, 0.3], [0.2, 0.7, 0.1], [0.33, 0.33, 0.34], [0.2, 0.2, 0.6]],
    )
    pareto = (
        Pareto(
            torch.tensor([2.5, 4.0, 2.5, 4.0]).expand(4, 4),
            torch.tensor([2.25, 3.75, 2.25, 3.75]).expand(4, 4),
        ),
        Pareto(
            torch.tensor([2.25, 3.75, 2.25, 3.8]).expand(4, 4),
            torch.tensor([2.25, 3.75, 2.25, 3.75]).expand(4, 4),
        ),
    )
    poisson = pairwise(Poisson, [0.3, 1.0, 5.0, 10.0])
    uniform_within_unit = pairwise(
        Uniform, [0.1, 0.9, 0.2, 0.75], [0.15, 0.95, 0.25, 0.8]
    )
    uniform_positive = pairwise(Uniform, [1, 1.5, 2, 4], [1.2, 2.0, 3, 7])
    uniform_real = pairwise(Uniform, [-2.0, -1, 0, 2], [-1.0, 1, 1, 4])
    uniform_pareto = pairwise(Uniform, [6.5, 7.5, 6.5, 8.5], [7.5, 8.5, 9.5, 9.5])
    continuous_bernoulli = pairwise(ContinuousBernoulli, [0.1, 0.2, 0.5, 0.9])

    # These tests should pass with precision = 0.01, but that makes tests very expensive.
    # Instead, we test with precision = 0.1 and only test with higher precision locally
    # when adding a new KL implementation.
    # The following pairs are not tested due to very high variance of the monte carlo
    # estimator; their implementations have been reviewed with extra care:
    # - (pareto, normal)
    self.precision = 0.1  # Set this to 0.01 when testing a new KL implementation.
    self.max_samples = int(1e07)  # Increase this when testing at smaller precision.
    self.samples_per_batch = int(1e04)
    self.finite_examples = [
        (bernoulli, bernoulli),
        (bernoulli, poisson),
        (beta, beta),
        (beta, chi2),
        (beta, exponential),
        (beta, gamma),
        (beta, normal),
        (binomial30, binomial30),
        (binomial_vectorized_count, binomial_vectorized_count),
        (categorical, categorical),
        (cauchy, cauchy),
        (chi2, chi2),
        (chi2, exponential),
        (chi2, gamma),
        (chi2, normal),
        (dirichlet, dirichlet),
        (exponential, chi2),
        (exponential, exponential),
        (exponential, gamma),
        (exponential, gumbel),
        (exponential, normal),
        (gamma, chi2),
        (gamma, exponential),
        (gamma, gamma),
        (gamma, gumbel),
        (gamma, normal),
        (gumbel, gumbel),
        (gumbel, normal),
        (halfnormal, halfnormal),
        (independent, independent),
        (inversegamma, inversegamma),
        (laplace, laplace),
        (lognormal, lognormal),
        (laplace, normal),
        (normal, gumbel),
        (normal, laplace),
        (normal, normal),
        (onehotcategorical, onehotcategorical),
        (pareto, chi2),
        (pareto, pareto),
        (pareto, exponential),
        (pareto, gamma),
        (poisson, poisson),
        (uniform_within_unit, beta),
        (uniform_positive, chi2),
        (uniform_positive, exponential),
        (uniform_positive, gamma),
        (uniform_real, gumbel),
        (uniform_real, normal),
        (uniform_pareto, pareto),
        (continuous_bernoulli, continuous_bernoulli),
        (continuous_bernoulli, exponential),
        (continuous_bernoulli, normal),
        (beta, continuous_bernoulli),
    ]

    self.infinite_examples = [
        (Bernoulli(0), Bernoulli(1)),
        (Bernoulli(1), Bernoulli(0)),
        (
            Categorical(torch.tensor([0.9, 0.1])),
            Categorical(torch.tensor([1.0, 0.0])),
        ),
        (
            Categorical(torch.tensor([[0.9, 0.1], [0.9, 0.1]])),
            Categorical(torch.tensor([1.0, 0.0])),
        ),
        (Beta(1, 2), Uniform(0.25, 1)),
        (Beta(1, 2), Uniform(0, 0.75)),
        (Beta(1, 2), Uniform(0.25, 0.75)),
        (Beta(1, 2), Pareto(1, 2)),
        (Binomial(31, 0.7), Binomial(30, 0.3)),
        (
            Binomial(torch.tensor([3, 4]), torch.tensor([0.4, 0.6])),
            Binomial(torch.tensor([2, 3]), torch.tensor([0.5, 0.8])),
        ),
        (Chi2(1), Beta(2, 3)),
        (Chi2(1), Pareto(2, 3)),
        (Chi2(1), Uniform(-2, 3)),
        (Exponential(1), Beta(2, 3)),
        (Exponential(1), Pareto(2, 3)),
        (Exponential(1), Uniform(-2, 3)),
        (Gamma(1, 2), Beta(3, 4)),
        (Gamma(1, 2), Pareto(3, 4)),
        (Gamma(1, 2), Uniform(-3, 4)),
        (Gumbel(-1, 2), Beta(3, 4)),
        (Gumbel(-1, 2), Chi2(3)),
        (Gumbel(-1, 2), Exponential(3)),
        (Gumbel(-1, 2), Gamma(3, 4)),
        (Gumbel(-1, 2), Pareto(3, 4)),
        (Gumbel(-1, 2), Uniform(-3, 4)),
        (Laplace(-1, 2), Beta(3, 4)),
        (Laplace(-1, 2), Chi2(3)),
        (Laplace(-1, 2), Exponential(3)),
        (Laplace(-1, 2), Gamma(3, 4)),
        (Laplace(-1, 2), Pareto(3, 4)),
        (Laplace(-1, 2), Uniform(-3, 4)),
        (Normal(-1, 2), Beta(3, 4)),
        (Normal(-1, 2), Chi2(3)),
        (Normal(-1, 2), Exponential(3)),
        (Normal(-1, 2), Gamma(3, 4)),
        (Normal(-1, 2), Pareto(3, 4)),
        (Normal(-1, 2), Uniform(-3, 4)),
        (Pareto(2, 1), Chi2(3)),
        (Pareto(2, 1), Exponential(3)),
        (Pareto(2, 1), Gamma(3, 4)),
        (Pareto(1, 2), Normal(-3, 4)),
        (Pareto(1, 2), Pareto(3, 4)),
        (Poisson(2), Bernoulli(0.5)),
        (Poisson(2.3), Binomial(10, 0.2)),
        (Uniform(-1, 1), Beta(2, 2)),
        (Uniform(0, 2), Beta(3, 4)),
        (Uniform(-1, 2), Beta(3, 4)),
        (Uniform(-1, 2), Chi2(3)),
        (Uniform(-1, 2), Exponential(3)),
        (Uniform(-1, 2), Gamma(3, 4)),
        (Uniform(-1, 2), Pareto(3, 4)),
        (ContinuousBernoulli(0.25), Uniform(0.25, 1)),
        (ContinuousBernoulli(0.25), Uniform(0, 0.75)),
        (ContinuousBernoulli(0.25), Uniform(0.25, 0.75)),
        (ContinuousBernoulli(0.25), Pareto(1, 2)),
        (Exponential(1), ContinuousBernoulli(0.75)),
        (Gamma(1, 2), ContinuousBernoulli(0.75)),
        (Gumbel(-1, 2), ContinuousBernoulli(0.75)),
        (Laplace(-1, 2), ContinuousBernoulli(0.75)),
        (Normal(-1, 2), ContinuousBernoulli(0.75)),
        (Uniform(-1, 1), ContinuousBernoulli(0.75)),
        (Uniform(0, 2), ContinuousBernoulli(0.75)),
        (Uniform(-1, 2), ContinuousBernoulli(0.75)),
    ]


TestKL.setUp = setup_TestKL
instantiate_device_type_tests(TestKL, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(
    TestConstraints, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestNumericalStability, globals(), only_for="xpu", allow_xpu=True
)


def setup_TestLazyLogitsInitialization(self):
    self.examples = [
        e
        for e in _get_examples()
        if e.Dist in (Categorical, OneHotCategorical, Bernoulli, Binomial, Multinomial)
    ]


TestLazyLogitsInitialization.setUp = setup_TestLazyLogitsInitialization
instantiate_device_type_tests(
    TestLazyLogitsInitialization, globals(), only_for="xpu", allow_xpu=True
)


def setup_TestAgainstScipy(self):
    positive_var = torch.randn(20, dtype=torch.double).exp()
    positive_var2 = torch.randn(20, dtype=torch.double).exp()
    random_var = torch.randn(20, dtype=torch.double)
    simplex_tensor = softmax(torch.randn(20, dtype=torch.double), dim=-1)
    cov_tensor = torch.randn(20, 20, dtype=torch.double)
    cov_tensor = cov_tensor @ cov_tensor.mT
    self.distribution_pairs = [
        (Bernoulli(simplex_tensor), scipy.stats.bernoulli(simplex_tensor)),
        (
            Beta(positive_var, positive_var2),
            scipy.stats.beta(positive_var, positive_var2),
        ),
        (
            Binomial(10, simplex_tensor),
            scipy.stats.binom(
                10 * np.ones(simplex_tensor.shape), simplex_tensor.numpy()
            ),
        ),
        (
            Cauchy(random_var, positive_var),
            scipy.stats.cauchy(loc=random_var, scale=positive_var),
        ),
        (Dirichlet(positive_var), scipy.stats.dirichlet(positive_var)),
        (
            Exponential(positive_var),
            scipy.stats.expon(scale=positive_var.reciprocal()),
        ),
        (
            FisherSnedecor(
                positive_var, 4 + positive_var2
            ),  # var for df2<=4 is undefined
            scipy.stats.f(positive_var, 4 + positive_var2),
        ),
        (
            Gamma(positive_var, positive_var2),
            scipy.stats.gamma(positive_var, scale=positive_var2.reciprocal()),
        ),
        (Geometric(simplex_tensor), scipy.stats.geom(simplex_tensor, loc=-1)),
        (
            Gumbel(random_var, positive_var2),
            scipy.stats.gumbel_r(random_var, positive_var2),
        ),
        (HalfCauchy(positive_var), scipy.stats.halfcauchy(scale=positive_var)),
        (HalfNormal(positive_var2), scipy.stats.halfnorm(scale=positive_var2)),
        (
            InverseGamma(positive_var, positive_var2),
            scipy.stats.invgamma(positive_var, scale=positive_var2),
        ),
        (
            Laplace(random_var, positive_var2),
            scipy.stats.laplace(random_var, positive_var2),
        ),
        (
            # Tests fail 1e-5 threshold if scale > 3
            LogNormal(random_var, positive_var.clamp(max=3)),
            scipy.stats.lognorm(s=positive_var.clamp(max=3), scale=random_var.exp()),
        ),
        (
            LowRankMultivariateNormal(
                random_var, torch.zeros(20, 1, dtype=torch.double), positive_var2
            ),
            scipy.stats.multivariate_normal(random_var, torch.diag(positive_var2)),
        ),
        (
            Multinomial(10, simplex_tensor),
            scipy.stats.multinomial(10, simplex_tensor),
        ),
        (
            MultivariateNormal(random_var, torch.diag(positive_var2)),
            scipy.stats.multivariate_normal(random_var, torch.diag(positive_var2)),
        ),
        (
            MultivariateNormal(random_var, cov_tensor),
            scipy.stats.multivariate_normal(random_var, cov_tensor),
        ),
        (
            Normal(random_var, positive_var2),
            scipy.stats.norm(random_var, positive_var2),
        ),
        (
            OneHotCategorical(simplex_tensor),
            scipy.stats.multinomial(1, simplex_tensor),
        ),
        (
            Pareto(positive_var, 2 + positive_var2),
            scipy.stats.pareto(2 + positive_var2, scale=positive_var),
        ),
        (Poisson(positive_var), scipy.stats.poisson(positive_var)),
        (
            StudentT(2 + positive_var, random_var, positive_var2),
            scipy.stats.t(2 + positive_var, random_var, positive_var2),
        ),
        (
            Uniform(random_var, random_var + positive_var),
            scipy.stats.uniform(random_var, positive_var),
        ),
        (
            VonMises(random_var, positive_var),
            scipy.stats.vonmises(positive_var, loc=random_var),
        ),
        (
            Weibull(
                positive_var[0], positive_var2[0]
            ),  # scipy var for Weibull only supports scalars
            scipy.stats.weibull_min(c=positive_var2[0], scale=positive_var[0]),
        ),
        (
            # scipy var for Wishart only supports scalars
            # SciPy allowed ndim -1 < df < ndim for Wishar distribution after version 1.7.0
            Wishart(
                (
                    20
                    if version.parse(scipy.__version__) < version.parse("1.7.0")
                    else 19
                )
                + positive_var[0],
                cov_tensor,
            ),
            scipy.stats.wishart(
                (
                    20
                    if version.parse(scipy.__version__) < version.parse("1.7.0")
                    else 19
                )
                + positive_var[0].item(),
                cov_tensor,
            ),
        ),
    ]


TestAgainstScipy.setUp = setup_TestAgainstScipy
instantiate_device_type_tests(
    TestAgainstScipy, globals(), only_for="xpu", allow_xpu=True
)

instantiate_device_type_tests(TestFunctors, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestValidation, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestJit, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
