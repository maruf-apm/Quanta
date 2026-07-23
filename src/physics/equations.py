"""
Physics module defining first-order ODEs and their exact solutions.
All equations from Audu et al. (2026) are implemented here.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict, Type


class ODEEquation(ABC):
    """Abstract base class for first-order ODE initial value problems."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the equation."""
        pass

    @abstractmethod
    def residual(
        self, n: torch.Tensor, m: torch.Tensor, dm_dn: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the ODE residual: R = dm/dn - f(n, m).
        For a well-satisfied equation, R should be zero.
        """
        pass

    @abstractmethod
    def exact(
        self, n: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Analytical exact solution m(n)."""
        pass

    @abstractmethod
    def initial_condition(self) -> Tuple[float, float]:
        """Return (n0, m0) initial condition tuple."""
        pass

    @property
    @abstractmethod
    def domain(self) -> Tuple[float, float]:
        """Return (n_min, n_max) computational domain."""
        pass

    def _torch_or_np(self, n, torch_fn, np_fn):
        """Helper to dispatch to torch or numpy depending on input type."""
        if isinstance(n, torch.Tensor):
            return torch_fn(n)
        return np_fn(n)


# ---------------------------------------------------------------------------
# Test 1: dm/dn = n^2 - 4m,  m(0) = 1
# Real: m(n) = (31/32)*exp(-4n) + (1/4)*n^2 - (1/8)*n + 1/32
# ---------------------------------------------------------------------------
class Test1Equation(ODEEquation):
    name = "Test1Equation"

    def residual(self, n, m, dm_dn):
        return dm_dn - (n**2 - 4.0 * m)

    def exact(self, n):
        def _fn(x):
            return (
                (31.0 / 32.0) * np.exp(-4.0 * x) + 0.25 * x**2 - 0.125 * x + 1.0 / 32.0
            )

        def _torch_fn(x):
            return (
                (31.0 / 32.0) * torch.exp(-4.0 * x)
                + 0.25 * x**2
                - 0.125 * x
                + 1.0 / 32.0
            )

        return self._torch_or_np(n, _torch_fn, _fn)

    def initial_condition(self):
        return (0.0, 1.0)

    @property
    def domain(self):
        return (0.0, 1.0)


# ---------------------------------------------------------------------------
# Test 2: dm/dn = exp(-2n) - 5m,  m(0) = 1
# Real: m(n) = (1/3)*exp(-2n) + (2/3)*exp(-5n)
# ---------------------------------------------------------------------------
class Test2Equation(ODEEquation):
    name = "Test2Equation"

    def residual(self, n, m, dm_dn):
        return dm_dn - (torch.exp(-2.0 * n) - 5.0 * m)

    def exact(self, n):
        def _fn(x):
            return (1.0 / 3.0) * np.exp(-2.0 * x) + (2.0 / 3.0) * np.exp(-5.0 * x)

        def _torch_fn(x):
            return (1.0 / 3.0) * torch.exp(-2.0 * x) + (2.0 / 3.0) * torch.exp(-5.0 * x)

        return self._torch_or_np(n, _torch_fn, _fn)

    def initial_condition(self):
        return (0.0, 1.0)

    @property
    def domain(self):
        return (0.0, 1.0)


# ---------------------------------------------------------------------------
# Test 3: dm/dn = m*n^3 - 1.5m,  m(0) = 1
# Real: m(n) = exp(1.5*n - n^4/4)
# ---------------------------------------------------------------------------
class Test3Equation(ODEEquation):
    name = "Test3Equation"

    def residual(self, n, m, dm_dn):
        return dm_dn - (m * n**3 - 1.5 * m)

    def exact(self, n):
        def _fn(x):
            return np.exp(-1.5 * x + x**4 / 4.0)

        def _torch_fn(x):
            return torch.exp(-1.5 * x + x**4 / 4.0)

        return self._torch_or_np(n, _torch_fn, _fn)

    def initial_condition(self):
        return (0.0, 1.0)

    @property
    def domain(self):
        return (0.0, 1.0)


# ---------------------------------------------------------------------------
# Test 4: dm/dn = -2*n*m / (1 + n^2),  m(0) = 1
# Real: m(n) = 1 / (1 + n^2)
# ---------------------------------------------------------------------------
class Test4Equation(ODEEquation):
    name = "Test4Equation"

    def residual(self, n, m, dm_dn):
        return dm_dn - (-2.0 * n * m / (1.0 + n**2))

    def exact(self, n):
        def _fn(x):
            return 1.0 / (1.0 + x**2)

        def _torch_fn(x):
            return 1.0 / (1.0 + x**2)

        return self._torch_or_np(n, _torch_fn, _fn)

    def initial_condition(self):
        return (0.0, 1.0)

    @property
    def domain(self):
        return (0.0, 1.0)


# ---------------------------------------------------------------------------
# Test 5: dm/dn = -0.5*(m - 3),  m(0) = 4
# Real: m(n) = 3 + exp(-n/2)
# ---------------------------------------------------------------------------
class Test5Equation(ODEEquation):
    name = "Test5Equation"

    def residual(self, n, m, dm_dn):
        return dm_dn - (-0.5 * (m - 3.0))

    def exact(self, n):
        def _fn(x):
            return 3.0 + np.exp(-0.5 * x)

        def _torch_fn(x):
            return 3.0 + torch.exp(-0.5 * x)

        return self._torch_or_np(n, _torch_fn, _fn)

    def initial_condition(self):
        return (0.0, 4.0)

    @property
    def domain(self):
        return (0.0, 1.0)


# ---------------------------------------------------------------------------
# Test 6: dm/dn = -2.2067e-12 * (m^4 - 81e8),  m(0) = 1000,  n in [0,500]
# No closed-form; exact obtained via high-fidelity RK45 integration.
# ---------------------------------------------------------------------------
class Test6Equation(ODEEquation):
    name = "Test6Equation"

    def __init__(self):
        super().__init__()
        from scipy.integrate import solve_ivp
        from scipy.interpolate import interp1d

        self._n_fine = np.linspace(0.0, 500.0, 20000)

        def _ode(t, y):
            return -2.2067e-12 * (y[0] ** 4 - 81.0e8)

        sol = solve_ivp(
            _ode,
            [0.0, 500.0],
            [1000.0],
            t_eval=self._n_fine,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        self._m_fine = sol.y[0]
        self._interp = interp1d(
            self._n_fine, self._m_fine, kind="cubic", fill_value="extrapolate"
        )

    def residual(self, n, m, dm_dn):
        return dm_dn - (-2.2067e-12 * (m**4 - 81.0e8))

    def exact(self, n):
        if isinstance(n, torch.Tensor):
            n_np = n.detach().cpu().numpy()
            val = self._interp(n_np)
            return torch.tensor(val, dtype=n.dtype, device=n.device)
        return self._interp(n)

    def initial_condition(self):
        return (0.0, 1000.0)

    @property
    def domain(self):
        return (0.0, 500.0)


# Registry for factory instantiation
EQUATION_REGISTRY: Dict[str, Type[ODEEquation]] = {
    "Test1Equation": Test1Equation,
    "Test2Equation": Test2Equation,
    "Test3Equation": Test3Equation,
    "Test4Equation": Test4Equation,
    "Test5Equation": Test5Equation,
    "Test6Equation": Test6Equation,
}
