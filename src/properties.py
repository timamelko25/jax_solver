"""Physical properties for two-phase flow in porous media.

This module implements:
- Relative permeability models (Brooks-Corey, Corey)
- Fractional flow function
- Mobility and total mobility
- Capillary pressure
"""

import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass


@dataclass
class FlowParams:
    """Flow parameters for two-phase system.

    Attributes:
        k: Absolute permeability (m^2)
        phi: Porosity (fraction)
        mu_w: Water viscosity (Pa·s)
        mu_o: Oil viscosity (Pa·s)
        krw0: Endpoint water relative permeability
        kro0: Endpoint oil relative permeability
        n_w: Brooks-Corey exponent for water
        n_o: Brooks-Corey exponent for oil
        Swc: Connate water saturation
        Sor: Residual oil saturation
        N_sin_alpha: Gravity term N*sin(α) for gravitational flow
        D_leading: Retardation factor D_I→II for CO2 solubility
        D_trailing: Retardation factor D_II→J for CO2 solubility
        rho_o: Oil density (reference, default 1.0)
        rho_w: Water density (reference, default 1.0)
    """

    k: float = 1e-12
    phi: float = 0.2
    mu_w: float = 1e-3
    mu_o: float = 1e-2
    krw0: float = 0.3
    kro0: float = 0.8
    n_w: float = 2.0
    n_o: float = 2.0
    Swc: float = 0.2
    Sor: float = 0.2
    N_sin_alpha: float = 0.0
    D_leading: float = 0.0
    D_trailing: float = 0.0
    rho_o: float = 1.0
    rho_w: float = 1.0


@jit
def effective_saturation(Sw: jnp.ndarray, Swc: float, Sor: float) -> jnp.ndarray:
    """Compute effective water saturation.

    Args:
        Sw: Water saturation
        Swc: Connate water saturation
        Sor: Residual oil saturation

    Returns:
        Effective saturation in range [0, 1]
    """
    return jnp.clip((Sw - Swc) / (1 - Swc - Sor), 0.0, 1.0)


def relative_permeability_brooks_corey(
    Sw: jnp.ndarray, params: FlowParams
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Brooks-Corey relative permeability model.

    Args:
        Sw: Water saturation
        params: Flow parameters

    Returns:
        Tuple of (kr_w, kr_o) relative permeabilities
    """
    Se = effective_saturation(Sw, params.Swc, params.Sor)

    kr_w = params.krw0 * jnp.power(Se, params.n_w)
    kr_o = params.kro0 * jnp.power(1.0 - Se, params.n_o)

    kr_w = jnp.where(Sw < params.Swc, 0.0, kr_w)
    kr_o = jnp.where(Sw > 1.0 - params.Sor, 0.0, kr_o)

    return kr_w, kr_o


def relative_permeability_corey(
    Sw: jnp.ndarray, params: FlowParams
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Corey relative permeability model (simpler than Brooks-Corey).

    Args:
        Sw: Water saturation
        params: Flow parameters

    Returns:
        Tuple of (kr_w, kr_o) relative permeabilities
    """
    Sw_star = jnp.clip((Sw - params.Swc) / (1 - params.Swc - params.Sor), 0.0, 1.0)

    kr_w = params.krw0 * jnp.power(Sw_star, params.n_w)
    kr_o = params.kro0 * jnp.power(1.0 - Sw_star, params.n_o)

    return kr_w, kr_o


def fractional_flow(Sw: jnp.ndarray, params: FlowParams) -> jnp.ndarray:
    """Compute fractional flow of water phase.

    f_w = (kr_w / mu_w) / (kr_w / mu_w + kr_o / mu_o)

    Args:
        Sw: Water saturation
        params: Flow parameters

    Returns:
        Fractional flow of water (0 to 1)
    """
    kr_w, kr_o = relative_permeability_brooks_corey(Sw, params)

    lambda_w = kr_w / params.mu_w
    lambda_o = kr_o / params.mu_o

    total_mobility = lambda_w + lambda_o

    fw = jnp.where(total_mobility > 0, lambda_w / total_mobility, 0.0)

    return fw


def df_dsaturation(
    Sw: jnp.ndarray, params: FlowParams, eps: float = 1e-6
) -> jnp.ndarray:
    """Compute derivative of fractional flow w.r.t. saturation.

    Uses numerical differentiation with central differences.

    Args:
        Sw: Water saturation
        params: Flow parameters
        eps: Small perturbation for numerical differentiation

    Returns:
        df_w/dS_w
    """
    f_plus = fractional_flow(Sw + eps, params)
    f_minus = fractional_flow(Sw - eps, params)

    return (f_plus - f_minus) / (2 * eps)


def compute_mobility(
    Sw: jnp.ndarray, params: FlowParams
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute phase mobilities and total mobility.

    Args:
        Sw: Water saturation
        params: Flow parameters

    Returns:
        Tuple of (lambda_w, lambda_o, lambda_total)
    """
    kr_w, kr_o = relative_permeability_brooks_corey(Sw, params)

    lambda_w = kr_w / params.mu_w
    lambda_o = kr_o / params.mu_o
    lambda_total = lambda_w + lambda_o

    return lambda_w, lambda_o, lambda_total


def capillary_pressure_leverett(
    Sw: jnp.ndarray, params: FlowParams, gamma: float = 0.03
) -> jnp.ndarray:
    """Leverett J-function for capillary pressure.

    pc = gamma * sqrt(phi/k) * J(Se)

    Args:
        Sw: Water saturation
        params: Flow parameters
        gamma: Interfacial tension (N/m)

    Returns:
        Capillary pressure (Pa)
    """
    Se = effective_saturation(Sw, params.Swc, params.Sor)

    J = jnp.where(Se > 0, 0.5 * (1 - Se) / Se**0.5 + 0.5, jnp.inf)

    pc = gamma * jnp.sqrt(params.phi / params.k) * J

    return pc


def welge_construct(
    Sw_inj: float, Sw_init: float, params: FlowParams
) -> tuple[float, float]:
    """Welge construction for shock front position.

    Finds the shock saturation using graphical tangent method.

    Args:
        Sw_inj: Injected water saturation
        Sw_init: Initial oil saturation
        params: Flow parameters

    Returns:
        Tuple of (S_shock, shock_velocity)
    """
    n_points = 1000
    S_range = jnp.linspace(Sw_init + 0.001, Sw_inj - 0.001, n_points)

    df = df_dsaturation(S_range, params)

    shock_candidates = jnp.where(
        (df >= df_dsaturation(Sw_inj, params)) & (df <= df_dsaturation(Sw_init, params))
    )[0]

    if jnp.size(shock_candidates) == 0:
        return Sw_inj, df_dsaturation(Sw_inj, params)

    S_shock = S_range[shock_candidates[-1]]
    f_shock = fractional_flow(S_shock, params)
    f_init = fractional_flow(Sw_init, params)

    shock_velocity = (f_shock - f_init) / (S_shock - Sw_init)

    return float(S_shock), float(shock_velocity)


def fractional_flow_with_gravity(Sw: jnp.ndarray, params: FlowParams) -> jnp.ndarray:
    """Fractional flow with gravity term (Zhang et al., Eq. 9).

    f_w = [1 - (1-S)^n_o * N*sin(α)] / [1 + (1-S)^n_o / (M * S^n_w)]

    Args:
        Sw: Water saturation
        params: Flow parameters with N_sin_alpha

    Returns:
        Fractional flow with gravity
    """
    S = effective_saturation(Sw, params.Swc, params.Sor)

    M = params.kro0 / params.mu_w * params.mu_o / params.krw0

    numerator = 1 - (1 - S) ** params.n_o * params.N_sin_alpha
    denominator = 1 + (1 - S) ** params.n_o / (M * S**params.n_w)

    return jnp.where(denominator > 0, numerator / denominator, 0.0)


def fractional_flow_pure_gravity(Sw: jnp.ndarray, params: FlowParams) -> jnp.ndarray:
    """Purely gravitational fractional flow - opposite shocks (Zhang et al., Eq. 23).

    f = S² / [S² + μw/μo*(1-S)²] * [(1-S)² * μw/μo * (1-ρo/ρw)]

    Args:
        Sw: Water saturation
        params: Flow parameters with rho_o, rho_w

    Returns:
        Fractional flow for pure gravity (opposite direction shocks)
    """
    mu_ratio = params.mu_w / params.mu_o
    rho_ratio = params.rho_o / params.rho_w

    s_sq = Sw**2
    term1 = s_sq / (s_sq + mu_ratio * (1 - Sw) ** 2)
    term2 = (1 - Sw) ** 2 * mu_ratio * (1 - rho_ratio)

    return term1 * term2


def welge_construct_dual_shock(
    Sw_inj: float, Sw_init: float, params: FlowParams
) -> tuple[float, float, float, float]:
    """Dual-shock Welge construction for CO2 injection with solubility (Zhang et al., Eq. 19-20).

    Leading shock: v = (f(Sg1) - D_leading) / (Sg1 - D_leading)
    Trailing shock: v = (f(Sg2) - D_trailing) / (Sg2 - D_trailing)

    Args:
        Sw_inj: Injected saturation (1 - Sor)
        Sw_init: Initial saturation (Swc)
        params: Flow parameters with D_leading, D_trailing

    Returns:
        Tuple of (S_g1, v_leading, S_g2, v_trailing)
    """
    n_points = 500

    if params.D_leading > 0:
        S_range = jnp.linspace(
            Sw_init + 0.01, min(params.D_leading + 0.01, Sw_inj - 0.01), n_points
        )

        if jnp.size(S_range) > 0:
            f_vals = fractional_flow(S_range, params)
            slopes = (f_vals - params.D_leading) / (S_range - params.D_leading + 1e-10)
            valid = jnp.where(slopes > 0)[0]
            if jnp.size(valid) > 0:
                S_g1 = S_range[valid[0]]
                f_g1 = fractional_flow(S_g1, params)
                v_leading = (f_g1 - params.D_leading) / (
                    S_g1 - params.D_leading + 1e-10
                )
            else:
                S_g1, v_leading = Sw_inj, 0.0
        else:
            S_g1, v_leading = Sw_inj, 0.0
    else:
        S_g1, v_leading = 0.0, 0.0

    if params.D_trailing > 0:
        S_range = jnp.linspace(
            max(params.D_trailing - 0.01, Sw_init + 0.01), Sw_inj - 0.01, n_points
        )

        if jnp.size(S_range) > 0:
            f_vals = fractional_flow(S_range, params)
            slopes = (f_vals - params.D_trailing) / (
                S_range - params.D_trailing + 1e-10
            )
            valid = jnp.where(slopes > 0)[0]
            if jnp.size(valid) > 0:
                S_g2 = S_range[valid[-1]]
                f_g2 = fractional_flow(S_g2, params)
                v_trailing = (f_g2 - params.D_trailing) / (
                    S_g2 - params.D_trailing + 1e-10
                )
            else:
                S_g2, v_trailing = Sw_init, 0.0
        else:
            S_g2, v_trailing = Sw_init, 0.0
    else:
        S_g2, v_trailing = 0.0, 0.0

    return float(S_g1), float(v_leading), float(S_g2), float(v_trailing)


def create_mobility_ratio_case(M: float) -> FlowParams:
    """Create flow parameters for specified mobility ratio.

    Args:
        M: Desired mobility ratio (lambda_w / lambda_o)

    Returns:
        FlowParams with adjusted viscosities
    """
    base_params = FlowParams()
    base_params.mu_w = base_params.mu_o / M

    return base_params
