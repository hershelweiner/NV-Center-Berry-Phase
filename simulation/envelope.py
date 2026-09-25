"""RF and MW envelopes for the rotating-frame NV simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np


LoopOrder: TypeAlias = Literal["C1C2", "C2C1", "no_loops"]
MWLoop: TypeAlias = Literal["1", "2", "const"]
MWOrder: TypeAlias = Literal["1", "2", "3"]
Coefficient: TypeAlias = Callable[..., complex | np.ndarray]


@dataclass(frozen=True)
class EnvelopeData:
    """Cached envelopes and solver-compatible coefficient functions."""

    mw_envelope_grids: dict[str, np.ndarray]
    mw_rotating_pulses: dict[LoopOrder, tuple[Coefficient, Coefficient, Coefficient]]
    rf_rotating_envelope: Coefficient


def is_iterable(value: object) -> bool:
    """Return whether ``value`` can be iterated over."""
    try:
        iter(value)  # type: ignore[arg-type]
        return True
    except TypeError:
        return False


def sin2_envelope(
    time_us: float | np.ndarray,
    ramp_time_us: float,
) -> float | np.ndarray:
    """Evaluate the original smooth :math:`\sin^2` ramp."""
    values = np.sin((np.pi * np.asarray(time_us)) / (2.0 * ramp_time_us)) ** 2
    return float(values) if values.ndim == 0 else values


def rf_envelope(
    time_us: float | np.ndarray,
    SIMULATION_PARAMETERS: Mapping[str, Any],
) -> float | np.ndarray:
    """Return the RF ramp-up, plateau, and ramp-down envelope.

    Parameters
    ----------
    time_us
        Scalar or array of physical times in microseconds.
    SIMULATION_PARAMETERS
        Shared parameter dictionary created in ``master.ipynb``.

    Returns
    -------
    float or numpy.ndarray
        Dimensionless RF envelope with the same shape as ``time_us``.
    """
    time = np.asarray(time_us)
    ramp_time = float(SIMULATION_PARAMETERS["rf_loop_ramp_us"])
    ramp_start = time - float(SIMULATION_PARAMETERS["t_rf_ramp_start_us"])
    ramp_end = time - float(SIMULATION_PARAMETERS["t_rf_ramp_down_start_us"])
    ramp_up = sin2_envelope(ramp_start, ramp_time)
    ramp_down = 1.0 - sin2_envelope(ramp_end, ramp_time)

    envelope = np.where(
        time < float(SIMULATION_PARAMETERS["t_rf_on_us"]),
        ramp_up,
        np.where(
            time < float(SIMULATION_PARAMETERS["t_rf_ramp_down_start_us"]),
            1.0,
            np.where(
                time < float(SIMULATION_PARAMETERS["TOTAL_TIME_US"]),
                ramp_down,
                0.0,
            ),
        ),
    )
    return float(envelope) if envelope.ndim == 0 else envelope


def mw_berry_loop_envelope(
    loop_time_us: float | np.ndarray,
    SIMULATION_PARAMETERS: Mapping[str, Any],
) -> complex | np.ndarray:
    """Return the complex unit-magnitude phase sweep for one Berry loop."""
    phase = 2.0 * np.pi * sin2_envelope(
        loop_time_us,
        float(SIMULATION_PARAMETERS["berry_loop_duration_us"]),
    )
    values = np.exp(-1j * phase)
    return complex(values) if np.ndim(values) == 0 else values


def mw_loop_sequence_envelope(
    time_us: np.ndarray,
    SIMULATION_PARAMETERS: Mapping[str, Any],
    loop: MWLoop = "1",
    order: MWOrder = "1",
) -> np.ndarray:
    """Build one complete complex MW envelope on the reporting grid.

    This retains the ramp and loop ordering from the original notebook. The
    three orders correspond to the ``-``, ``0``, and ``+`` tripod legs.
    """
    time = np.asarray(time_us, dtype=float)
    order_index = int(order) - 1
    if order_index not in (0, 1, 2):
        raise ValueError("order must be '1', '2', or '3'")
    if loop not in ("1", "2", "const"):
        raise ValueError("loop must be '1', '2', or 'const'")

    ramp_starts = np.array(
        [
            SIMULATION_PARAMETERS["t_mw1_ramp_start_us"],
            SIMULATION_PARAMETERS["t_mw2_ramp_start_us"],
            SIMULATION_PARAMETERS["t_mw3_ramp_start_us"],
        ],
        dtype=float,
    )
    ramp_times = np.asarray(SIMULATION_PARAMETERS["mw_loop_ramp_us"], dtype=float)
    ramp_start = float(ramp_starts[order_index])
    ramp_end = ramp_start + float(ramp_times[order_index])
    loop_1_start = float(SIMULATION_PARAMETERS["t_loop_1_start_us"])
    loop_2_start = float(SIMULATION_PARAMETERS["t_loop_2_start_us"])
    mw_ramp_down = float(SIMULATION_PARAMETERS["t_mw_ramp_down_start_us"])
    rf_ramp_down = float(SIMULATION_PARAMETERS["t_rf_ramp_down_start_us"])
    total_time = float(SIMULATION_PARAMETERS["TOTAL_TIME_US"])

    envelope = np.zeros(time.shape, dtype=complex)

    ramp_up_mask = (time >= ramp_start) & (time < ramp_end)
    if np.any(ramp_up_mask):
        local_time = time[ramp_up_mask] - ramp_start
        sampled_ramp_duration = float(np.max(local_time))
        envelope[ramp_up_mask] = sin2_envelope(
            local_time, sampled_ramp_duration
        )

    pre_loop_mask = (time >= ramp_end) & (time < loop_1_start)
    envelope[pre_loop_mask] = 1.0

    loop_1_mask = (time >= loop_1_start) & (time < loop_2_start)
    loop_2_mask = (time >= loop_2_start) & (time < mw_ramp_down)
    loop_1_values = mw_berry_loop_envelope(
        time[loop_1_mask] - loop_1_start, SIMULATION_PARAMETERS
    )
    loop_2_values = mw_berry_loop_envelope(
        time[loop_2_mask] - loop_2_start, SIMULATION_PARAMETERS
    )

    if loop == "1":
        envelope[loop_1_mask] = loop_1_values
        envelope[loop_2_mask] = 1.0
    elif loop == "2":
        envelope[loop_1_mask] = 1.0
        envelope[loop_2_mask] = loop_2_values
    else:
        envelope[loop_1_mask | loop_2_mask] = 1.0

    ramp_down_mask = (time >= mw_ramp_down) & (time < rf_ramp_down)
    if np.any(ramp_down_mask):
        local_time = time[ramp_down_mask] - mw_ramp_down
        sampled_ramp_duration = float(np.max(local_time))
        envelope[ramp_down_mask] = 1.0 - sin2_envelope(
            local_time,
            sampled_ramp_duration,
        )

    envelope[(time >= rf_ramp_down) & (time <= total_time)] = 0.0
    return envelope


def build_mw_envelope_grids(
    SIMULATION_PARAMETERS: Mapping[str, Any],
    verbosity: bool = False,
) -> dict[str, np.ndarray]:
    """Precompute all nine loop/order envelopes used by the solver."""
    simulation_time = np.asarray(SIMULATION_PARAMETERS["SIMULATION_TIME"])
    grids = {
        f"{loop}-{order}": mw_loop_sequence_envelope(
            simulation_time,
            SIMULATION_PARAMETERS,
            loop=loop,
            order=order,
        )
        for loop in ("1", "2", "const")
        for order in ("1", "2", "3")
    }
    if verbosity:
        print(f"Cached {len(grids)} MW envelope grids.")
    return grids


def rf_lab_coefficient_tone_1(
    solver_time: float | np.ndarray,
    SIMULATION_PARAMETERS: Mapping[str, Any],
    **_: Any,
) -> float | np.ndarray:
    """Return the laboratory RF tone resonant with ``x- <-> x0``."""
    carrier_frequencies = np.asarray(
        SIMULATION_PARAMETERS["RF_CARRIER_FREQ_MHZ"], dtype=float
    )
    phases = np.asarray(SIMULATION_PARAMETERS["RF_PHASES"], dtype=float)
    physical_time_us = np.asarray(solver_time) / (2.0 * np.pi)
    tone = float(SIMULATION_PARAMETERS["OMEGA_RF_MHZ"]) * np.cos(
        np.asarray(solver_time) * carrier_frequencies[0] + phases[0]
    )
    values = rf_envelope(physical_time_us, SIMULATION_PARAMETERS) * tone
    return float(values) if np.ndim(values) == 0 else values


def rf_lab_coefficient_tone_2(
    solver_time: float | np.ndarray,
    SIMULATION_PARAMETERS: Mapping[str, Any],
    **_: Any,
) -> float | np.ndarray:
    """Return the laboratory RF tone resonant with ``x0 <-> x+``."""
    carrier_frequencies = np.asarray(
        SIMULATION_PARAMETERS["RF_CARRIER_FREQ_MHZ"], dtype=float
    )
    phases = np.asarray(SIMULATION_PARAMETERS["RF_PHASES"], dtype=float)
    physical_time_us = np.asarray(solver_time) / (2.0 * np.pi)
    tone = float(SIMULATION_PARAMETERS["OMEGA_RF_MHZ"]) * np.cos(
        np.asarray(solver_time) * carrier_frequencies[1] + phases[1]
    )
    values = rf_envelope(physical_time_us, SIMULATION_PARAMETERS) * tone
    return float(values) if np.ndim(values) == 0 else values


def mw_loop_sequence_envelope_float_resolvable(
    time_us: float | np.ndarray,
    SIMULATION_PARAMETERS: Mapping[str, Any],
    mw_envelope_grids: Mapping[str, np.ndarray],
    loop: MWLoop,
    order: MWOrder,
    **_: Any,
) -> complex | np.ndarray:
    """Evaluate a cached complex MW envelope at scalar solver time."""
    if is_iterable(time_us):
        return mw_loop_sequence_envelope(
            np.asarray(time_us), SIMULATION_PARAMETERS, loop, order
        )

    grid = mw_envelope_grids[f"{loop}-{order}"]
    simulation_time = np.asarray(SIMULATION_PARAMETERS["SIMULATION_TIME"])
    return complex(np.interp(float(time_us), simulation_time, grid))


def _make_mw_coefficient(
    SIMULATION_PARAMETERS: Mapping[str, Any],
    mw_envelope_grids: Mapping[str, np.ndarray],
    loop: MWLoop,
    order: MWOrder,
    phase_rad: float,
    name: str,
) -> Coefficient:
    """Create one QuaCCAToo-compatible rotating-frame MW coefficient."""

    def coefficient(solver_time: float | np.ndarray, **_: Any) -> complex | np.ndarray:
        physical_time_us = np.asarray(solver_time) / (2.0 * np.pi)
        envelope = mw_loop_sequence_envelope_float_resolvable(
            physical_time_us,
            SIMULATION_PARAMETERS,
            mw_envelope_grids,
            loop,
            order,
        )
        return np.exp(-1j * phase_rad) * envelope

    coefficient.__name__ = name
    return coefficient


def build_envelope_data(
    SIMULATION_PARAMETERS: Mapping[str, Any],
    verbosity: bool = False,
) -> EnvelopeData:
    """Build cached envelopes and all solver coefficient functions.

    Returns
    -------
    EnvelopeData
        Contains the cached grids, loop-ordered MW functions, and RF function.
    """
    grids = build_mw_envelope_grids(SIMULATION_PARAMETERS, verbosity=verbosity)
    phases = np.asarray(SIMULATION_PARAMETERS["MW_PHASES"], dtype=float)
    loop_definitions: dict[LoopOrder, tuple[MWLoop, MWLoop, MWLoop]] = {
        "C1C2": ("1", "const", "2"),
        "C2C1": ("2", "const", "1"),
        "no_loops": ("const", "const", "const"),
    }
    pulse_map: dict[LoopOrder, tuple[Coefficient, Coefficient, Coefficient]] = {}
    for loop_order, leg_loops in loop_definitions.items():
        pulse_map[loop_order] = tuple(
            _make_mw_coefficient(
                SIMULATION_PARAMETERS,
                grids,
                loop=leg_loop,
                order=str(index + 1),  # type: ignore[arg-type]
                phase_rad=float(phases[index]),
                name=f"mw_{index}_{loop_order}",
            )
            for index, leg_loop in enumerate(leg_loops)
        )  # type: ignore[assignment]

    def rf_rotating_envelope(
        solver_time: float | np.ndarray, **_: Any
    ) -> float | np.ndarray:
        return rf_envelope(
            np.asarray(solver_time) / (2.0 * np.pi),
            SIMULATION_PARAMETERS,
        )

    if verbosity:
        print("Built RF coefficient and MW pulse sets for C1C2, C2C1, and no_loops.")
    return EnvelopeData(grids, pulse_map, rf_rotating_envelope)
