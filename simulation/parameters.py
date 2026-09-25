"""Parameter construction and validation for the NV Berry-phase simulation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np


SimulationParameters: TypeAlias = dict[str, Any]


def create_simulation_parameters(
    overrides: Mapping[str, Any] | None = None,
    verbosity: bool = False,
) -> SimulationParameters:
    """Create the shared parameter dictionary used by the complete workflow.

    Parameters
    ----------
    overrides
        Optional replacements for the independent default parameters. Derived
        amplitudes, event times, and the reporting grid are recalculated after
        the replacements are applied.
    verbosity
        Print the resulting physical and timing parameters when ``True``.

    Returns
    -------
    dict
        A dictionary containing every value that was a global variable in
        ``NV_Berry_Phase_sim.ipynb``. This dictionary is passed explicitly to
        all simulation and envelope functions.
    """
    SIMULATION_PARAMETERS: SimulationParameters = {
        # NV system
        "B0_mT": 20.0,
        "N_ISOTOPE": 14,
        # RF dressing
        "OMEGA_RF_MHZ": 1.5,
        "DETUNINGS_MHZ": np.zeros(3, dtype=float),
        "RF_PHASES": np.zeros(2, dtype=float),
        # MW tripod legs
        "MW_PHASES": np.zeros(3, dtype=float),
        "TARGET_MW_LEG_MHZ": np.full(3, 0.005, dtype=float),
        # Pulse timing
        "rf_loop_ramp_us": 4.0,
        "rf_settle_us": 2.0,
        "mw_loop_ramp_scale_us": 30.0,
        "mw_pulse_spacing_us": 2.0,
        "berry_loop_duration_us": 40.0,
        "loop_reporting_step_us": 0.5,
        # Relaxation parameters
        "ELECTRON_T1_US": 5_500.0,
        "NUCLEAR_T1_US": 43.0e6,
        # Solver settings retained from the original notebook
        "SOLVER_OPTIONS": {
            "method": "vern9",
            "nsteps": 20_000,
            "atol": 1e-9,
            "rtol": 1e-7,
        },
        "STATE_OVERLAP_THRESHOLD": 0.9,
    }

    if overrides is not None:
        SIMULATION_PARAMETERS.update(dict(overrides))

    target_legs = np.asarray(
        SIMULATION_PARAMETERS["TARGET_MW_LEG_MHZ"], dtype=float
    )
    if target_legs.shape != (3,) or np.any(target_legs <= 0.0):
        raise ValueError("TARGET_MW_LEG_MHZ must contain three positive values")

    mw_rabi_mhz = 2.0 * target_legs
    mw_rabi_total_mhz = float(np.sqrt(np.sum(mw_rabi_mhz**2)))
    scaled_mw_rabi = mw_rabi_mhz / mw_rabi_total_mhz
    mw_loop_amplitudes = scaled_mw_rabi / np.max(scaled_mw_rabi)
    mw_loop_ramp_us = (
        float(SIMULATION_PARAMETERS["mw_loop_ramp_scale_us"])
        * mw_loop_amplitudes
    )

    t_rf_ramp_start_us = 0.0
    t_rf_on_us = t_rf_ramp_start_us + float(
        SIMULATION_PARAMETERS["rf_loop_ramp_us"]
    )
    t_mw1_ramp_start_us = t_rf_on_us + float(
        SIMULATION_PARAMETERS["rf_settle_us"]
    )
    mw_spacing = float(SIMULATION_PARAMETERS["mw_pulse_spacing_us"])
    t_mw2_ramp_start_us = t_mw1_ramp_start_us + mw_spacing
    t_mw3_ramp_start_us = t_mw2_ramp_start_us + mw_spacing
    mw_ramp_starts = np.array(
        [t_mw1_ramp_start_us, t_mw2_ramp_start_us, t_mw3_ramp_start_us]
    )
    t_mw_ramp_end_us = mw_ramp_starts + mw_loop_ramp_us
    t_loop_1_start_us = float(np.max(t_mw_ramp_end_us))
    loop_duration = float(SIMULATION_PARAMETERS["berry_loop_duration_us"])
    t_loop_2_start_us = t_loop_1_start_us + loop_duration
    t_mw_ramp_down_start_us = t_loop_2_start_us + loop_duration
    t_mw_zero_us = t_mw_ramp_down_start_us + float(np.max(mw_loop_ramp_us))
    t_rf_ramp_down_start_us = t_mw_zero_us
    total_time_us = t_rf_ramp_down_start_us + float(
        SIMULATION_PARAMETERS["rf_loop_ramp_us"]
    )
    reporting_step = float(SIMULATION_PARAMETERS["loop_reporting_step_us"])
    simulation_time = np.arange(
        0.0, total_time_us + reporting_step, reporting_step
    )

    SIMULATION_PARAMETERS.update(
        {
            "TARGET_MW_LEG_MHZ": target_legs,
            "DETUNINGS_MHZ": np.asarray(
                SIMULATION_PARAMETERS["DETUNINGS_MHZ"], dtype=float
            ),
            "RF_PHASES": np.asarray(
                SIMULATION_PARAMETERS["RF_PHASES"], dtype=float
            ),
            "MW_PHASES": np.asarray(
                SIMULATION_PARAMETERS["MW_PHASES"], dtype=float
            ),
            "MW_RABI_MHZ": mw_rabi_mhz,
            "MW_RABI_TOTAL_MHZ": mw_rabi_total_mhz,
            "scaled_mw_rabi_total_mhz": scaled_mw_rabi,
            "mw_loop_amplitudes_mhz": mw_loop_amplitudes,
            "mw_loop_ramp_us": mw_loop_ramp_us,
            "t_rf_ramp_start_us": t_rf_ramp_start_us,
            "t_rf_on_us": t_rf_on_us,
            "t_mw1_ramp_start_us": t_mw1_ramp_start_us,
            "t_mw2_ramp_start_us": t_mw2_ramp_start_us,
            "t_mw3_ramp_start_us": t_mw3_ramp_start_us,
            "t_mw_ramp_end_us": t_mw_ramp_end_us,
            "t_loop_1_start_us": t_loop_1_start_us,
            "t_loop_2_start_us": t_loop_2_start_us,
            "t_mw_ramp_down_start_us": t_mw_ramp_down_start_us,
            "t_mw_zero_us": t_mw_zero_us,
            "t_rf_ramp_down_start_us": t_rf_ramp_down_start_us,
            "TOTAL_TIME_US": total_time_us,
            "SIMULATION_TIME": simulation_time,
            "solver_total_simulation_time": simulation_time * (2.0 * np.pi),
            "t_mw1_on_us": t_mw_ramp_down_start_us - t_mw1_ramp_start_us,
            "t_mw2_on_us": t_mw_ramp_down_start_us - t_mw2_ramp_start_us,
            "t_mw3_on_us": t_mw_ramp_down_start_us - t_mw3_ramp_start_us,
        }
    )

    validate_simulation_parameters(SIMULATION_PARAMETERS)
    if verbosity:
        print_simulation_parameters(SIMULATION_PARAMETERS)
    return SIMULATION_PARAMETERS


def validate_simulation_parameters(
    SIMULATION_PARAMETERS: Mapping[str, Any],
) -> None:
    """Validate array sizes and the ordering of pulse events."""
    expected_shapes = {
        "DETUNINGS_MHZ": (3,),
        "RF_PHASES": (2,),
        "MW_PHASES": (3,),
        "TARGET_MW_LEG_MHZ": (3,),
        "mw_loop_ramp_us": (3,),
    }
    for key, shape in expected_shapes.items():
        if np.asarray(SIMULATION_PARAMETERS[key]).shape != shape:
            raise ValueError(f"{key} must have shape {shape}")

    loop_start = float(SIMULATION_PARAMETERS["t_loop_1_start_us"])
    ramp_starts = np.array(
        [
            SIMULATION_PARAMETERS["t_mw1_ramp_start_us"],
            SIMULATION_PARAMETERS["t_mw2_ramp_start_us"],
            SIMULATION_PARAMETERS["t_mw3_ramp_start_us"],
        ],
        dtype=float,
    )
    ramp_ends = ramp_starts + np.asarray(
        SIMULATION_PARAMETERS["mw_loop_ramp_us"], dtype=float
    )
    if np.any(ramp_ends > loop_start + 1e-12):
        raise ValueError("Every MW ramp must finish before the first loop starts")


def print_simulation_parameters(
    SIMULATION_PARAMETERS: Mapping[str, Any],
) -> None:
    """Print a compact summary of the shared parameter dictionary."""
    print("=========== NV Parameters ===========")
    print(f"B0: {SIMULATION_PARAMETERS['B0_mT']} mT")
    print(f"Nitrogen isotope: {SIMULATION_PARAMETERS['N_ISOTOPE']}")
    print("\n========== RF/MW Parameters ==========")
    print(f"RF Rabi frequency: {SIMULATION_PARAMETERS['OMEGA_RF_MHZ']} MHz")
    print(f"RF phases: {SIMULATION_PARAMETERS['RF_PHASES']} rad")
    print(f"MW Rabi frequencies: {SIMULATION_PARAMETERS['MW_RABI_MHZ']} MHz")
    print(f"MW phases: {SIMULATION_PARAMETERS['MW_PHASES']} rad")
    print(f"Detunings: {SIMULATION_PARAMETERS['DETUNINGS_MHZ']} MHz")
    print("\n======== Simulation Parameters ========")
    print(f"RF ramp time: {SIMULATION_PARAMETERS['rf_loop_ramp_us']} us")
    print(f"MW ramp times: {SIMULATION_PARAMETERS['mw_loop_ramp_us']} us")
    print(f"Loop duration: {SIMULATION_PARAMETERS['berry_loop_duration_us']} us")
    print(f"Reporting step: {SIMULATION_PARAMETERS['loop_reporting_step_us']} us")
    print(f"Total time: {SIMULATION_PARAMETERS['TOTAL_TIME_US']} us\n")
