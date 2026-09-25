"""High-level setup, execution, and plotting workflow for the simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from quaccatoo import NV, PulsedSim
from qutip import Qobj, ket2dm

from utils.plotting_functions import (
    plot_coherence_phase_grid,
    plot_envelopes,
    plot_final_population_comparison,
    plot_loop_population_comparison,
    plot_mw_pulses,
    plot_rf_lab_tones,
    plot_rf_dressing_comparison,
)

from .envelope import (
    EnvelopeData,
    build_envelope_data,
    rf_lab_coefficient_tone_1,
    rf_lab_coefficient_tone_2,
)
from .nv import (
    NVStateData,
    ProjectionData,
    RFDressingData,
    RotatingFrameModel,
    TransitionData,
    build_rotating_frame_model,
    construct_initial_state,
    construct_projection_operators,
    construct_rf_dressing,
    create_nv_system,
    derive_nv_states,
    derive_transition_data,
    get_final_populations,
    run_loop_experiments,
)


@dataclass(frozen=True)
class PreparedSimulation:
    """Named outputs of the setup stages, retained for analysis and plotting."""

    parameters: Mapping[str, Any]
    nv: NV
    states: NVStateData
    transitions: TransitionData
    dressing: RFDressingData
    projections: ProjectionData
    envelopes: EnvelopeData
    model: RotatingFrameModel


def prepare_nv_simulation(
    SIMULATION_PARAMETERS: Mapping[str, Any],
    verbosity: bool = False,
) -> PreparedSimulation:
    """Run every deterministic setup stage before time evolution.

    Parameters
    ----------
    SIMULATION_PARAMETERS
        Complete parameter dictionary created in ``master.ipynb``.
    verbosity
        Print a stage-by-stage description and key numerical diagnostics.

    Returns
    -------
    PreparedSimulation
        NV object, ordered states, transition data, RF dressing, projectors,
        cached envelopes, and the final rotating-frame Hamiltonian model.
    """
    if verbosity:
        print("\n[1/7] Creating the physical NV system")
    nv = create_nv_system(SIMULATION_PARAMETERS, verbosity=verbosity)

    if verbosity:
        print("\n[2/7] Matching the physical NV eigenstates")
    states = derive_nv_states(nv, SIMULATION_PARAMETERS, verbosity=verbosity)

    if verbosity:
        print("\n[3/7] Deriving RF and MW transition data")
    transitions = derive_transition_data(nv, states, verbosity=verbosity)

    # Keep notebook-compatible derived frequencies in the shared dictionary.
    # SIMULATION_PARAMETERS is deliberately created as a mutable dict in master.
    if isinstance(SIMULATION_PARAMETERS, dict):
        SIMULATION_PARAMETERS.update(
            {
                "RF_CARRIER_FREQ_MHZ": transitions.rf_carrier_freq_mhz,
                "ground_transition_frequencies": transitions.ground_transition_freq_mhz,
                "OMEGA_MW_MHZ": transitions.omega_mw_mhz,
                "plus_mw_frequencies_mhz": transitions.plus_mw_freq_mhz,
            }
        )

    if verbosity:
        print("\n[4/7] Constructing the RF-dressed manifold")
    dressing = construct_rf_dressing(
        states, transitions, SIMULATION_PARAMETERS, verbosity=verbosity
    )
    if isinstance(SIMULATION_PARAMETERS, dict):
        SIMULATION_PARAMETERS["MW_DRESSED_CARRIER_FREQ_MHZ"] = (
            transitions.omega_mw_mhz + dressing.ancilla_quasienergy_mhz
        )

    if verbosity:
        print("\n[5/7] Constructing observables and initial state")
    projections = construct_projection_operators(states, dressing, verbosity=verbosity)
    initial_state = construct_initial_state(
        states, transitions, SIMULATION_PARAMETERS, verbosity=verbosity
    )

    if verbosity:
        print("\n[6/7] Caching all control envelopes")
    envelopes = build_envelope_data(SIMULATION_PARAMETERS, verbosity=verbosity)

    if verbosity:
        print("\n[7/7] Building the rotating-frame control model")
    model = build_rotating_frame_model(
        nv,
        states,
        dressing,
        initial_state,
        SIMULATION_PARAMETERS,
        verbosity=verbosity,
    )
    if verbosity:
        print("\nSimulation preparation complete.\n")
    return PreparedSimulation(
        SIMULATION_PARAMETERS,
        nv,
        states,
        transitions,
        dressing,
        projections,
        envelopes,
        model,
    )


def run_prepared_simulation(
    prepared: PreparedSimulation,
    collapse_operators: Sequence[Qobj] | None = None,
    verbosity: bool = False,
) -> dict[str, PulsedSim]:
    """Run all loop orders using an already prepared physical model."""
    return run_loop_experiments(
        prepared.model,
        prepared.projections,
        prepared.envelopes,
        prepared.parameters,
        collapse_operators=collapse_operators,
        verbosity=verbosity,
    )


def plot_prepared_controls(
    prepared: PreparedSimulation,
) -> dict[str, tuple[Any, Any]]:
    """Plot control envelopes, C1C2 MW quadratures, and RF dressed energies."""
    envelope_figure = plot_envelopes(prepared.parameters)
    time_us = np.asarray(prepared.parameters["SIMULATION_TIME"])
    solver_time = 2.0 * np.pi * time_us
    mw_figures = {}
    for loop_order in ("C1C2", "C2C1"):
        pulses = prepared.envelopes.mw_rotating_pulses[loop_order]
        pulse_values = tuple(np.asarray(pulse(solver_time)) for pulse in pulses)
        mw_figures[loop_order] = plot_mw_pulses(
            time_us, *pulse_values, loop_label=loop_order
        )

    rf_tone_figure = plot_rf_lab_tones(
        time_us,
        np.asarray(rf_lab_coefficient_tone_1(solver_time, prepared.parameters)),
        np.asarray(rf_lab_coefficient_tone_2(solver_time, prepared.parameters)),
    )

    energy_fig, energy_axes = plt.subplots(1, 2, figsize=(9, 5), constrained_layout=True)
    bare_relative = (
        prepared.states.bare_excited_expected_energies_mhz
        - np.mean(prepared.states.bare_excited_expected_energies_mhz)
    )
    plot_rf_dressing_comparison(
        energy_axes,
        bare_relative,
        prepared.dressing.dressed_quasienergies_mhz,
        bare_labels=(r"$|x_-\rangle$", r"$|x_0\rangle$", r"$|x_+\rangle$"),
        dressed_labels=(r"$|e\rangle$", r"$|e_0\rangle$", r"$|e_+\rangle$"),
    )
    return {
        "envelopes": envelope_figure,
        "mw_quadratures_C1C2": mw_figures["C1C2"],
        "mw_quadratures_C2C1": mw_figures["C2C1"],
        "rf_lab_tones": rf_tone_figure,
        "rf_dressing": (energy_fig, energy_axes),
    }


def plot_simulation_results(
    prepared: PreparedSimulation,
    experiments: Mapping[str, PulsedSim],
    verbosity: bool = False,
) -> dict[str, tuple[Any, Any]]:
    """Recreate the population, coherence-phase, and final-state figures."""
    population_figure = plot_loop_population_comparison(
        experiments,
        prepared.parameters,
        prepared.projections.loop_population_labels,
    )
    phase_figure = plot_coherence_phase_grid(
        experiments["C1C2"],
        prepared.states.all_states,
        prepared.projections.state_labels,
        prepared.states.ground_states[0],
        prepared.parameters,
    )
    initial_populations = get_final_populations(
        ket2dm(prepared.model.initial_state), prepared.states
    )
    final_populations = {
        key: get_final_populations(experiment, prepared.states)
        for key, experiment in experiments.items()
    }
    final_figure = plot_final_population_comparison(
        prepared.projections.state_labels,
        initial_populations,
        final_populations,
        verbosity=verbosity,
    )
    return {
        "population_dynamics": population_figure,
        "coherence_phases": phase_figure,
        "final_populations": final_figure,
    }
