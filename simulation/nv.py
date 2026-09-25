"""Physical NV construction and rotating-frame tripod simulation functions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from quaccatoo import NV, PulsedSim, QSys
from qutip import Qobj, basis, expect, ket2dm, tensor

from .envelope import EnvelopeData, LoopOrder


@dataclass(frozen=True)
class NVStateData:
    """Bare NV eigenstates in the fixed ``(-, 0, +)`` nuclear ordering."""

    ground_states: tuple[Qobj, Qobj, Qobj]
    bare_excited_states: tuple[Qobj, Qobj, Qobj]
    plus_states: tuple[Qobj, Qobj, Qobj]
    ground_energies_mhz: np.ndarray
    bare_excited_energies_mhz: np.ndarray
    plus_energies_mhz: np.ndarray
    ground_expected_energies_mhz: np.ndarray
    bare_excited_expected_energies_mhz: np.ndarray
    plus_expected_energies_mhz: np.ndarray
    reference_states: tuple[Qobj, ...]

    @property
    def all_states(self) -> tuple[Qobj, ...]:
        """Return states in ``g-,g0,g+,x-,x0,x+,s-,s0,s+`` order."""
        return self.ground_states + self.bare_excited_states + self.plus_states


@dataclass(frozen=True)
class TransitionData:
    """RF and MW transition frequencies and projected drive matrices."""

    rf_carrier_freq_mhz: np.ndarray
    ground_transition_freq_mhz: np.ndarray
    omega_mw_mhz: np.ndarray
    plus_mw_freq_mhz: np.ndarray
    rf_matrix: np.ndarray
    mw_transition_matrix: np.ndarray


@dataclass(frozen=True)
class RFDressingData:
    """Three-level RF RWA Hamiltonian and its dressed excited states."""

    rf_projected_matrix: np.ndarray
    g_minus_zero: complex
    g_zero_plus: complex
    h_dressed_mhz: np.ndarray
    dressed_quasienergies_mhz: np.ndarray
    dressed_coefficient_vectors: np.ndarray
    dressed_excited_states: tuple[Qobj, Qobj, Qobj]
    ancilla_state: Qobj
    ancilla_quasienergy_mhz: float
    ancilla_bare_coefficients: np.ndarray


@dataclass(frozen=True)
class ProjectionData:
    """Named state projectors and observables used by every simulation."""

    named: dict[str, Qobj]
    bare_projection_operators: tuple[Qobj, ...]
    loop_population_observables: tuple[Qobj, ...]
    loop_population_labels: tuple[str, ...]
    state_labels: tuple[str, ...]


@dataclass(frozen=True)
class RotatingFrameModel:
    """Static Hamiltonian and calibrated rotating-frame control operators."""

    h_loop_static_mhz: Qobj
    h_rf_detuning_mhz: Qobj
    h_rf_coupling_mhz: Qobj
    mw_drive_amplitudes_mhz: np.ndarray
    mw_bare_matrix_elements: np.ndarray
    mw_phase_calibration: np.ndarray
    mw_forward_operators: tuple[Qobj, Qobj, Qobj]
    initial_state: Qobj


def create_nv_system(
    SIMULATION_PARAMETERS: Mapping[str, Any],
    N: int | None = None,
    verbosity: bool = False,
) -> NV:
    """Create and normalize the full nine-state QuaCCAToo NV system.

    Parameters
    ----------
    SIMULATION_PARAMETERS
        Shared simulation dictionary containing at least ``B0_mT`` and
        ``N_ISOTOPE``.
    N
        Optional isotope override. When omitted, ``N_ISOTOPE`` is used.
    verbosity
        Print Hamiltonian dimensions and QuaCCAToo reference frequencies.

    Returns
    -------
    NV
        The physical QuaCCAToo NV object before any truncation.
    """
    isotope = int(SIMULATION_PARAMETERS["N_ISOTOPE"] if N is None else N)
    nv = NV(
        B0=float(SIMULATION_PARAMETERS["B0_mT"]),
        units_B0="mT",
        N=isotope,
    )
    nv.rho0 = nv.rho0 / nv.rho0.tr()

    if nv.H0.shape != (9, 9):
        raise ValueError(f"Expected a 9x9 NV Hamiltonian, received {nv.H0.shape}")
    if not np.isclose(nv.rho0.tr(), 1.0):
        raise ValueError("The normalized NV initial state must have unit trace")

    if verbosity:
        print("Created physical NV system.")
        print("  Full Hamiltonian shape:", nv.H0.shape)
        print("  Initial-state trace:", nv.rho0.tr())
        print("  MW reference frequencies (MHz):", nv.MW_freqs)
        print("  RF reference frequencies (MHz):", nv.RF_freqs)
    return nv


def derive_nv_states(
    nv: NV,
    SIMULATION_PARAMETERS: Mapping[str, Any],
    verbosity: bool = False,
) -> NVStateData:
    """Match physical eigenstates to the nine reference spin states.

    Returns
    -------
    NVStateData
        Ground, ``m_s=-1`` (bare excited), and ``m_s=+1`` states. Every
        three-state tuple uses the required ``(-, 0, +)`` nuclear ordering.
    """
    ground_references = tuple(
        tensor(basis(3, 1), basis(3, index)) for index in (2, 1, 0)
    )
    excited_references = tuple(
        tensor(basis(3, 2), basis(3, index)) for index in (2, 1, 0)
    )
    plus_references = tuple(
        tensor(basis(3, 0), basis(3, index)) for index in (2, 1, 0)
    )
    reference_states = ground_references + excited_references + plus_references
    reference_names = (
        "g_minus",
        "g_zero",
        "g_plus",
        "e_minus",
        "e_zero",
        "e_plus",
        "s_minus",
        "s_zero",
        "s_plus",
    )

    threshold = float(SIMULATION_PARAMETERS["STATE_OVERLAP_THRESHOLD"])
    eigenenergies_mhz, eigenstates = nv.H0.eigenstates()
    matched: dict[str, tuple[Qobj, float]] = {}
    for eigenstate, energy in zip(eigenstates, eigenenergies_mhz):
        overlaps = np.array(
            [complex(eigenstate.dag() * state) for state in reference_states]
        )
        matches = np.flatnonzero(np.abs(overlaps) > threshold)
        if len(matches) != 1:
            raise ValueError(
                "Each physical eigenstate must overlap exactly one reference "
                f"state above {threshold:.3f}; found {len(matches)}"
            )
        name = reference_names[int(matches[0])]
        if name in matched:
            raise KeyError(f"Reference state {name} was matched more than once")
        matched[name] = (eigenstate, float(energy))

    def ordered(prefix: str) -> tuple[tuple[Qobj, Qobj, Qobj], np.ndarray]:
        names = (f"{prefix}_minus", f"{prefix}_zero", f"{prefix}_plus")
        states = tuple(matched[name][0] for name in names)
        energies = np.array([matched[name][1] for name in names], dtype=float)
        return states, energies  # type: ignore[return-value]

    ground_states, ground_energies = ordered("g")
    excited_states, excited_energies = ordered("e")
    plus_states, plus_energies = ordered("s")

    ground_expected = np.array([expect(nv.H0, state) for state in ground_states])
    excited_expected = np.array([expect(nv.H0, state) for state in excited_states])
    plus_expected = np.array([expect(nv.H0, state) for state in plus_states])

    state_data = NVStateData(
        ground_states,
        excited_states,
        plus_states,
        ground_energies,
        excited_energies,
        plus_energies,
        ground_expected.astype(float),
        excited_expected.astype(float),
        plus_expected.astype(float),
        reference_states,
    )
    if verbosity:
        print("Matched physical NV eigenstates in (-, 0, +) order.")
        print("  Ground energies (MHz):", state_data.ground_energies_mhz)
        print("  Bare excited energies (MHz):", state_data.bare_excited_energies_mhz)
        print("  Plus-manifold energies (MHz):", state_data.plus_energies_mhz)
        print(
            "  Maximum eigenvalue/expectation mismatch (MHz):",
            max(
                np.max(np.abs(ground_energies - ground_expected)),
                np.max(np.abs(excited_energies - excited_expected)),
                np.max(np.abs(plus_energies - plus_expected)),
            ),
        )
    return state_data


def derive_transition_data(
    nv: NV,
    states: NVStateData,
    verbosity: bool = False,
) -> TransitionData:
    """Calculate allowed RF/MW frequencies and drive matrix elements."""
    rf_carriers = np.abs(np.diff(states.bare_excited_expected_energies_mhz))
    ground_carriers = np.abs(np.diff(states.ground_expected_energies_mhz))
    omega_mw = np.abs(
        states.bare_excited_expected_energies_mhz
        - states.ground_expected_energies_mhz
    )
    plus_mw = np.abs(
        states.plus_expected_energies_mhz - states.ground_expected_energies_mhz
    )
    rf_matrix = np.array(
        [
            [complex(final.dag() * nv.RF_h1 * initial) for initial in states.bare_excited_states]
            for final in states.bare_excited_states
        ]
    )
    mw_matrix = np.array(
        [
            [complex(excited.dag() * nv.MW_h1 * ground) for ground in states.ground_states]
            for excited in states.bare_excited_states
        ]
    )
    transitions = TransitionData(
        rf_carriers,
        ground_carriers,
        omega_mw,
        plus_mw,
        rf_matrix,
        mw_matrix,
    )
    if verbosity:
        print("Derived physical transition data.")
        print("  RF e-<->e0 and e0<->e+ carriers (MHz):", rf_carriers)
        print("  MW g<->e carriers (MHz):", omega_mw)
        print("  RF matrix in (e-, e0, e+) basis:\n", np.real_if_close(rf_matrix))
        print("  MW g-to-e matrix:\n", np.real_if_close(mw_matrix))
    return transitions


def construct_rf_dressing(
    states: NVStateData,
    transitions: TransitionData,
    SIMULATION_PARAMETERS: Mapping[str, Any],
    verbosity: bool = False,
) -> RFDressingData:
    """Construct and diagonalize the three-state RF RWA Hamiltonian."""
    rf_projected = transitions.rf_matrix.copy()
    d_minus_zero = rf_projected[1, 0]
    d_zero_plus = rf_projected[1, 2]
    if not np.isclose(d_minus_zero, rf_projected[0, 1].conjugate()):
        raise ValueError("The e-<->e0 RF projection is not Hermitian")
    if not np.isclose(d_zero_plus, rf_projected[2, 1].conjugate()):
        raise ValueError("The e0<->e+ RF projection is not Hermitian")

    omega_rf = float(SIMULATION_PARAMETERS["OMEGA_RF_MHZ"])
    rf_phases = np.asarray(SIMULATION_PARAMETERS["RF_PHASES"], dtype=float)
    g_minus_zero = 0.5 * omega_rf * d_minus_zero * np.exp(-1j * rf_phases[0])
    g_zero_plus = 0.5 * omega_rf * d_zero_plus * np.exp(-1j * rf_phases[1])

    h_dressed = np.diag(
        np.asarray(SIMULATION_PARAMETERS["DETUNINGS_MHZ"], dtype=float)
    ).astype(complex)
    h_dressed[1, 0] = g_minus_zero
    h_dressed[0, 1] = np.conj(g_minus_zero)
    h_dressed[1, 2] = g_zero_plus
    h_dressed[2, 1] = np.conj(g_zero_plus)
    quasienergies, coefficient_vectors = np.linalg.eigh(h_dressed)

    dressed_states: list[Qobj] = []
    for coefficients in coefficient_vectors.T:
        dressed_state = 0.0 * states.bare_excited_states[0]
        for coefficient, bare_state in zip(coefficients, states.bare_excited_states):
            dressed_state += coefficient * bare_state
        dressed_states.append(dressed_state.unit())

    ancilla = dressed_states[0]
    ancilla_coefficients = np.array(
        [complex(state.dag() * ancilla) for state in states.bare_excited_states]
    )
    largest_index = int(np.argmax(np.abs(ancilla_coefficients)))
    global_phase = np.exp(-1j * np.angle(ancilla_coefficients[largest_index]))
    ancilla = global_phase * ancilla
    dressed_states[0] = ancilla
    ancilla_coefficients = np.array(
        [complex(state.dag() * ancilla) for state in states.bare_excited_states]
    )

    bare_projector_sum = sum(ket2dm(state) for state in states.bare_excited_states)
    dressed_projector_sum = sum(ket2dm(state) for state in dressed_states)
    if (dressed_projector_sum - bare_projector_sum).norm() >= 1e-12:
        raise ValueError("Bare and dressed excited bases do not span the same subspace")

    dressing = RFDressingData(
        rf_projected,
        complex(g_minus_zero),
        complex(g_zero_plus),
        h_dressed,
        quasienergies,
        coefficient_vectors,
        tuple(dressed_states),  # type: ignore[arg-type]
        ancilla,
        float(quasienergies[0]),
        ancilla_coefficients,
    )
    if verbosity:
        print("Constructed RF-dressed excited manifold.")
        print("  RF RWA Hamiltonian (MHz):\n", h_dressed)
        print("  Dressed quasienergies (MHz):", quasienergies)
        print("  Ancilla quasienergy (MHz):", dressing.ancilla_quasienergy_mhz)
        print("  Ancilla bare coefficients:", ancilla_coefficients)
    return dressing


def construct_projection_operators(
    states: NVStateData,
    dressing: RFDressingData,
    verbosity: bool = False,
) -> ProjectionData:
    """Create state projectors in the exact observable order used downstream."""
    named = {
        "P_g_minus": ket2dm(states.ground_states[0]),
        "P_g_zero": ket2dm(states.ground_states[1]),
        "P_g_plus": ket2dm(states.ground_states[2]),
        "P_ancilla": ket2dm(dressing.ancilla_state),
        "P_e_zero": ket2dm(dressing.dressed_excited_states[1]),
        "P_e_plus": ket2dm(dressing.dressed_excited_states[2]),
        "P_x_minus": ket2dm(states.bare_excited_states[0]),
        "P_x_zero": ket2dm(states.bare_excited_states[1]),
        "P_x_plus": ket2dm(states.bare_excited_states[2]),
        "P_s_minus": ket2dm(states.plus_states[0]),
        "P_s_zero": ket2dm(states.plus_states[1]),
        "P_s_plus": ket2dm(states.plus_states[2]),
    }
    named["P_tripod_states"] = (
        named["P_g_minus"]
        + named["P_g_zero"]
        + named["P_g_plus"]
        + named["P_ancilla"]
    )
    named["P_all_ground_and_excited_states"] = (
        named["P_tripod_states"] + named["P_e_zero"] + named["P_e_plus"]
    )
    named["P_all_states"] = (
        named["P_all_ground_and_excited_states"]
        + named["P_s_minus"]
        + named["P_s_zero"]
        + named["P_s_plus"]
    )

    bare_projectors = tuple(
        named[key]
        for key in (
            "P_g_minus",
            "P_g_zero",
            "P_g_plus",
            "P_x_minus",
            "P_x_zero",
            "P_x_plus",
            "P_s_minus",
            "P_s_zero",
            "P_s_plus",
        )
    )
    observables = tuple(
        named[key]
        for key in (
            "P_g_minus",
            "P_g_zero",
            "P_g_plus",
            "P_ancilla",
            "P_tripod_states",
            "P_all_ground_and_excited_states",
            "P_all_states",
        )
    )
    labels = (
        r"$P_{g_-}$",
        r"$P_{g_0}$",
        r"$P_{g_+}$",
        r"$P_{\mathrm{ancilla}}$",
        r"$P_{\mathrm{tripod}}$",
        r"$P_{G+E}$",
        r"$P_{\rm all}$",
    )
    state_labels = (
        r"$|g_- \rangle$",
        r"$|g_0 \rangle$",
        r"$|g_+ \rangle$",
        r"$|x_- \rangle$",
        r"$|x_0 \rangle$",
        r"$|x_+ \rangle$",
        r"$|s_- \rangle$",
        r"$|s_0 \rangle$",
        r"$|s_+ \rangle$",
    )
    projections = ProjectionData(named, bare_projectors, observables, labels, state_labels)
    if verbosity:
        print("Constructed 9 bare projectors and 7 loop observables.")
        print("  Observable order:", labels)
    return projections


def construct_initial_state(
    states: NVStateData,
    transitions: TransitionData,
    SIMULATION_PARAMETERS: Mapping[str, Any],
    verbosity: bool = False,
) -> Qobj:
    """Construct the same normalized initial dark-state superposition."""
    phases = np.asarray(SIMULATION_PARAMETERS["MW_PHASES"], dtype=float)
    omega_mw = transitions.omega_mw_mhz
    term_1 = omega_mw[0] * np.exp(-1j * phases[0]) * states.ground_states[0]
    term_2 = omega_mw[1] * np.exp(-1j * phases[1]) * states.ground_states[1]
    denominator = np.sqrt(omega_mw[0] ** 2 + omega_mw[1] ** 2)
    initial_state = (term_1 - term_2) / denominator
    if not np.isclose(initial_state.norm(), 1.0):
        raise ValueError("The constructed initial state is not normalized")
    if verbosity:
        print("Constructed initial state.")
        print("  <psi|psi> =", initial_state.dag() * initial_state)
    return initial_state


def build_rotating_frame_model(
    nv: NV,
    states: NVStateData,
    dressing: RFDressingData,
    initial_state: Qobj,
    SIMULATION_PARAMETERS: Mapping[str, Any],
    verbosity: bool = False,
) -> RotatingFrameModel:
    """Build the static Hamiltonian and calibrated MW leg operators.

    Returns
    -------
    RotatingFrameModel
        All Hamiltonian components needed by ``PulsedSim``. MW operators retain
        the ``(-, 0, +)`` order used by the envelope functions.
    """
    active_excited = sum(ket2dm(state) for state in states.bare_excited_states)
    h_rf_detuning = sum(
        detuning * ket2dm(state)
        for detuning, state in zip(
            np.asarray(SIMULATION_PARAMETERS["DETUNINGS_MHZ"]),
            states.bare_excited_states,
        )
    )
    e_minus, e_zero, e_plus = states.bare_excited_states
    h_rf_coupling = (
        dressing.g_minus_zero * e_zero * e_minus.dag()
        + np.conj(dressing.g_minus_zero) * e_minus * e_zero.dag()
        + dressing.g_zero_plus * e_zero * e_plus.dag()
        + np.conj(dressing.g_zero_plus) * e_plus * e_zero.dag()
    )
    h_loop_static = (
        h_rf_detuning - dressing.ancilla_quasienergy_mhz * active_excited
    )

    eta = np.array(
        [
            complex(dressing.ancilla_state.dag() * nv.MW_h1 * ground)
            for ground in states.ground_states
        ]
    )
    if np.any(np.abs(eta) < 1e-12):
        raise ValueError("A selected MW leg has a vanishing ancilla matrix element")

    target_legs = np.asarray(SIMULATION_PARAMETERS["TARGET_MW_LEG_MHZ"])
    drive_amplitudes = 2.0 * target_legs / np.abs(eta)
    bare_matrix_elements = np.array(
        [
            complex(excited.dag() * nv.MW_h1 * ground)
            for excited, ground in zip(
                states.bare_excited_states, states.ground_states
            )
        ]
    )
    phase_calibration = np.exp(-1j * np.angle(eta))
    forward_operators = tuple(
        0.5
        * amplitude
        * phase_correction
        * matrix_element
        * excited
        * ground.dag()
        for amplitude, phase_correction, matrix_element, excited, ground in zip(
            drive_amplitudes,
            phase_calibration,
            bare_matrix_elements,
            states.bare_excited_states,
            states.ground_states,
        )
    )
    model = RotatingFrameModel(
        h_loop_static,
        h_rf_detuning,
        h_rf_coupling,
        drive_amplitudes,
        bare_matrix_elements,
        phase_calibration,
        forward_operators,  # type: ignore[arg-type]
        initial_state,
    )
    if verbosity:
        print("Built rotating-frame loop model.")
        print("  Static Hamiltonian shape:", h_loop_static.shape)
        print("  Target projected MW legs (MHz):", target_legs)
        print("  Physical MW drive amplitudes (MHz):", drive_amplitudes)
    return model


def conjugate_coefficient(coefficient: Callable[..., Any]) -> Callable[..., Any]:
    """Return a coefficient function that is the complex conjugate of its input."""

    def conjugated(time: float | np.ndarray, **kwargs: Any) -> Any:
        return np.conj(coefficient(time, **kwargs))

    return conjugated


def build_pulsedsim_controls(
    model: RotatingFrameModel,
    envelopes: EnvelopeData,
    loop_order: LoopOrder = "C1C2",
) -> tuple[list[Qobj], list[Callable[..., Any]]]:
    """Assemble control Hamiltonians and coefficients in matching order."""
    if loop_order not in envelopes.mw_rotating_pulses:
        raise ValueError("loop_order must be 'C1C2', 'C2C1', or 'no_loops'")

    control_hamiltonians = [model.h_rf_coupling_mhz]
    pulse_shapes: list[Callable[..., Any]] = [envelopes.rf_rotating_envelope]
    for forward_operator, pulse in zip(
        model.mw_forward_operators,
        envelopes.mw_rotating_pulses[loop_order],
    ):
        control_hamiltonians.extend([forward_operator, forward_operator.dag()])
        pulse_shapes.extend([pulse, conjugate_coefficient(pulse)])
    return control_hamiltonians, pulse_shapes


def run_rotating_experiment_pulsedsim(
    model: RotatingFrameModel,
    projections: ProjectionData,
    envelopes: EnvelopeData,
    SIMULATION_PARAMETERS: Mapping[str, Any],
    loop_order: LoopOrder = "C1C2",
    collapse_operators: Sequence[Qobj] | None = None,
    verbosity: bool = False,
) -> PulsedSim:
    """Run one complete loop order and retain a density matrix at every time.

    Parameters
    ----------
    model, projections, envelopes
        Prepared simulation components returned by the setup functions.
    SIMULATION_PARAMETERS
        Shared parameters, including the reporting grid and solver options.
    loop_order
        ``C1C2``, ``C2C1``, or the constant-phase ``no_loops`` control.
    collapse_operators
        Optional Lindblad collapse operators. Density matrices are used even
        when this is ``None`` so open and closed runs have identical datatypes.
    verbosity
        Print start, progress, and normalization diagnostics.

    Returns
    -------
    PulsedSim
        QuaCCAToo experiment with ``rho``, ``results``, ``final_state``, and
        ``variable`` populated over the complete reporting grid.
    """
    initial_density = ket2dm(model.initial_state)
    system = QSys(
        H0=model.h_loop_static_mhz,
        rho0=initial_density,
        c_ops=None if collapse_operators is None else list(collapse_operators),
        observable=list(projections.loop_population_observables),
        units_H0="MHz",
    )
    experiment = PulsedSim(system=system)
    h1, pulse_shapes = build_pulsedsim_controls(model, envelopes, loop_order)
    simulation_time = np.asarray(SIMULATION_PARAMETERS["SIMULATION_TIME"])
    solver_options = dict(SIMULATION_PARAMETERS["SOLVER_OPTIONS"])
    trajectory = [initial_density.copy()]

    if verbosity:
        collapse_count = 0 if collapse_operators is None else len(collapse_operators)
        print(
            f"Running {loop_order}: {len(simulation_time) - 1} intervals, "
            f"{collapse_count} collapse operators."
        )
    report_every = max(1, (len(simulation_time) - 1) // 4)
    for interval, (start_time, end_time) in enumerate(
        zip(simulation_time[:-1], simulation_time[1:]), start=1
    ):
        experiment.add_pulse(
            duration=float(end_time - start_time),
            h1=h1,
            pulse_shape=pulse_shapes,
            time_steps=2,
            options=solver_options,
        )
        trajectory.append(experiment.rho.copy())
        if verbosity and (
            interval % report_every == 0 or interval == len(simulation_time) - 1
        ):
            print(f"  {loop_order}: completed {interval}/{len(simulation_time) - 1}")

    experiment.final_state = trajectory[-1]
    experiment.rho = trajectory
    experiment.variable = simulation_time.copy()
    experiment._get_results()
    if verbosity:
        normalization_error = float(
            np.max(np.abs(np.asarray(experiment.results[6]) - 1.0))
        )
        print(f"  {loop_order}: maximum trace error = {normalization_error:.3e}")
    return experiment


def run_loop_experiments(
    model: RotatingFrameModel,
    projections: ProjectionData,
    envelopes: EnvelopeData,
    SIMULATION_PARAMETERS: Mapping[str, Any],
    collapse_operators: Sequence[Qobj] | None = None,
    verbosity: bool = False,
) -> dict[LoopOrder, PulsedSim]:
    """Run ``C1C2``, ``C2C1``, and the no-loop reference in fixed order."""
    experiments: dict[LoopOrder, PulsedSim] = {}
    for loop_order in ("C1C2", "C2C1", "no_loops"):
        experiments[loop_order] = run_rotating_experiment_pulsedsim(
            model,
            projections,
            envelopes,
            SIMULATION_PARAMETERS,
            loop_order=loop_order,
            collapse_operators=collapse_operators,
            verbosity=verbosity,
        )
    if verbosity:
        print("Completed all three loop-order experiments.\n")
    return experiments


def construct_collapse_operators(
    initial_state: Qobj,
    final_state: Qobj,
    gamma_per_us: float,
) -> tuple[Qobj, Qobj]:
    """Return equal-rate forward and reverse Lindblad jump operators."""
    amplitude = np.sqrt(gamma_per_us / (2.0 * np.pi))
    forward = amplitude * final_state * initial_state.dag()
    return forward, forward.dag()


def make_nv14_collapse_operators(
    states: NVStateData,
    SIMULATION_PARAMETERS: Mapping[str, Any],
    electron_t1_us: float | None | Literal["from_parameters"] = "from_parameters",
    nuclear_t1_us: float | None | Literal["from_parameters"] = "from_parameters",
    include_plus_manifold: bool = True,
    verbosity: bool = False,
) -> list[Qobj]:
    """Build the same electron and nuclear T1 operator network as the notebook."""
    electron_t1 = (
        SIMULATION_PARAMETERS["ELECTRON_T1_US"]
        if electron_t1_us == "from_parameters"
        else electron_t1_us
    )
    nuclear_t1 = (
        SIMULATION_PARAMETERS["NUCLEAR_T1_US"]
        if nuclear_t1_us == "from_parameters"
        else nuclear_t1_us
    )
    collapse_operators: list[Qobj] = []

    if electron_t1 is not None:
        branch_rate = 1.0 / ((3.0 if include_plus_manifold else 2.0) * electron_t1)
        for ground, excited in zip(states.ground_states, states.bare_excited_states):
            collapse_operators.extend(
                construct_collapse_operators(ground, excited, branch_rate)
            )
        if include_plus_manifold:
            for ground, plus_state in zip(states.ground_states, states.plus_states):
                collapse_operators.extend(
                    construct_collapse_operators(ground, plus_state, branch_rate)
                )

    if nuclear_t1 is not None:
        nuclear_rate = 1.0 / nuclear_t1
        manifolds = [states.ground_states, states.bare_excited_states]
        if include_plus_manifold:
            manifolds.append(states.plus_states)
        for manifold in manifolds:
            for left_state, right_state in zip(manifold[:-1], manifold[1:]):
                collapse_operators.extend(
                    construct_collapse_operators(left_state, right_state, nuclear_rate)
                )

    if verbosity:
        print(f"Constructed {len(collapse_operators)} NV-14 collapse operators.")
        print(f"  Electron T1: {electron_t1} us")
        print(f"  Nuclear T1: {nuclear_t1} us")
    return collapse_operators


def get_final_populations(
    experiment_or_state: PulsedSim | Qobj,
    states: NVStateData,
) -> np.ndarray:
    """Return final populations in the nine-state bare ordering."""
    state = (
        experiment_or_state.final_state
        if isinstance(experiment_or_state, PulsedSim)
        else experiment_or_state
    )
    populations = expect([ket2dm(ket) for ket in states.all_states], state)
    return np.abs(np.asarray(np.real_if_close(populations), dtype=float))
