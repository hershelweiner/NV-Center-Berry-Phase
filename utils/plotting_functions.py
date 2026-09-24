import numpy as np
import matplotlib.pyplot as plt

def plot_individual_population_check(ax, var, P_g_minus_t, P_g_zero_t, P_g_plus_t, P_excited_t, ground_labels):
    """
    The first plot resolves the three tripod ground-state populations and the selected dressed-ancilla population. 
    Population not represented by these four curves can reside in the two excited superpositions orthogonal to $|e\rangle$.
    """

    total_pop = np.zeros_like(P_g_minus_t)
    for population, label in zip(
            (P_g_minus_t, P_g_zero_t, P_g_plus_t),
            ground_labels,
        ):
            total_pop += population
            ax.plot(var, population, lw=2, label=label)
            ax.set(
                xlabel=r"Pulse duration ($\mu$s)",
                ylabel="Population",
                title=fr"Resolved tripod-state populations",
                ylim=(-0.02, 1.02),
            )
            
    ax.plot(var, P_excited_t, lw=2.2, ls="--", label=r"$P_e$")
    total_pop += P_excited_t
    ax.plot(var, total_pop, lw=1.5, ls='--', label="total tripod population")
    ax.legend(ncol=2)

def plot_aggregate_population_checks(ax, var, P_ground_total_t, P_excited_t, P_all_states_t):
    """
    Compare

    $$P_G=P_{g_-}+P_{g_0}+P_{g_+},\qquad P_e=\langle e|\rho|e\rangle,$$

    and the trace check

    $$P_{\mathrm{all}}=\operatorname{Tr}\rho=1.$$

    $P_G+P_e$ need not equal one because the truncated NV space contains two additional excited combinations orthogonal to the chosen $|e\rangle$.
    """
    ax.plot(
        var,
        P_ground_total_t,
        lw=2.2,
        label=r"Total ground population $P_G$",
    )
    ax.plot(
        var,
        P_excited_t,
        lw=2.2,
        label=r"Selected excited population $P_e$",
    )
    ax.plot(
        var,
        P_all_states_t,
        color="black",
        lw=1.8,
        ls="--",
        label=r"Total tripod population $P_{\rm all}$",
    )
    ax.set(
        xlabel=r"Pulse duration ($\mu$s)",
        ylabel="Population",
        title="Aggregate population diagnostics",
        ylim=(-0.02, 1.05),
    )
    ax.legend()
    # plt.show()

    print(f"Maximum normalization error: {np.max(np.abs(P_all_states_t - 1.0)):.3e}")
    print(f"Maximum selected ancilla population: {np.max(P_excited_t):.6f}")
    print(f"Minimum total ground population: {np.min(P_ground_total_t):.6f}\n")

    assert np.allclose(P_all_states_t, 1.0, atol=1e-6)

    return ax

def plot_energy_check(ax, var, total_energy):
    ax.plot(var, total_energy, lw=2, label='Total Energy')
    ax.set(
        xlabel=r"Pulse duration ($\mu$s)",
        ylabel="Energy (MHz)",
        title=r"Resolved tripod-state energy",
        # ylim=(-0.02, 1.02),
    )
    ax.legend(ncol=2)
    # plt.show()

    return ax


def plot_energy_level_diagram(
        ax,
        energy_levels,
        labels=None,
        title="Energy levels",
        ylabel="Energy (MHz)",
        color="tab:blue",
        x_position=0.5,
        half_width=0.28,
        ):
    """Draw a compact, labeled energy-level diagram on an existing axis."""
    energy_levels = np.asarray(energy_levels, dtype=float)
    if labels is None:
        labels = [f"level {index}" for index in range(len(energy_levels))]
    if len(labels) != len(energy_levels):
        raise ValueError("labels and energy_levels must have the same length")

    # Separate text positions for levels that are close compared with the full
    # plotted span. The horizontal lines remain at the physical energies and
    # light connector lines point from each label back to its level.
    energy_span = max(float(np.ptp(energy_levels)), 1.0)
    minimum_text_spacing = 0.05 * energy_span
    order = np.argsort(energy_levels)
    text_positions = energy_levels.astype(float).copy()
    cluster = [int(order[0])] if len(order) else []
    clusters = []
    for previous, current in zip(order[:-1], order[1:]):
        if energy_levels[current] - energy_levels[previous] < minimum_text_spacing:
            cluster.append(int(current))
        else:
            clusters.append(cluster)
            cluster = [int(current)]
    if cluster:
        clusters.append(cluster)

    for members in clusters:
        if len(members) > 1:
            center = float(np.mean(energy_levels[members]))
            offsets = (np.arange(len(members)) - 0.5 * (len(members) - 1)) * minimum_text_spacing
            for member, offset in zip(members, offsets):
                text_positions[member] = center + offset

    for index, (energy, label) in enumerate(zip(energy_levels, labels)):
        ax.hlines(
            energy,
            x_position - half_width,
            x_position + half_width,
            color=color,
            linewidth=2.2,
        )
        ax.annotate(
            f"{label}: {energy:.6f}",
            xy=(x_position + half_width, energy),
            xytext=(x_position + half_width + 0.04, text_positions[index]),
            textcoords="data",
            va="center",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.55, "lw": 0.8},
        )

    ax.set(
        xlim=(0.0, 1.35),
        ylabel=ylabel,
        title=title,
    )
    ax.set_xticks([])
    all_vertical_positions = np.concatenate((energy_levels, text_positions))
    padding = max(0.05 * float(np.ptp(all_vertical_positions)), 0.25)
    ax.set_ylim(np.min(all_vertical_positions) - padding, np.max(all_vertical_positions) + padding)
    return ax


def plot_rf_dressing_comparison(
        axes,
        bare_energies_mhz,
        dressed_energies_mhz,
        bare_labels=None,
        dressed_labels=None,
        ):
    """Compare an undressed manifold with its rotating-frame quasienergies."""
    plot_energy_level_diagram(
        axes[0],
        bare_energies_mhz,
        labels=bare_labels,
        title=r"Bare $m_s=-1$ manifold",
        ylabel="Energy relative to manifold center (MHz)",
        color="tab:gray",
    )
    plot_energy_level_diagram(
        axes[1],
        dressed_energies_mhz,
        labels=dressed_labels,
        title="RF-dressed rotating-frame quasienergies",
        ylabel="Quasienergy (MHz)",
        color="tab:purple",
    )
    return axes


def plot_control_pulse_envelopes(
        ax,
        time_us,
        rf_envelope_mhz,
        mw_envelope_mhz,
        mw_start_us=None,
        ):
    """Plot laboratory control envelopes without obscuring them by the carriers."""
    ax.plot(time_us, rf_envelope_mhz, lw=2.2, label="RF dressing envelope")
    ax.plot(time_us, mw_envelope_mhz, lw=2.2, label="MW envelope")
    if mw_start_us is not None:
        ax.axvline(
            mw_start_us,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="MW begins",
        )
    ax.set(
        xlabel=r"Time ($\mu$s)",
        ylabel="Drive amplitude (MHz)",
        title="Continuous RF dressing with delayed microwave control",
    )
    ax.legend()
    return ax


def plot_rabi_diagnostics(
        time_us,
        populations,
        ground_labels,
        total_energy,
        figsize=(8, 10),
        ):
    """Create the standard three-panel population and energy diagnostic plot."""
    (
        P_g_minus_t,
        P_g_zero_t,
        P_g_plus_t,
        P_excited_t,
        P_ground_total_t,
        P_all_states_t,
    ) = populations

    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    plot_individual_population_check(
        axes[0],
        time_us,
        P_g_minus_t,
        P_g_zero_t,
        P_g_plus_t,
        P_excited_t,
        ground_labels,
    )
    plot_aggregate_population_checks(
        axes[1],
        time_us,
        P_ground_total_t,
        P_excited_t,
        P_all_states_t,
    )
    plot_energy_check(axes[2], time_us, total_energy)
    return fig, axes


def plot_tripod_leakage(
        ax,
        time_us,
        tripod_population,
        higher_dressed_population,
        all_state_population,
        title="Tripod-subspace leakage",
        ):
    """
    Plot population retained in the intended tripod and leaked into the two
    higher RF-dressed states.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the three population curves will be drawn.
    time_us : array-like
        Simulation or reporting times in microseconds.
    tripod_population : array-like
        Population in ``|g_->``, ``|g_0>``, ``|g_+>``, and the selected
        lowest dressed ancilla ``|e>``.
    higher_dressed_population : array-like
        Combined population in the middle and highest RF-dressed states
        ``|e_0>`` and ``|e_+>``.
    all_state_population : array-like
        Trace over the complete retained NV Hilbert space; this should remain
        one for trace-preserving evolution.
    title : str
        Descriptive title for the leakage diagnostic.

    Returns
    -------
    matplotlib.axes.Axes
        The supplied axes containing the three population-versus-time curves.
    """
    time_us = np.asarray(time_us)
    tripod_population = np.asarray(tripod_population)
    higher_dressed_population = np.asarray(higher_dressed_population)
    all_state_population = np.asarray(all_state_population)

    if not (
        time_us.shape
        == tripod_population.shape
        == higher_dressed_population.shape
        == all_state_population.shape
    ):
        raise ValueError("time and leakage-population arrays must have matching shapes")

    ax.plot(
        time_us,
        tripod_population,
        lw=2.2,
        label=r"$P_{g_-}+P_{g_0}+P_{g_+}+P_e$",
    )
    ax.plot(
        time_us,
        higher_dressed_population,
        lw=2.2,
        label=r"$P_{e_+}+P_{e_0}$",
    )
    ax.plot(
        time_us,
        all_state_population,
        color="black",
        lw=1.8,
        ls="--",
        label=r"$P_{\rm all}$",
    )
    ax.set(
        xlabel=r"Pulse duration ($\mu$s)",
        ylabel="Population",
        title=title,
        ylim=(-0.02, 1.05),
    )
    ax.legend()
    return ax

def plot_mw_pulses(SIMULATION_TIME, mw_minus_pulse, mw_zero_pulse, mw_plus_pulse):

    fig, axes = plt.subplots(3, 3, figsize=(12, 6), sharex=True)
    axes[0, 0].plot(SIMULATION_TIME, np.abs(mw_minus_pulse), alpha=0.4, color='green', label='mw pulse magnitude')
    axes[1, 0].plot(SIMULATION_TIME, np.abs(mw_zero_pulse), alpha=0.4, color='green', label='mw pulse magnitude')
    axes[2, 0].plot(SIMULATION_TIME, np.abs(mw_plus_pulse), alpha=0.4, color='green', label='mw pulse magnitude')

    axes[0, 1].plot(SIMULATION_TIME, mw_minus_pulse.real, alpha=0.4, color='red', label='mw pulse real')
    axes[1, 1].plot(SIMULATION_TIME, mw_zero_pulse.real, alpha=0.4, color='red', label='mw pulse real')
    axes[2, 1].plot(SIMULATION_TIME, mw_plus_pulse.real, alpha=0.4, color='red', label='mw pulse real')

    axes[0, 2].plot(SIMULATION_TIME, mw_minus_pulse.imag, alpha=0.4, color='blue', label='mw pulse imag')
    axes[1, 2].plot(SIMULATION_TIME, mw_zero_pulse.imag, alpha=0.4, color='blue', label='mw pulse imag')
    axes[2, 2].plot(SIMULATION_TIME, mw_plus_pulse.imag, alpha=0.4, color='blue', label='mw pulse imag')

    axes[0, 0].legend(loc='right', bbox_to_anchor=(1.5, 1))
    axes[0, 1].legend(loc='right', bbox_to_anchor=(1.5, 1))
    axes[0, 2].legend(loc='right', bbox_to_anchor=(1.5, 1))

    axes[1, 0].legend(loc='right', bbox_to_anchor=(1.5, 1))
    axes[1, 1].legend(loc='right', bbox_to_anchor=(1.5, 1))
    axes[1, 2].legend(loc='right', bbox_to_anchor=(1.5, 1))

    axes[2, 0].legend(loc='right', bbox_to_anchor=(1.5, 1))
    axes[2, 1].legend(loc='right', bbox_to_anchor=(1.5, 1))
    axes[2, 2].legend(loc='right', bbox_to_anchor=(1.5, 1))

    return (fig, axes)