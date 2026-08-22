import numpy as np
from quaccatoo import NV
from qutip import Qobj

def extract_ancilla_coefficients(
        nv: NV,
        excited_basis: tuple[Qobj, ...],
        rf_rabi_mhz: float = 1.0,
        detunings_mhz: np.ndarray | None = None,
        ancilla_branch: str = "highest",
        return_details: bool = False,
        ):
    """
    Construct the rotating-frame RF-dressed Hamiltonian in a selected
    three-state manifold and return a selected isolated dressed state.

    $$\mathcal B_e=(|{-1,-1}\rangle,|{-1,0}\rangle,|{-1,+1}\rangle). $$

    The RF matrix elements are obtained from QuaCCAToo's built-in
    ``nv.RF_h1`` operator. Under resonant driving and the rotating-wave
    approximation, the effective Hamiltonian is

    $$H_{\rm dress}=\operatorname{diag}(\Delta_m)
      +\frac{\Omega_{\rm RF}}{2}H_{\rm RF}^{(e)}.$$

    The diagonal detunings are zero for two resonant RF tones.
    ``ancilla_branch="highest"`` selects the largest dressed quasienergy,
    while ``ancilla_branch="lowest"`` selects the smallest. For equal
    resonant couplings, the highest branch gives

    $$|e\rangle=\frac{1}{2}|{-1,-1}\rangle+\frac{1}{\sqrt2}|{-1,0}\rangle+\frac{1}{2}|{-1,+1}\rangle,$$

    up to an irrelevant global phase. ``return_details=True`` also returns
    the dressed eigenvalues, eigenvectors, projected RF matrix, and dressed
    Hamiltonian.
    """

    if len(excited_basis) != 3:
        raise ValueError("excited_basis must contain exactly three states")
    if not np.isreal(rf_rabi_mhz) or rf_rabi_mhz <= 0:
        raise ValueError("rf_rabi_mhz must be a positive real number")
    if ancilla_branch not in {"lowest", "highest"}:
        raise ValueError("ancilla_branch must be either 'lowest' or 'highest'")

    # RF control operator projected onto the selected three-state manifold.
    H_rf_excited = np.array([
        [complex(bra.dag() * nv.RF_h1 * ket) for ket in excited_basis]
        for bra in excited_basis
    ], dtype=complex)

    if detunings_mhz is None:
        detunings_mhz = np.zeros(3, dtype=float)
    detunings_mhz = np.asarray(detunings_mhz, dtype=float)
    if detunings_mhz.shape != (3,):
        raise ValueError("detunings_mhz must have shape (3,)")

    # Remove diagonal RF terms: resonant magnetic-dipole dressing uses the
    # transition matrix elements between adjacent nuclear-spin states.
    H_rf_transitions = H_rf_excited - np.diag(np.diag(H_rf_excited))
    H_dressed = np.diag(detunings_mhz) + 0.5 * rf_rabi_mhz * H_rf_transitions

    dressed_eigenvalues, dressed_eigenvectors = np.linalg.eigh(H_dressed)
    if ancilla_branch == "lowest":
        ancilla_index = int(np.argmin(dressed_eigenvalues))
    else:
        ancilla_index = int(np.argmax(dressed_eigenvalues))
    ancilla_coefficients = dressed_eigenvectors[:, ancilla_index].astype(complex)

    # Fix the arbitrary global phase so the largest component is real and positive.
    phase_reference = ancilla_coefficients[np.argmax(np.abs(ancilla_coefficients))]
    ancilla_coefficients *= np.exp(-1j * np.angle(phase_reference))
    ancilla_coefficients /= np.linalg.norm(ancilla_coefficients)

    print("Projected RF control matrix in the supplied excited basis:")
    print(np.real_if_close(H_rf_excited))
    print("Rotating-frame dressed Hamiltonian (MHz):")
    print(np.real_if_close(H_dressed))
    print("Dressed quasienergies (MHz):", np.real_if_close(dressed_eigenvalues))
    print(f"Selected {ancilla_branch} dressed-state coefficients:", np.real_if_close(ancilla_coefficients))

    assert np.allclose(H_rf_excited, H_rf_excited.conj().T)
    assert np.allclose(H_dressed, H_dressed.conj().T)
    assert np.isclose(np.linalg.norm(ancilla_coefficients), 1.0)

    if return_details:
        return (
            ancilla_coefficients,
            dressed_eigenvalues,
            dressed_eigenvectors,
            H_rf_excited,
            H_dressed,
        )
    return ancilla_coefficients
