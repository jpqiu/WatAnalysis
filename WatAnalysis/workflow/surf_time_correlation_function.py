# SPDX-License-Identifier: LGPL-3.0-or-later
from scipy.optimize import curve_fit
from MDAnalysis.lib.distances import calc_angles, capped_distance
from typing import Optional, Tuple

import numpy as np
#from scipy import special
from scipy.stats import linregress

from WatAnalysis import utils, waterdynamics
from WatAnalysis.workflow.base import (
    OneDimCoordSingleAnalysis,
    PlanarInterfaceAnalysisBase,
)
from WatAnalysis.workflow.dipole import DipoleBaseSingleAnalysis


class FluxCorrelationFunction(OneDimCoordSingleAnalysis):
    """
    Calculate the flux correlation function for a given selection of atoms.
    Ref: Limmer, D. T., et al. J. Phys. Chem. C 2015, 119 (42), 24016-24024.


    Parameters
    ----------
    selection : str
        Atom selection string used by MDAnalysis.core.universe.Universe.select_atoms(selection)
    label : str
        Label to identify the intermediate results in the analysis object
    interval_i : Tuple[Optional[float], Optional[float]],
        interval for initial state
    interval_f : Tuple[Optional[float], Optional[float]],
        interval for final state
    """

    def __init__(
        self,
        selection: str,
        label: str,
        interval_i: Tuple[Optional[float], Optional[float]],
        interval_f: Tuple[Optional[float], Optional[float]],
        exclude_number: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(selection=selection, label=label)

        self.interval_i = interval_i
        self.interval_f = interval_f
        self.exclude_number = exclude_number
        self.acf_kwargs = kwargs

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        super()._conclude(analyser)

        mask_lo, mask_hi = utils.get_region_masks(
            self.r_wrapped.squeeze(),
            analyser.r_surf_lo,
            analyser.r_surf_hi,
            self.interval_i,
        )
        mask_i = mask_lo | mask_hi
        # convert bool to float
        indicator_i = mask_i.astype(float)[:, :, np.newaxis]

        mask_lo, mask_hi = utils.get_region_masks(
            self.r_wrapped.squeeze(),
            analyser.r_surf_lo,
            analyser.r_surf_hi,
            self.interval_f,
        )
        mask_f = mask_lo | mask_hi
        # convert bool to float
        indicator_f = mask_f.astype(float)[:, :, np.newaxis]

        tau, cf = waterdynamics.calc_vector_correlation(
            vector_a=indicator_i,
            vector_b=indicator_f,
            **self.acf_kwargs,
        )
        self.results.tau = tau
        self.results.cf = cf / (
            np.mean(indicator_i) - self.exclude_number / self.ag.n_atoms
        )


class SurvivalProbability(OneDimCoordSingleAnalysis):
    """
    Calculate the flux correlation function for a given selection of atoms.
    Ref: Limmer, D. T., et al. J. Phys. Chem. C 2015, 119 (42), 24016-24024.


    Parameters
    ----------
    selection : str
        Atom selection string used by MDAnalysis.core.universe.Universe.select_atoms(selection)
    label : str
        Label to identify the intermediate results in the analysis object
    interval : Tuple[Optional[float], Optional[float]],
        The interval for selection of atoms
    exclude_number : int
        Number of atoms to exclude from the calculation
        Useful when there are some atoms need to be excluded from the calculation
    """

    def __init__(
        self,
        selection: str,
        label: str,
        interval: Tuple[Optional[float], Optional[float]],
        exclude_number: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(selection=selection, label=label)

        self.interval = interval
        self.exclude_number = exclude_number
        self.acf_kwargs = kwargs

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        super()._conclude(analyser)

        mask_lo, mask_hi = utils.get_region_masks(
            self.r_wrapped.squeeze(),
            analyser.r_surf_lo,
            analyser.r_surf_hi,
            self.interval,
        )
        mask = mask_lo | mask_hi
        # convert bool to float
        ad_indicator = mask.astype(float)[:, :, np.newaxis]
        self.acf_kwargs["normalize"] = False
        tau, cf = waterdynamics.calc_vector_autocorrelation(
            vectors=ad_indicator, **self.acf_kwargs
        )

        self.results.tau = tau
        self.results.cf = (cf - self.exclude_number / self.ag.n_atoms) / (
            np.mean(ad_indicator) - self.exclude_number / self.ag.n_atoms
        )


class WaterReorientation(DipoleBaseSingleAnalysis):
    def __init__(
        self,
        selection_oxygen: str,
        selection_hydrogen: str,
        label: str,
        interval: Tuple[Optional[float], Optional[float]],
        **kwargs,
    ) -> None:
        super().__init__(
            selection_oxygen=selection_oxygen,
            selection_hydrogen=selection_hydrogen,
            label=label,
            interval=interval,
        )

        self.acf_kwargs = kwargs

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        super()._conclude(analyser)

        mask_lo, mask_hi = utils.get_region_masks(
            self.r_wrapped.squeeze(),
            analyser.r_surf_lo,
            analyser.r_surf_hi,
            self.interval,
        )
        mask_cn = getattr(analyser, f"cn_{self.label}") == 2
        mask = (mask_lo | mask_hi) & mask_cn.squeeze()

        # normalize the dipole vectors
        vectors = getattr(analyser, f"dipole_{self.label}")
        vectors = vectors / np.linalg.norm(vectors, axis=2)[:, :, np.newaxis]
        # replace nan with 0
        vectors[np.isnan(vectors)] = 0
        self.acf_kwargs.update({"normalize": False})
        tau, cf = waterdynamics.calc_vector_autocorrelation(
            vectors=vectors,
            mask=mask,
            modifier_func=np.poly1d(special.legendre(2)),
            **self.acf_kwargs,
        )
        self.results.tau = tau
        self.results.cf = cf


class Surface2DMSD(OneDimCoordSingleAnalysis):
    def __init__(
        self,
        selection: str,
        label: str,
        interval: Tuple[Optional[float], Optional[float]],
        dt: float = 0.0005,
        min_probability: float = 0.9,
        max_lag_fraction: float = 0.3,
        **kwargs,
    ):
        super().__init__(selection=selection, label=label)
        self.interval = interval
        self.dt = dt
        self.min_probability = min_probability
        self.max_lag_fraction = max_lag_fraction
        self.acf_kwargs = kwargs

        self.traj_x = None
        self.traj_y = None

    def _fft_msd_1d(self, positions: np.ndarray, n_max_lag: int) -> np.ndarray:
        """Compute 1D MSD using FFT for a single atom trajectory."""
        n_frames = positions.shape[0]
        msd = np.zeros(n_max_lag + 1)
        for dim in range(positions.shape[1]):  # x or y
            pos_dim = positions[:, dim]
            padded_len = 2 * n_frames
            padded = np.zeros(padded_len)
            padded[:n_frames] = pos_dim

            fft_pos = np.fft.fft(padded)
            autocorr = np.fft.ifft(fft_pos * np.conj(fft_pos)).real[:n_frames]

            mean_r2 = np.mean(pos_dim**2)
            for lag in range(n_max_lag + 1):
                if lag == 0:
                    msd[lag] += 0.0
                else:
                    n_pairs = n_frames - lag
                    msd[lag] += 2 * (mean_r2 - autocorr[lag] / n_pairs)
        return msd

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        super()._conclude(analyser)
        n_frames = self.r_wrapped.shape[0]
        n_atoms = self.r_wrapped.shape[1]

        # --- Identify atoms in surface region ---
        mask_lo, mask_hi = utils.get_region_masks(
            self.r_wrapped.squeeze(),
            analyser.r_surf_lo,
            analyser.r_surf_hi,
            self.interval
        )
        mask = mask_lo | mask_hi
        probabilities = mask.sum(axis=0) / n_frames
        selected_indices = np.where(probabilities >= self.min_probability)[0]

        if len(selected_indices) == 0:
            raise RuntimeError("No atoms satisfy the minimum probability condition.")

        atom_subset = analyser.universe.select_atoms(self.selection)
        atom_group = atom_subset[selected_indices]
        n_selected = len(selected_indices)

        # --- Construct XY trajectory arrays ---
        self.traj_x = np.zeros((n_frames, n_selected))
        self.traj_y = np.zeros((n_frames, n_selected))
        for t, ts in enumerate(analyser.universe.trajectory):
            coords = atom_group.positions[:, :2]
            self.traj_x[t, :] = coords[:, 0]
            self.traj_y[t, :] = coords[:, 1]

        # --- Compute max lag ---
        max_lag = min(int(n_frames * self.max_lag_fraction), n_frames // 2)

        # --- Compute MSD using FFT for each atom ---
        msd_x = np.zeros(max_lag + 1)
        msd_y = np.zeros(max_lag + 1)
        for i in range(n_selected):
            msd_x += self._fft_msd_1d(self.traj_x[:, i:i+1], max_lag)
            msd_y += self._fft_msd_1d(self.traj_y[:, i:i+1], max_lag)
        msd_x /= n_selected
        msd_y /= n_selected
        msd_xy = msd_x + msd_y

        # --- Generate time array ---
        time_ps = np.arange(max_lag + 1) * self.dt

        # --- Fit first half for diffusion coefficient ---
        fit_idx = max_lag // 2
        slope_xy, _, _, _, _ = linregress(time_ps[1:fit_idx], msd_xy[1:fit_idx])
        slope_x, _, _, _, _ = linregress(time_ps[1:fit_idx], msd_x[1:fit_idx])
        slope_y, _, _, _, _ = linregress(time_ps[1:fit_idx], msd_y[1:fit_idx])

        D_2D = slope_xy / 4.0
        D_xx = slope_x / 2.0
        D_yy = slope_y / 2.0

        # --- Store results ---
        self.results.time_ps = time_ps
        self.results.msd_xy = msd_xy
        self.results.msd_x = msd_x
        self.results.msd_y = msd_y
        self.results.D_2D_cm2s = D_2D * 1e-4
        self.results.D_xx_cm2s = D_xx * 1e-4
        self.results.D_yy_cm2s = D_yy * 1e-4


class HydrogenBondLifetime(OneDimCoordSingleAnalysis):
    """
    Calculate hydrogen bond lifetime using the Stable State Picture (SSP) approach.

    Parameters
    ----------
    o_selection : str
        Atom selection string for oxygen atoms (e.g., "resname SOL and name O").
    h_selection : str
        Atom selection string for hydrogen atoms (e.g., "resname SOL and name H").
    label : str
        Label identifying this analysis.
    interval : tuple of float
        Distance interval relative to surface defining region of interest.
    R_strict : float
        O–O cutoff (Å) for a stable hydrogen bond (default: 3.2 Å).
    A_strict : float
        H–O–O angle cutoff (°) for a stable hydrogen bond (default: 20°).
    R_break : float
        O–O cutoff (Å) for a broken hydrogen bond (default: 3.8 Å).
    A_break : float
        H–O–O angle cutoff (°) for a broken hydrogen bond (default: 40°).
    dt : float
        Time step between frames (ps).
    max_lag_fraction : float
        Fraction of total frames to use as maximum lag time.
    """

    def __init__(
        self,
        selection_oxygen: str,
        selection_hydrogen: str,
        label: str,
        interval: Tuple[Optional[float], Optional[float]] = (0.0, 5.0),
        R_strict: float = 3.2,
        A_strict: float = 20.0,
        R_break: float = 3.8,
        A_break: float = 40.0,
        dt: float = 0.0005,
        max_lag_fraction: float = 0.3,
    ):
        super().__init__(selection=selection_oxygen, label=label)
        self.selection_hydrogen = selection_hydrogen
        self.interval = interval
        self.R_strict = R_strict
        self.A_strict = A_strict
        self.R_break = R_break
        self.A_break = A_break
        self.dt = dt
        self.max_lag_fraction = max_lag_fraction

    def _conclude(self, analyser: PlanarInterfaceAnalysisBase):
        """Compute hydrogen bond existence and SSP lifetime."""
        super()._conclude(analyser)  # 保证 self.r_wrapped 已生成

        # --- 选择界面 O 原子 ---
        r_wrapped = self.r_wrapped.squeeze()
        mask_lo, mask_hi = utils.get_region_masks(
            r_wrapped, analyser.r_surf_lo, analyser.r_surf_hi, self.interval
        )
        mask = mask_lo | mask_hi
        probabilities = mask.sum(axis=0) / analyser.n_frames
        selected_indices = np.where(probabilities >= 0.9)[0]

        if len(selected_indices) == 0:
            raise RuntimeError("No oxygen atoms found in the interface region.")

        ag_O = analyser.universe.select_atoms(self.selection)[selected_indices]
        ag_H = analyser.universe.select_atoms(self.selection_hydrogen)

        n_frames = analyser.n_frames
        max_pairs = len(ag_O) * 4
        hbond_exist = np.zeros((n_frames, max_pairs), dtype=bool)
        pair_id_map = {}
        next_pair_index = 0

        # --- 遍历每一帧 ---
        for frame_idx, ts in enumerate(analyser.universe.trajectory):
            O_pos = ag_O.positions
            H_pos = ag_H.positions

            # O-H pairs
            OH_pairs = capped_distance(O_pos, H_pos, max_cutoff=1.3, box=analyser.universe.dimensions, return_distances=False)
            if OH_pairs.size == 0:
                continue

            donors = O_pos[OH_pairs[:, 0]]
            hydrogens = H_pos[OH_pairs[:, 1]]

            # O-O distances
            OO_pairs, OO_dist = capped_distance(donors, O_pos, max_cutoff=self.R_break, box=analyser.universe.dimensions, return_distances=True)
            if OO_pairs.size == 0:
                continue

            donor_atoms = donors[OO_pairs[:, 0]]
            acceptor_atoms = O_pos[OO_pairs[:, 1]]
            angles = np.rad2deg(calc_angles(hydrogens[OO_pairs[:, 0]], donor_atoms, acceptor_atoms, box=analyser.universe.dimensions))

            # 严格 H-bond
            strict_mask = (OO_dist < self.R_strict) & (angles < self.A_strict)
            donor_indices = OH_pairs[OO_pairs[:, 0], 0]
            acceptor_indices = OO_pairs[:, 1]

            for d_idx, a_idx, keep in zip(donor_indices, acceptor_indices, strict_mask):
                if not keep:
                    continue
                pair_id = (int(d_idx), int(a_idx))
                if pair_id not in pair_id_map:
                    pair_id_map[pair_id] = next_pair_index
                    next_pair_index += 1
                col_idx = pair_id_map[pair_id]
                if col_idx < hbond_exist.shape[1]:
                    hbond_exist[frame_idx, col_idx] = True

        # --- SSP correlation function ---
        n_pairs = next_pair_index
        hbond_exist = hbond_exist[:, :n_pairs]
        max_lag = int(n_frames * self.max_lag_fraction)
        c_t = np.zeros(max_lag)

        for lag in range(max_lag):
            valid_frames = n_frames - lag
            if valid_frames <= 0:
                break
            overlap = np.sum(hbond_exist[:valid_frames] & hbond_exist[lag:lag + valid_frames], axis=1)
            h0_sum = np.sum(hbond_exist[:valid_frames], axis=1)
            nonzero = h0_sum > 0
            c_t[lag] = np.mean(overlap[nonzero] / h0_sum[nonzero]) if np.any(nonzero) else 0.0

        # --- Exponential fit ---
        def exp_decay(t, tau):
            return np.exp(-t / tau)

        t_ps = np.arange(max_lag) * self.dt
        valid = c_t > 0.1 * c_t[0]
        try:
            popt, _ = curve_fit(exp_decay, t_ps[valid], c_t[valid], p0=[5.0])
            lifetime = popt[0]
        except Exception:
            lifetime = np.nan

        self.results.time_ps = t_ps
        self.results.correlation = c_t
        self.results.lifetime_ps = lifetime

