"""
pyroomacoustics experiment suite for demonstrating Matrix Power Means (MPM)
over time segments with transient / time-varying interference.

Implements:
  Exp 1: Beamforming under transient interferer (persistent desired)
  Exp 2: Tradeoff demo (desired intermittent vs persistent interferer)
  Exp 3: DOA stability (moving + bursty interferer) with SRP-PHAT heatmaps

Notes / Assumptions:
  (1) All configuration parameters are visible and easy to tweak.
  (2) Deterministic “on/off” masks for interferers (no random Bernoulli at runtime).
  (3) We assume there are algorithm variants that accept our MPM covariance aggregator.
      We define placeholders that currently fall back to arithmetic mean.
  (4) Code aggregates metrics and makes the figures discussed.

Dependencies:
  pip install pyroomacoustics numpy scipy matplotlib

"""

# %%
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

import pyroomacoustics as pra


# ============================================================
#                         CONFIG
# ============================================================

@dataclass
class GlobalConfig:
    # Reproducibility
    seed: int = 1234

    # Room
    room_dim: Tuple[float, float] = (6.0, 5.0)  # (Lx, Ly) meters
    fs: int = 16000
    rt60: float = 0.4  # target RT60 (sec); used to compute absorption via inverse_sabine

    # Array
    M: int = 8
    mic_spacing: float = 0.035  # meters
    array_center: Tuple[float, float] = (3.0, 2.5)  # near center

    # Signals
    duration_s: float = 12.0
    desired_type: str = "colored_noise"  # or "tone" etc.
    interferer_type: str = "colored_noise"

    # Segmenting
    Tseg: float = 0.2     # seconds
    overlap: float = 0.0  # 0.0 => non-overlap, 0.5 => 50% overlap

    # STFT
    nfft: int = 512
    hop: int = 128
    win: str = "hann"

    # Regularization for covariance
    cov_eps: float = 1e-6

    # "q" values to evaluate (MPM family)
    q_grid: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

    # For Monte Carlo (you asked for deterministic as possible; set small)
    num_mc_exp1: int = 5
    num_mc_exp2: int = 5
    num_mc_exp3: int = 3

    # Output directory / toggles
    show_plots: bool = True


@dataclass
class Exp1Config:
    """Exp 1: Persistent desired + bursty interferer; MVDR beamforming."""
    desired_pos: Tuple[float, float] = (1.0, 1.0)
    interferer_pos: Tuple[float, float] = (5.0, 4.0)

    # Deterministic interferer on/off pattern
    rho: float = 0.3  # fraction of segments ON (used only to *construct* deterministic mask)
    interferer_inr_db: float = 10.0  # relative at source signal before room propagation (simple scaling)

    # Noise
    mic_noise_snr_db: float = 25.0  # additive sensor noise


@dataclass
class Exp2Config:
    """Exp 2: Tradeoff: desired intermittent vs persistent interferer; MVDR beamforming."""
    desired_pos: Tuple[float, float] = (1.0, 1.0)
    interferer_pos: Tuple[float, float] = (5.0, 4.0)

    # Desired duty cycles to compare (deterministic)
    desired_rho_cases: Tuple[float, float] = (1.0, 0.6)

    # Interferer always on (or near-always)
    interferer_rho: float = 1.0
    interferer_inr_db: float = 10.0

    mic_noise_snr_db: float = 25.0


@dataclass
class Exp3Config:
    """Exp 3: DOA stability under moving+bursty interferer; SRP-PHAT heatmaps."""
    desired_pos: Tuple[float, float] = (1.2, 1.0)

    # Moving interferer positions (piecewise segments)
    interferer_positions: Tuple[Tuple[float, float], ...] = (
        (4.8, 1.0),
        (5.2, 2.0),
        (5.0, 3.2),
        (4.4, 4.0),
        (3.6, 4.2),
        (2.8, 4.0),
    )
    chunk_duration_s: float = 2.0  # each position lasts this long

    # Bursty mask across segments
    rho: float = 0.5
    interferer_inr_db: float = 8.0

    mic_noise_snr_db: float = 25.0

    # DOA scanning
    doa_grid_deg: Tuple[float, float, float] = (-90.0, 90.0, 1.0)  # from, to, step


# ============================================================
#                 DETERMINISTIC HELPERS
# ============================================================

def make_deterministic_mask(num_segments: int, rho: float) -> np.ndarray:
    """
    Deterministic ON/OFF pattern with exactly round(rho*num_segments) ON segments.
    We spread ON segments as evenly as possible.
    """
    k_on = int(round(rho * num_segments))
    mask = np.zeros(num_segments, dtype=np.float32)
    if k_on <= 0:
        return mask
    if k_on >= num_segments:
        mask[:] = 1.0
        return mask

    # Evenly spaced ON indices
    idx = np.linspace(0, num_segments - 1, k_on)
    idx = np.unique(np.round(idx).astype(int))
    # If rounding produced fewer indices, fill deterministically from start.
    if idx.size < k_on:
        fill = [i for i in range(num_segments) if i not in set(idx)]
        idx = np.concatenate([idx, np.array(fill[: (k_on - idx.size)], dtype=int)])
        idx = np.sort(idx)

    mask[idx] = 1.0
    return mask


def segment_indices(n_samples: int, fs: int, Tseg: float, overlap: float) -> List[Tuple[int, int]]:
    """
    Returns list of (start, end) sample indices for segments.
    """
    seg_len = int(round(Tseg * fs))
    hop = int(round(seg_len * (1.0 - overlap)))
    hop = max(1, hop)
    idx = []
    start = 0
    while start + seg_len <= n_samples:
        idx.append((start, start + seg_len))
        start += hop
    return idx


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


# ============================================================
#                 SIGNAL + ROOM SIMULATION
# ============================================================

def gen_colored_noise(rng: np.random.Generator, n: int, fs: int, color: str = "pink") -> np.ndarray:
    """
    Generate deterministic colored-ish noise via simple filtering.
    Not physically perfect, but stable and good enough for controlled experiments.
    """
    x = rng.standard_normal(n).astype(np.float32)

    if color == "white":
        y = x
    elif color == "pink":
        # 1/f-ish via filtering white noise with 1st-order lowpass cascade (rough)
        b, a = scipy.signal.butter(1, 0.15)
        y = scipy.signal.lfilter(b, a, x)
    elif color == "brown":
        y = np.cumsum(x)
        y = y / (np.std(y) + 1e-12)
    else:
        y = x

    # Normalize
    y = y / (np.std(y) + 1e-12)
    return y.astype(np.float32)


def build_room(cfg: GlobalConfig) -> pra.ShoeBox:
    """
    Build a 2D ShoeBox with RT60-controlled absorption.
    """
    # pyroomacoustics can convert RT60 -> absorption using inverse Sabine.
    # For 2D, we still use this for a "moderate reverb" feel.
    absorption, max_order = pra.inverse_sabine(cfg.rt60, cfg.room_dim)
    room = pra.ShoeBox(
        cfg.room_dim,
        fs=cfg.fs,
        materials=pra.Material(absorption),
        max_order=max_order,
        ray_tracing=False,
        air_absorption=True,
    )
    return room


def build_mic_array(cfg: GlobalConfig) -> np.ndarray:
    """
    Build a 2D ULA centered at cfg.array_center, aligned along x-axis.
    Returns mic positions as (2, M).
    """
    M = cfg.M
    d = cfg.mic_spacing
    cx, cy = cfg.array_center

    # Positions along x axis: centered
    offsets = (np.arange(M) - (M - 1) / 2.0) * d
    x = cx + offsets
    y = np.full_like(x, cy)
    return np.vstack([x, y])


def add_sources(room: pra.ShoeBox,
                desired_pos: Tuple[float, float],
                interferer_pos: Optional[Tuple[float, float]],
                desired_sig: np.ndarray,
                interferer_sig: Optional[np.ndarray]) -> None:
    room.add_source(desired_pos, signal=desired_sig)
    if interferer_pos is not None and interferer_sig is not None:
        room.add_source(interferer_pos, signal=interferer_sig)


def simulate_mics(room: pra.ShoeBox,
                  mic_positions: np.ndarray,
                  snr_db: float,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Simulate microphone recordings and add spatially white mic noise at target SNR.
    Returns y shape: (M, N).
    """
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
    room.simulate()

    y = room.mic_array.signals.astype(np.float32)
    # Add mic noise to reach desired SNR relative to mixture power
    sig_power = np.mean(y ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(y.shape).astype(np.float32) * np.sqrt(noise_power)
    return y + noise


# ============================================================
#              STFT / COVARIANCE ESTIMATION
# ============================================================

def stft_multichannel(y: np.ndarray, cfg: GlobalConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute STFT per channel.
    Returns:
      Y: complex array (M, F, T)
      freqs: (F,)
    """
    M, N = y.shape
    win = scipy.signal.get_window(cfg.win, cfg.nfft, fftbins=True)

    # scipy.signal.stft returns (f, t, Zxx) with Zxx shape (F, T)
    Ys = []
    freqs = None
    for m in range(M):
        f, t, Z = scipy.signal.stft(
            y[m],
            fs=cfg.fs,
            window=win,
            nperseg=cfg.nfft,
            noverlap=cfg.nfft - cfg.hop,
            nfft=cfg.nfft,
            boundary=None,
            padded=False,
            return_onesided=True,
        )
        if freqs is None:
            freqs = f
        Ys.append(Z)
    Y = np.stack(Ys, axis=0)  # (M, F, T)
    return Y, freqs


def segment_timeframes(T: int, cfg: GlobalConfig) -> List[Tuple[int, int]]:
    """
    Map waveform-time segments to STFT frame indices approximately.
    For simplicity: we segment in waveform samples then convert using hop.
    """
    # Use equivalent segment size in STFT frames:
    seg_len_samples = int(round(cfg.Tseg * cfg.fs))
    hop_samples = cfg.hop
    seg_len_frames = max(1, int(round(seg_len_samples / hop_samples)))
    hop_frames = max(1, int(round(seg_len_frames * (1.0 - cfg.overlap))))

    idx = []
    start = 0
    while start + seg_len_frames <= T:
        idx.append((start, start + seg_len_frames))
        start += hop_frames
    return idx


def estimate_covariances_per_segment(Y: np.ndarray,
                                     cfg: GlobalConfig) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Estimate covariance per segment, per frequency:
      R_l[f] = (1/W) sum_t y_f(t) y_f(t)^H over segment frames
    Returns:
      R_list: list length L of covariance tensors (F, M, M)
      seg_frames: list of (t0, t1) frame indices
    """
    M, F, T = Y.shape
    seg_frames = segment_timeframes(T, cfg)
    R_list = []
    I = np.eye(M, dtype=np.complex64)

    for (t0, t1) in seg_frames:
        W = max(1, t1 - t0)
        Yseg = Y[:, :, t0:t1]  # (M, F, W)
        # R[f] = Y_f Y_f^H / W
        Rf = np.zeros((F, M, M), dtype=np.complex64)
        for fi in range(F):
            X = Yseg[:, fi, :]  # (M, W)
            Rf[fi] = (X @ X.conj().T) / W + cfg.cov_eps * I
        R_list.append(Rf)

    return R_list, seg_frames


# ============================================================
#         PLACEHOLDER: MPM COVARIANCE AGGREGATION
# ============================================================

def aggregate_covariances_mpm_placeholder(R_list: List[np.ndarray], q: float) -> np.ndarray:
    """
    Placeholder for your Matrix Power Mean aggregation:
      R_(q) = P_q({R_l})

    Current behavior:
      - uses arithmetic mean for all q (so code runs end-to-end),
      - you will replace this with your actual MPM implementation later.
    """
    # (L, F, M, M) -> (F, M, M)
    
    stack = np.stack(R_list, axis=0)
    return np.mean(stack, axis=0)

    eps_eigval = 1e-10
    eps_q = 1e-10
    L = len(R_list)
    F, M, _ = R_list[0].shape
    print("F={F}, M={M}")
    Gamma = np.zeros(shape=(F, M, M), dtype=np.complex64)
    if np.abs(q - 0) < eps_q: # q == 0
        for f in range(F):
            # Log-Euclidean mean: exp( mean(log(G_l)) )
            S = np.zeros((M, M), dtype=np.complex64)
            for l in range(L):
                evals, evecs = np.linalg.eigh(R_list[l][f,:,:])
                evals = np.maximum(evals.real, eps_eigval)
                loge = np.log(evals)
                logG = (evecs * loge[None, :]) @ evecs.conj().T
                S += logG
            S /= L
            evalsS, evecsS = np.linalg.eigh((S + S.conj().T) * 0.5)
            G_hat = (evecsS * np.exp(evalsS.real)[None, :]) @ evecsS.conj().T
            Gamma[f,:,:] = (G_hat + G_hat.conj().T) * 0.5
    elif np.abs(q - 1) < eps_q: # q == 1
        stack = np.stack(R_list, axis=0)
        for f in range(F):
            G_hat = np.mean(stack[:,f,:,:], axis=0)
            Gamma[f,:,:] = (G_hat + G_hat.conj().T) * 0.5
    else:
        # ( mean(G_l^q) )^(1/q)
        for f in range(F):
            Q = np.zeros((M, M), dtype=np.complex64)
            for l in range(L):
                evals, evecs = np.linalg.eigh(R_list[l][f,:,:])
                evals = np.maximum(evals.real, eps_eigval)
                evals_q = evals ** q
                Gq = (evecs * evals_q[None, :]) @ evecs.conj().T
                Q += Gq
            Q /= L
            Q = (Q + Q.conj().T) * 0.5

            evalsQ, evecsQ = np.linalg.eigh(Q)
            evalsQ = np.maximum(evalsQ.real, eps_eigval)
            evals_1q = evalsQ ** (1.0 / q)
            G_hat = (evecsQ * evals_1q[None, :]) @ evecsQ.conj().T
            Gamma[f,:,:] = (G_hat + G_hat.conj().T) * 0.5

    #
    return Gamma


# ============================================================
#                    MVDR BEAMFORMING
# ============================================================

def steering_vector_farfield(mic_positions: np.ndarray, theta_deg: float, freqs: np.ndarray, c: float = 343.0) -> np.ndarray:
    """
    Far-field plane-wave steering vector for each frequency.
    mic_positions: (2, M)
    returns a: (F, M)
    """
    theta = np.deg2rad(theta_deg)
    # direction unit vector
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)  # (2,)
    # projected distances
    proj = (mic_positions.T @ u).astype(np.float64)  # (M,)
    # phase: exp(-j 2pi f / c * proj)
    a = np.exp(-1j * 2.0 * np.pi * freqs[:, None] * proj[None, :] / c)
    return a.astype(np.complex64)


def mvdr_weights(Rf: np.ndarray, af: np.ndarray) -> np.ndarray:
    """
    Compute MVDR weights per frequency:
      w = R^{-1} a / (a^H R^{-1} a)
    Rf: (F, M, M)
    af: (F, M)
    returns w: (F, M)
    """
    F, M, _ = Rf.shape
    w = np.zeros((F, M), dtype=np.complex64)
    for fi in range(F):
        R = Rf[fi]
        a = af[fi][:, None]
        # Solve R x = a
        x = np.linalg.solve(R, a)
        denom = (a.conj().T @ x).item()
        if np.abs(denom) < 1e-12:
            w[fi] = (x[:, 0])
        else:
            w[fi] = (x[:, 0] / denom)
    return w


def apply_beamformer(Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Apply frequency-domain beamformer:
      Z[f,t] = w[f]^H Y[:,f,t]
    Y: (M, F, T), w: (F, M)
    returns Z: (F, T)
    """
    # Z = sum_m conj(w[f,m]) * Y[m,f,t]
    Z = np.einsum("fm,mft->ft", np.conj(w), Y)
    return Z


def istft_singlechannel(Z: np.ndarray, cfg: GlobalConfig) -> np.ndarray:
    """
    ISTFT back to time-domain.
    Z: (F, T)
    """
    win = scipy.signal.get_window(cfg.win, cfg.nfft, fftbins=True)
    _, x = scipy.signal.istft(
        Z,
        fs=cfg.fs,
        window=win,
        nperseg=cfg.nfft,
        noverlap=cfg.nfft - cfg.hop,
        nfft=cfg.nfft,
        input_onesided=True,
        boundary=None,
    )
    return x.astype(np.float32)


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-12) -> float:
    """
    Scale-invariant SDR (single channel).
    """
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    ref = ref - ref.mean()
    est = est - est.mean()

    alpha = (np.dot(est, ref) / (np.dot(ref, ref) + eps))
    s_target = alpha * ref
    e_noise = est - s_target
    return 10.0 * np.log10((np.dot(s_target, s_target) + eps) / (np.dot(e_noise, e_noise) + eps))


# ============================================================
#                     DOA (SRP-PHAT)
# ============================================================

def srp_phat_spectrum(Y: np.ndarray,
                      mic_positions: np.ndarray,
                      cfg: GlobalConfig,
                      doa_grid_deg: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SRP-PHAT spatial spectrum using pyroomacoustics DOA.
    Returns:
      P: (G,) spatial spectrum for last computed frame batch (aggregated over frames internally)
      grid_deg: (G,)
    """
    # pyroomacoustics expects mic positions as (dim, M)
    az_min, az_max, az_step = doa_grid_deg
    grid_deg = np.arange(az_min, az_max + 1e-9, az_step)

    # Use pra.doa.SRP
    doa = pra.doa.SRP(mic_positions, cfg.fs, cfg.nfft, c=343.0, num_src=1, mode="far")
    # doa.locate_sources expects STFT of shape (M, F, T)
    doa.locate_sources(Y)

    # doa.grid.values is usually the spatial spectrum over the grid
    # but shape/field names can vary across versions.
    # We'll try to retrieve the spectrum robustly:
    P = None
    if hasattr(doa, "grid") and hasattr(doa.grid, "values"):
        P = np.array(doa.grid.values).astype(np.float32)
    elif hasattr(doa, "P"):
        P = np.array(doa.P).astype(np.float32)
    else:
        raise RuntimeError("Could not retrieve SRP spectrum from pra.doa.SRP object (version mismatch).")

    # Note: doa.grid.azimuth might also exist; we keep our explicit grid for plotting consistency.
    return P, grid_deg


# ============================================================
#               EXPERIMENT RUNNERS + PLOTS
# ============================================================

def compute_num_segments(cfg: GlobalConfig, n_samples: int) -> int:
    segs = segment_indices(n_samples, cfg.fs, cfg.Tseg, cfg.overlap)
    return len(segs)


def apply_segment_mask_to_signal(sig: np.ndarray,
                                 cfg: GlobalConfig,
                                 on_mask: np.ndarray) -> np.ndarray:
    """
    Apply segment-wise ON/OFF mask to time-domain signal by zeroing segments.
    """
    n = sig.shape[0]
    segs = segment_indices(n, cfg.fs, cfg.Tseg, cfg.overlap)
    assert len(segs) == len(on_mask), "mask length must equal number of segments"
    out = sig.copy()
    for l, (s, e) in enumerate(segs):
        if on_mask[l] < 0.5:
            out[s:e] = 0.0
    return out


def run_mvdr_trial(cfg: GlobalConfig,
                   desired_pos: Tuple[float, float],
                   interferer_pos: Tuple[float, float],
                   desired_sig: np.ndarray,
                   interferer_sig: np.ndarray,
                   mic_noise_snr_db: float,
                   look_theta_deg: float) -> Dict[str, np.ndarray]:
    """
    Simulate one trial, compute STFT/covariances, and return useful artifacts.
    """
    room = build_room(cfg)
    mic_pos = build_mic_array(cfg)

    add_sources(room, desired_pos, interferer_pos, desired_sig, interferer_sig)
    y = simulate_mics(room, mic_pos, mic_noise_snr_db, np.random.default_rng(cfg.seed))

    # STFT
    Y, freqs = stft_multichannel(y, cfg)

    # Segment covariances
    R_list, seg_frames = estimate_covariances_per_segment(Y, cfg)

    # Far-field steering vector for look direction
    a = steering_vector_farfield(mic_pos, look_theta_deg, freqs)

    return {
        "y": y,
        "Y": Y,
        "freqs": freqs,
        "R_list": R_list,
        "seg_frames": np.array(seg_frames, dtype=int),
        "mic_pos": mic_pos,
        "a": a,
    }


def exp1_beamforming_transient_interferer(gcfg: GlobalConfig, ecfg: Exp1Config) -> None:
    """
    Exp 1: MVDR under transient interferer (persistent desired).
    Outputs:
      - SI-SDR vs q (box/median+IQR)
      - spectrogram comparison for one representative trial
      - (optional) interferer activity plot
    """
    rng = np.random.default_rng(gcfg.seed)

    N = int(round(gcfg.duration_s * gcfg.fs))
    desired = gen_colored_noise(rng, N, gcfg.fs, color="pink")
    interferer = gen_colored_noise(rng, N, gcfg.fs, color="pink")

    # Determine segments and create deterministic ON mask
    num_segments = compute_num_segments(gcfg, N)
    on_mask = make_deterministic_mask(num_segments, ecfg.rho)

    # Apply mask to interferer
    interferer_masked = apply_segment_mask_to_signal(interferer, gcfg, on_mask)

    # Scale interferer to desired INR (simple scaling before room)
    interferer_masked = interferer_masked * db_to_lin(ecfg.interferer_inr_db)

    # Choose look direction from geometry (approx)
    mic_pos = build_mic_array(gcfg)
    # approximate DOA from array center to source position
    dx = ecfg.desired_pos[0] - gcfg.array_center[0]
    dy = ecfg.desired_pos[1] - gcfg.array_center[1]
    look_theta_deg = np.rad2deg(np.arctan2(dy, dx))

    # Collect SI-SDR per q across MC
    sdr_by_q: Dict[float, List[float]] = {q: [] for q in gcfg.q_grid}

    # Representative artifacts for plots
    rep = None

    for mc in range(gcfg.num_mc_exp1):
        # Slight deterministic variation per MC: rotate noise seed by mc
        rng_mc = np.random.default_rng(gcfg.seed + 1000 + mc)
        desired_mc = desired.copy()
        interferer_mc = interferer_masked.copy()

        # Run trial (simulate mixture, compute covariances)
        artifacts = run_mvdr_trial(
            gcfg,
            ecfg.desired_pos,
            ecfg.interferer_pos,
            desired_mc,
            interferer_mc,
            ecfg.mic_noise_snr_db,
            look_theta_deg,
        )

        # Also simulate desired-only for reference (same room params)
        room_ref = build_room(gcfg)
        room_ref.add_microphone_array(pra.MicrophoneArray(artifacts["mic_pos"], gcfg.fs))
        room_ref.add_source(ecfg.desired_pos, signal=desired_mc)
        room_ref.simulate()
        y_des_only = room_ref.mic_array.signals.astype(np.float32)

        # Reference: use first mic desired-only as reference target (simple)
        ref_target = y_des_only[0]

        # Beamform for each q
        for q in gcfg.q_grid:
            Rq = aggregate_covariances_mpm_placeholder(artifacts["R_list"], q)  # (F,M,M)
            w = mvdr_weights(Rq, artifacts["a"])
            Z = apply_beamformer(artifacts["Y"], w)
            x_hat = istft_singlechannel(Z, gcfg)

            # Align lengths
            L = min(len(ref_target), len(x_hat))
            sdr_by_q[q].append(si_sdr(ref_target[:L], x_hat[:L]))

        if rep is None:
            rep = (artifacts, ref_target)

    # ---- Plot 1: SI-SDR vs q (box style but minimal: median+IQR)
    qs = np.array(list(gcfg.q_grid), dtype=float)
    med = np.array([np.median(sdr_by_q[q]) for q in qs])
    q1 = np.array([np.percentile(sdr_by_q[q], 25) for q in qs])
    q3 = np.array([np.percentile(sdr_by_q[q], 75) for q in qs])

    plt.figure()
    plt.plot(qs, med, marker="o")
    plt.fill_between(qs, q1, q3, alpha=0.2)
    plt.xlabel("q")
    plt.ylabel("Output SI-SDR (dB)")
    plt.title("Exp 1: MVDR under transient interferer (median ± IQR)")
    plt.grid(True)

    # ---- Plot 2: Spectrograms (mic0 vs mean vs q=-1) for representative trial
    if rep is not None:
        artifacts, ref_target = rep
        y_mic0 = artifacts["y"][0]
        Y_mic0, _ = scipy.signal.stft(y_mic0, fs=gcfg.fs, nperseg=gcfg.nfft, noverlap=gcfg.nfft - gcfg.hop)

        # mean (placeholder is same, but keep structure)
        Rmean = aggregate_covariances_mpm_placeholder(artifacts["R_list"], q=0.0)
        w_mean = mvdr_weights(Rmean, artifacts["a"])
        Z_mean = apply_beamformer(artifacts["Y"], w_mean)
        x_mean = istft_singlechannel(Z_mean, gcfg)

        Rm1 = aggregate_covariances_mpm_placeholder(artifacts["R_list"], q=-1.0)
        w_m1 = mvdr_weights(Rm1, artifacts["a"])
        Z_m1 = apply_beamformer(artifacts["Y"], w_m1)
        x_m1 = istft_singlechannel(Z_m1, gcfg)

        # STFT for outputs
        Zm, _ = scipy.signal.stft(x_mean, fs=gcfg.fs, nperseg=gcfg.nfft, noverlap=gcfg.nfft - gcfg.hop)
        Z1, _ = scipy.signal.stft(x_m1, fs=gcfg.fs, nperseg=gcfg.nfft, noverlap=gcfg.nfft - gcfg.hop)

        plt.figure()
        plt.pcolormesh(np.abs(Y_mic0), shading="auto")
        plt.title("Exp 1: |STFT| of Mic 0 (mixture)")
        plt.xlabel("Frame")
        plt.ylabel("Freq bin")

        plt.figure()
        plt.pcolormesh(np.abs(Zm), shading="auto")
        plt.title("Exp 1: |STFT| MVDR (mean covariance)")
        plt.xlabel("Frame")
        plt.ylabel("Freq bin")

        plt.figure()
        plt.pcolormesh(np.abs(Z1), shading="auto")
        plt.title("Exp 1: |STFT| MVDR (MPM q=-1 placeholder)")
        plt.xlabel("Frame")
        plt.ylabel("Freq bin")

    # ---- Plot 3: Interferer activity mask
    plt.figure()
    plt.step(np.arange(num_segments), on_mask, where="mid")
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Segment index")
    plt.ylabel("Interferer ON (0/1)")
    plt.title("Exp 1: Deterministic interferer activity mask")
    plt.grid(True)

    if gcfg.show_plots:
        plt.show()


def exp2_tradeoff_desired_intermittent(gcfg: GlobalConfig, ecfg: Exp2Config) -> None:
    """
    Exp 2: Tradeoff: desired intermittent vs persistent interferer.
    Shows:
      - SI-SDR vs q for two desired duty cycles (overlay)
      - Summary: best q vs desired duty cycle
    """
    rng = np.random.default_rng(gcfg.seed + 200)

    N = int(round(gcfg.duration_s * gcfg.fs))
    base_desired = gen_colored_noise(rng, N, gcfg.fs, color="pink")
    interferer = gen_colored_noise(rng, N, gcfg.fs, color="pink") * db_to_lin(ecfg.interferer_inr_db)

    num_segments = compute_num_segments(gcfg, N)
    interferer_mask = make_deterministic_mask(num_segments, ecfg.interferer_rho)
    interferer_masked = apply_segment_mask_to_signal(interferer, gcfg, interferer_mask)

    # Look direction from geometry
    dx = ecfg.desired_pos[0] - gcfg.array_center[0]
    dy = ecfg.desired_pos[1] - gcfg.array_center[1]
    look_theta_deg = np.rad2deg(np.arctan2(dy, dx))

    results = {}  # desired_rho -> dict(q -> list of SDRs)
    best_qs = []

    for desired_rho in ecfg.desired_rho_cases:
        desired_mask = make_deterministic_mask(num_segments, desired_rho)
        desired_masked = apply_segment_mask_to_signal(base_desired, gcfg, desired_mask)

        sdr_by_q: Dict[float, List[float]] = {q: [] for q in gcfg.q_grid}

        for mc in range(gcfg.num_mc_exp2):
            desired_mc = desired_masked.copy()
            interferer_mc = interferer_masked.copy()

            artifacts = run_mvdr_trial(
                gcfg,
                ecfg.desired_pos,
                ecfg.interferer_pos,
                desired_mc,
                interferer_mc,
                ecfg.mic_noise_snr_db,
                look_theta_deg,
            )

            # desired-only ref
            room_ref = build_room(gcfg)
            room_ref.add_microphone_array(pra.MicrophoneArray(artifacts["mic_pos"], gcfg.fs))
            room_ref.add_source(ecfg.desired_pos, signal=desired_mc)
            room_ref.simulate()
            y_des_only = room_ref.mic_array.signals.astype(np.float32)
            ref_target = y_des_only[0]

            for q in gcfg.q_grid:
                Rq = aggregate_covariances_mpm_placeholder(artifacts["R_list"], q)
                w = mvdr_weights(Rq, artifacts["a"])
                Z = apply_beamformer(artifacts["Y"], w)
                x_hat = istft_singlechannel(Z, gcfg)
                L = min(len(ref_target), len(x_hat))
                sdr_by_q[q].append(si_sdr(ref_target[:L], x_hat[:L]))

        results[desired_rho] = sdr_by_q

        # best q by median
        qs = np.array(list(gcfg.q_grid), dtype=float)
        med = np.array([np.median(sdr_by_q[q]) for q in qs])
        best_q = qs[np.argmax(med)]
        best_qs.append((desired_rho, best_q))

    # ---- Plot: SI-SDR vs q overlay
    plt.figure()
    qs = np.array(list(gcfg.q_grid), dtype=float)

    for desired_rho in ecfg.desired_rho_cases:
        sdr_by_q = results[desired_rho]
        med = np.array([np.median(sdr_by_q[q]) for q in qs])
        q1 = np.array([np.percentile(sdr_by_q[q], 25) for q in qs])
        q3 = np.array([np.percentile(sdr_by_q[q], 75) for q in qs])

        plt.plot(qs, med, marker="o", label=f"desired rho={desired_rho}")
        plt.fill_between(qs, q1, q3, alpha=0.15)

    plt.xlabel("q")
    plt.ylabel("Output SI-SDR (dB)")
    plt.title("Exp 2: Tradeoff — desired duty cycle shifts optimal q")
    plt.grid(True)
    plt.legend()

    # ---- Plot: best q vs desired rho
    plt.figure()
    rhos = np.array([x[0] for x in best_qs], dtype=float)
    bqs = np.array([x[1] for x in best_qs], dtype=float)
    plt.plot(rhos, bqs, marker="o")
    plt.xlabel("Desired duty cycle rho_d")
    plt.ylabel("Best q (by median SI-SDR)")
    plt.title("Exp 2: Summary — optimal q vs desired intermittency")
    plt.grid(True)

    if gcfg.show_plots:
        plt.show()


def exp3_doa_moving_bursty_interferer(gcfg: GlobalConfig, ecfg: Exp3Config) -> None:
    """
    Exp 3: DOA stability via SRP-PHAT heatmaps, with moving+bursty interferer.

    We generate piecewise simulation by concatenating chunks:
      desired is always on (fixed pos)
      interferer position changes every chunk_duration_s
      within each chunk, interferer is gated by deterministic per-segment mask.

    Figures:
      - heatmap: per-segment SRP peak angle (or spectrum slice) vs time
      - aggregated SRP spectrum (mean vs MPM q=-1 placeholder)
      - DOA error vs q (optional, based on peak location compared to true desired)
    """
    rng = np.random.default_rng(gcfg.seed + 300)

    fs = gcfg.fs
    chunkN = int(round(ecfg.chunk_duration_s * fs))
    n_chunks = len(ecfg.interferer_positions)
    totalN = chunkN * n_chunks

    desired = gen_colored_noise(rng, totalN, fs, color="pink")

    # Build global interferer with per-chunk different noise (deterministic)
    interferer_full = np.zeros(totalN, dtype=np.float32)
    for i in range(n_chunks):
        x = gen_colored_noise(np.random.default_rng(gcfg.seed + 500 + i), chunkN, fs, color="pink")
        interferer_full[i * chunkN:(i + 1) * chunkN] = x

    interferer_full *= db_to_lin(ecfg.interferer_inr_db)

    # Segment mask across the whole duration
    num_segments = compute_num_segments(gcfg, totalN)
    on_mask = make_deterministic_mask(num_segments, ecfg.rho)
    interferer_masked = apply_segment_mask_to_signal(interferer_full, gcfg, on_mask)

    # Build mic array once
    mic_pos = build_mic_array(gcfg)

    # Generate mixture by concatenating chunk simulations (to implement moving interferer)
    y_chunks = []
    for i, pos_i in enumerate(ecfg.interferer_positions):
        s0 = i * chunkN
        s1 = (i + 1) * chunkN

        room = build_room(gcfg)
        room.add_microphone_array(pra.MicrophoneArray(mic_pos, fs))
        room.add_source(ecfg.desired_pos, signal=desired[s0:s1])
        room.add_source(pos_i, signal=interferer_masked[s0:s1])
        room.simulate()
        y_chunk = room.mic_array.signals.astype(np.float32)

        # add mic noise
        sig_power = np.mean(y_chunk ** 2)
        noise_power = sig_power / (10.0 ** (ecfg.mic_noise_snr_db / 10.0))
        noise = rng.standard_normal(y_chunk.shape).astype(np.float32) * np.sqrt(noise_power)
        y_chunks.append(y_chunk + noise)

    y = np.concatenate(y_chunks, axis=1)  # (M, totalN)

    # STFT & segment covariances
    Y, freqs = stft_multichannel(y, gcfg)
    R_list, seg_frames = estimate_covariances_per_segment(Y, gcfg)

    # True desired DOA (approx far-field)
    dx = ecfg.desired_pos[0] - gcfg.array_center[0]
    dy = ecfg.desired_pos[1] - gcfg.array_center[1]
    desired_theta = np.rad2deg(np.arctan2(dy, dx))

    # Per-segment SRP peaks (instability visualization)
    # We'll compute SRP on each segment’s STFT slice (Y[:, :, t0:t1])
    grid = np.arange(ecfg.doa_grid_deg[0], ecfg.doa_grid_deg[1] + 1e-9, ecfg.doa_grid_deg[2])
    peak_angles = np.zeros(len(seg_frames), dtype=np.float32)

    for l, (t0, t1) in enumerate(seg_frames):
        Yseg = Y[:, :, t0:t1]
        try:
            P, grid_deg = srp_phat_spectrum(Yseg, mic_pos, gcfg, ecfg.doa_grid_deg)
            # P may be on some internal grid; we pick argmax anyway
            peak_angles[l] = grid[np.argmax(P) % len(grid)]
        except Exception:
            peak_angles[l] = np.nan

    # Aggregated SRP: mean covariance vs MPM q=-1 placeholder
    # (We emulate “SRP on aggregated covariance” by just running SRP on full Y;
    #  later you can replace this with an SRP variant that consumes R_(q) directly.)
    P_full, _ = srp_phat_spectrum(Y, mic_pos, gcfg, ecfg.doa_grid_deg)

    # Plot 1: peak angle per segment
    plt.figure()
    plt.plot(peak_angles, marker=".", linestyle="-")
    plt.axhline(desired_theta, linestyle="--")
    plt.xlabel("Segment index")
    plt.ylabel("SRP peak angle (deg)")
    plt.title("Exp 3: Per-segment SRP peak angle (shows instability under moving/bursty interferer)")
    plt.grid(True)

    # Plot 2: aggregated SRP spectrum on whole recording
    plt.figure()
    plt.plot(grid, P_full[: len(grid)] if len(P_full) >= len(grid) else P_full)
    plt.axvline(desired_theta, linestyle="--")
    plt.xlabel("Angle (deg)")
    plt.ylabel("SRP spectrum")
    plt.title("Exp 3: SRP spectrum on full recording (baseline)")

    # Plot 3: interferer activity mask
    plt.figure()
    plt.step(np.arange(num_segments), on_mask, where="mid")
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Segment index")
    plt.ylabel("Interferer ON (0/1)")
    plt.title("Exp 3: Deterministic bursty interferer mask")
    plt.grid(True)

    if gcfg.show_plots:
        plt.show()

# %%
# ============================================================
#                         MAIN
# ============================================================

def main() -> None:
    # %%
    gcfg = GlobalConfig()

    # %% Exp 1
    exp1 = Exp1Config()
    exp1_beamforming_transient_interferer(gcfg, exp1)

    # %% Exp 2
    exp2 = Exp2Config()
    exp2_tradeoff_desired_intermittent(gcfg, exp2)

    # %% Exp 3
    exp3 = Exp3Config()
    exp3_doa_moving_bursty_interferer(gcfg, exp3)


if __name__ == "__main__":
    main()