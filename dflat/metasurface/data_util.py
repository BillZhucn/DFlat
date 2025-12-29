import torch
import numpy as np
import pandas as pd
import re


def build_transfer_targets(
    desired_amp,
    desired_phase,
    incident_amp,
    incident_phase,
    eps=1e-6,
):
    """Convert desired output and incident fields into per-cell transfer targets for the neural surrogate.

    The surrogate predicts a complex transmission coefficient t_i under unit, normally incident illumination.
    For an arbitrary incident field E_in, the required transfer is t_target = E_out / E_in.

    Args:
        desired_amp (array): Desired amplitude |E_out|, shape [..., H, W].
        desired_phase (array): Desired phase arg(E_out) (rad), same shape as desired_amp.
        incident_amp (array): Incident amplitude |E_in|, same shape.
        incident_phase (array): Incident phase arg(E_in) (rad), same shape.
        eps (float): Small floor to avoid division by zero when |E_in| is tiny.

    Returns:
        tuple: (amp_target, phase_target, mask) where amp_target/phase_target can be
        passed to reverse_lookup_optimize and mask is 1 where |E_in| >= eps else 0.
    """
    desired_amp_t = (
        torch.tensor(desired_amp, dtype=torch.float32)
        if not torch.is_tensor(desired_amp)
        else desired_amp.to(dtype=torch.float32)
    )
    desired_phase_t = (
        torch.tensor(desired_phase, dtype=torch.float32)
        if not torch.is_tensor(desired_phase)
        else desired_phase.to(dtype=torch.float32)
    )
    incident_amp_t = (
        torch.tensor(incident_amp, dtype=torch.float32)
        if not torch.is_tensor(incident_amp)
        else incident_amp.to(dtype=torch.float32)
    )
    incident_phase_t = (
        torch.tensor(incident_phase, dtype=torch.float32)
        if not torch.is_tensor(incident_phase)
        else incident_phase.to(dtype=torch.float32)
    )

    zero = torch.tensor(0.0, dtype=desired_amp_t.dtype, device=desired_amp_t.device)
    desired_field = torch.complex(desired_amp_t, zero) * torch.exp(
        torch.complex(zero, desired_phase_t)
    )
    incident_field = torch.complex(incident_amp_t, zero) * torch.exp(
        torch.complex(zero, incident_phase_t)
    )

    mag = torch.abs(incident_field)
    safe_incident = torch.where(mag >= eps, incident_field, torch.complex(torch.ones_like(incident_amp_t) * eps, zero))
    transfer = desired_field / safe_incident

    amp_target = torch.abs(transfer)
    phase_target = torch.angle(transfer)
    mask = (mag >= eps).to(desired_amp_t.dtype)
    return amp_target, phase_target, mask


def incident_field_to_amp_phase(Ex, Ey, Ez, pol_vector=(1.0, 1.0, 1.0)):
    """Project a vector incident field onto a polarization direction and return scalar amp/phase.

    Args:
        Ex, Ey, Ez (array): Complex field components (a + b j), matching shapes.
        pol_vector (tuple/list/np.array): Length-3 polarization direction to project onto
            (e.g., (1,0,0) for x-pol, (0,1,0) for y-pol). Will be normalized.

    Returns:
        amp = |E_proj|.
        phase = arg(E_proj).
    """
    pol = torch.tensor(pol_vector, dtype=torch.float32)
    pol = pol / torch.norm(pol)

    def to_cplx(x):
        if torch.is_tensor(x):
            return x.to(dtype=torch.complex64)
        x_np = np.asarray(x)
        return torch.tensor(x_np, dtype=torch.complex64)

    Ex_t = to_cplx(Ex)
    Ey_t = to_cplx(Ey)
    Ez_t = to_cplx(Ez)

    E_proj = pol[0] * Ex_t + pol[1] * Ey_t + pol[2] * Ez_t
    amp = torch.abs(E_proj)
    phase = torch.angle(E_proj)
    return amp, phase


def load_field_table(path):
    """Load a whitespace-delimited field file with header units into a pandas DataFrame.

    Expects a header like ``x [um] y [um] ... ExRe [V/m] ExIm [V/m]`` on the first line
    and a separator line on the second line. Returns a DataFrame with those column names.
    """
    with open(path, "r") as f:
        first_line = f.readline().strip()

    column_names = re.findall(r'([^\s\[\]]+ ?\[[^\]]+\])', first_line)
    df = pd.read_csv(path, sep=r'\s+', skiprows=[0, 1], header=None, names=column_names)
    return df
