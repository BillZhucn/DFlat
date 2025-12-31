import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import re
from .load_utils import load_optical_model
from .latent import latent_to_param


def find_closest_r_indices_batched(
    library, library_lambdas, target_phase, at_lambda, batch_size=10000
):
    """
    Efficiently find the index r for each pair in target_phase and at_lambda such that
    library[r, closest_lambda] is closest to exp(1j * target_phase), with internal batching.

    Parameters:
    - library: 2D NumPy array of shape (n_r, n_lam) with complex numbers.
    - library_lambdas: 1D NumPy array of shape (n_lam,) in meters.
    - target_phase: 1D NumPy array of shape (N,) with phase values (in radians).
    - at_lambda: 1D NumPy array of shape (N,) with wavelengths (in meters).
    - batch_size: int, number of pixels per batch to process.

    Returns:
    - r_indices: 1D NumPy array of shape (N,) with radius indices.
    """
    library = np.asarray(library)
    library_lambdas = (np.asarray(library_lambdas) * 1e9).astype(int)
    target_phase = np.asarray(target_phase)
    at_lambda = (np.asarray(at_lambda) * 1e9).astype(int)

    # Build fast LUT for wavelength index lookup
    lambda_to_index = np.full(library_lambdas.max() + 1, -1, dtype=np.int32)
    for lam in np.unique(at_lambda):
        lambda_to_index[lam] = np.abs(library_lambdas - lam).argmin()

    # Prepare output
    r_indices_all = np.empty_like(target_phase, dtype=np.int32)

    for start in range(0, len(target_phase), batch_size):
        end = start + batch_size
        batch_phase = target_phase[start:end]
        batch_lambda = at_lambda[start:end]

        desired_complex = np.exp(1j * batch_phase)
        nearest_indices = lambda_to_index[batch_lambda]  # vectorized lookup

        selected_columns = library[:, nearest_indices]  # (n_r, B)
        differences = np.abs(selected_columns - desired_complex[np.newaxis, :]) ** 2
        r_indices_all[start:end] = differences.argmin(axis=0)

    return r_indices_all


def reverse_lookup_optimize(
    amp,
    phase,
    wavelength_set_m,
    model_name,
    lr=1e-1,
    err_thresh=1e-2,
    max_iter=1000,
    opt_phase_only=False,
    force_cpu=False,
    batch_size=None,
    pbounds=[0.0, 1.0],
):
    """Given a stack of wavelength dependent amplitude and phase profiles, runs a reverse optimization to identify the nanostructures that
    implements the desired profile across wavelength by minimizing the mean absolute errors of complex fields.

    Args:
        amp (float): Target amplitude of shape [B, Pol, Lam, H, W].
        phase (float): Target phase of shape [B, Pol, Lam, H, W].
        wavelength_set_m (list): List of wavelengths corresponding to the Lam dimension of the target profiles.
        model_name (str): Model name. Either in the local path "DFlat/Models/NAME/" or to be retrieved from online.
        lr (float, optional): Optimization learning rate. Defaults to 1e-1.
        err_thresh (float, optional): Early termination threshold. Defaults to 0.1.
        max_iter (int, optional): Maximum number of steps. Defaults to 1000.
        batch_size (int, optional): Number of cells to evaluate at once via model batching.
        pbounds (list, optional): Min and max range [0,1] of the library params

    Returns:
        list: Returns normalized and unnormalized metasurface design parameters of shape [B, H, W, D] where D is the number of shape parameters. Last item in list is the MAE loss for each step.
    """
    B, P, L, H, W = amp.shape
    assert amp.shape == phase.shape
    assert (
        len(wavelength_set_m) == L
    ), "Wavelength list should match amp,phase wavelength dim (dim3)."

    if force_cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running optimization with device {device}")
    model = load_optical_model(model_name).to(device)
    pg = model.dim_out // 3
    assert pg == P, f"Polarization dimension of amp, phase (dim1) expected to be {pg}."

    shape_dim = model.dim_in - 1
    # z = np.random.rand(B, H, W, shape_dim)
    z = np.zeros((B, H, W, shape_dim))
    z = torch.tensor(z, device=device, dtype=torch.float32, requires_grad=True)

    wavelength = (
        torch.tensor(wavelength_set_m)
        if not torch.is_tensor(wavelength_set_m)
        else wavelength_set_m
    )
    wavelength = wavelength.to(dtype=torch.float32, device=device)
    wavelength = model.normalize_wavelength(wavelength)
    torch_zero = torch.tensor(0.0, dtype=z.dtype, device=device)
    amp = (
        torch.tensor(amp, dtype=torch.float32, device=device)
        if not torch.is_tensor(amp)
        else amp.to(dtype=torch.float32, device=device)
    )
    phase = (
        torch.tensor(phase, dtype=torch.float32, device=device)
        if not torch.is_tensor(phase)
        else phase.to(dtype=torch.float32, device=device)
    )
    target_field = torch.complex(amp, torch_zero) * torch.exp(
        torch.complex(torch_zero, phase)
    )

    # Optimize
    err = 1e3
    steps = 0
    err_list = []
    optimizer = optim.AdamW([z], lr=lr)
    pbar = tqdm(total=max_iter, desc="Optimization Progress")
    while err > err_thresh:
        if steps >= max_iter:
            pbar.close()
            break

        optimizer.zero_grad()
        pred_amp, pred_phase = model(
            latent_to_param(z, pmin=pbounds[0], pmax=pbounds[1]),
            wavelength,
            pre_normalized=True,
            batch_size=batch_size,
        )

        if opt_phase_only:
            loss = torch.mean(
                torch.abs(
                    torch.exp(torch.complex(torch_zero, pred_phase))
                    - torch.exp(torch.complex(torch_zero, phase))
                )
            )
        else:
            pred_field = torch.complex(pred_amp, torch_zero) * torch.exp(
                torch.complex(torch_zero, pred_phase)
            )
            loss = torch.mean(torch.abs(pred_field - target_field))
        loss.backward()
        optimizer.step()
        err = loss.item()
        steps += 1
        err_list.append(err)
        pbar.update(1)
        pbar.set_description(f"Loss: {err:.4f}")
    pbar.close()

    op_param = (
        latent_to_param(z, pmin=pbounds[0], pmax=pbounds[1]).detach().cpu().numpy()
    )
    return op_param, model.denormalize(op_param), err_list


def _infer_pitch_height_from_name(model_name):
    """Attempt to infer unit-cell pitch and height (in meters) from a model name like ``Nanocylinders_TiO2_U300H600``."""
    pitch_m = None
    height_m = None
    pitch_match = re.search(r"_U(\d+)", model_name)
    height_match = re.search(r"H(\d+)", model_name)
    if pitch_match:
        pitch_m = float(pitch_match.group(1)) * 1e-9
    if height_match:
        height_m = float(height_match.group(1)) * 1e-9
    return pitch_m, height_m


def _infer_material_from_name(model_name):
    """Infer a simple material label (e.g., TiO2, Si3N4) from the model name."""
    match = re.search(r"(TiO2|Si3N4|SiO2|Si)", model_name)
    return match.group(1) if match else None


def export_geometry_to_comsol(
    params_m,
    model_name,
    mph_client=None,
    mph_model=None,
    pitch_m=None,
    height_m=None,
    component="comp1",
    geometry="geom1",
    model_label="dflat_metasurface",
    z_offset=0.0,
    material_tag=None,
    material_label=None,
    show_progress=True,
    debug=False,
    log_every=10000,
    save_path=None,
):
    """Build a COMSOL 3D geometry from reverse-optimized parameters using the ``mph`` package.

    This helper assumes the reverse optimizer output has already been denormalized to meters
    (e.g., the second element returned by :func:`reverse_lookup_optimize`). For single-parameter
    cells a cylinder is created (radius = param); for two-parameter cells a block is created
    (size x/y = params). The cells are placed on a square grid with pitch inferred from the
    model name (``_U###``) unless ``pitch_m`` is provided directly.

    Args:
        params_m (array): Metasurface parameters in meters with shape [H, W, D] or [B, H, W, D].
        model_name (str): Name of the optical model used for inference; used to infer pitch/height.
        mph_client (mph.Client, optional): Existing mph client. If omitted a new one is started.
        mph_model (mph.Model, optional): Existing mph model to populate. If omitted a new model is created.
        pitch_m (float, optional): Unit-cell pitch (meters). Overrides inference from ``model_name``.
        height_m (float, optional): Feature height (meters). Overrides inference from ``model_name``.
        component (str, optional): COMSOL component tag to use/create.
        geometry (str, optional): COMSOL geometry tag to use/create.
        model_label (str, optional): Label for a newly created COMSOL model.
        z_offset (float, optional): Starting z-position for the extruded features (meters).
        material_tag (str, optional): Material tag to create/select; defaults to "mat1" if needed.
        material_label (str, optional): Material label to set; defaults to an inferred material from the model name.

    Debug Args:
        show_progress (bool, optional): If True, display a progress bar while
            populating the unit cells.
        debug (bool, optional): If True, print per-chunk timing and per-cell creation tags.
        log_every (int, optional): Print a timing heartbeat every N cells when debug is True.
    """
    import time

    try:
        import mph
    except ImportError as exc:
        raise ImportError(
            "The `mph` package is required to export geometry to COMSOL. "
            "Install it and ensure a COMSOL server is reachable."
        ) from exc

    arr = np.asarray(params_m)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError("Expected params_m with shape [H, W, D] or [B, H, W, D].")

    h_cells, w_cells, d_params = arr.shape
    inferred_pitch, inferred_height = _infer_pitch_height_from_name(model_name)
    inferred_material = _infer_material_from_name(model_name)
    pitch_m = pitch_m or inferred_pitch
    height_m = height_m or inferred_height
    if pitch_m is None:
        raise ValueError("pitch_m could not be inferred from model_name; please provide it.")
    if height_m is None:
        raise ValueError("height_m could not be inferred from model_name; please provide it.")
    if save_path is None:
        raise ValueError("save_path must be provided to save the COMSOL model.")

    if mph_model is None:
        mph_client = mph_client or mph.start()
        mph_model = mph_client.create(model_label)
    comp_root = mph_model.java.component()
    try:
        comp_root.create(component, True)
    except Exception:
        pass
    comp = mph_model.java.component(component)
    geom_root = comp.geom()
    try:
        geom_root.create(geometry, 3)
    except Exception:
        pass
    geom = comp.geom(geometry)

    # Clear any previous metasurface features under this geometry tag.
    try:
        for tag in list(geom.feature().tags()):
            geom.feature(tag).remove()
    except Exception:
        pass

    total_cells = h_cells * w_cells
    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None
        if tqdm:
            pbar = tqdm(total=total_cells, desc="Exporting unit cells")

    start_time = time.perf_counter()
    for iy in range(h_cells):
        for ix in range(w_cells):
            if debug and (iy * w_cells + ix) % log_every == 0:
                print(
                    f"[export_geometry_to_comsol] at cell {iy}/{h_cells}, {ix}/{w_cells} "
                    f"elapsed={time.perf_counter() - start_time:.1f}s"
                )
            feature_params = arr[iy, ix]
            cx = (ix - w_cells / 2 + 0.5) * pitch_m
            cy = (iy - h_cells / 2 + 0.5) * pitch_m
            tag = f"cell_{iy}_{ix}"

            if d_params == 1:
                radius = float(feature_params[0])
                if debug:
                    print(f"[export_geometry_to_comsol] create Cylinder {tag}")
                geom.create(tag, "Cylinder")
                geom.feature(tag).set("r", radius)
                geom.feature(tag).set("h", height_m)
                geom.feature(tag).set("pos", [cx, cy, z_offset])
            elif d_params == 2:
                sx, sy = [float(v) for v in feature_params]
                if debug:
                    print(f"[export_geometry_to_comsol] create Block {tag}")
                geom.create(tag, "Block")
                geom.feature(tag).set("size", [sx, sy, height_m])
                geom.feature(tag).set("pos", [cx - sx / 2, cy - sy / 2, z_offset])
            else:
                raise ValueError(
                    f"Unsupported number of parameters per cell ({d_params}). "
                    "Only 1 (cylinder radius) or 2 (block x/y) are supported."
                )
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    if debug:
        print(
            f"[export_geometry_to_comsol] Finished creating {total_cells} cells in "
            f"{time.perf_counter() - start_time:.1f}s"
        )

    mph_model.save(save_path, format="Comsol")
    
    print(f"\nCOMSOL model saved as {save_path}!")

    return


def reverse_lookup_optimize_rcwa(
    target_amp,
    target_phase,
    wavelength_set_m,
    rcwa_kwargs,
    lr=1e-2,
    err_thresh=1e-3,
    max_iter=500,
    binarize=True,
):
    """By Codex without proofreading & testing, do not trust blindly.
    
    Topology optimization wrapper that uses the RCWA solver instead of a neural surrogate.

    This assumes the RCWA solver is parameterized by a binary (or grayscale) pattern of
    shape [Layers, Nx, Ny] and that the target field is specified at the zero-order port.

    Args:
        target_amp (array): Desired amplitude of shape [Pol=2, Lam, Px, Py].
        target_phase (array): Desired phase (rad) of shape [Pol=2, Lam, Px, Py].
        wavelength_set_m (array): Wavelengths used in the RCWA simulation.
        rcwa_kwargs (dict): Keyword args forwarded to RCWA_Solver (e.g., thetas, phis,
            pte, ptm, pixelsX, pixelsY, PQ, lux, luy, layer_heights, layer_embed_mats,
            material_dielectric, Nx, Ny, er1, er2, ...).
        lr (float): Learning rate for Adam.
        err_thresh (float): Early stopping threshold on MAE.
        max_iter (int): Maximum optimization steps.
        binarize (bool): If True, optimizes unconstrained logits and uses sigmoid to keep
            patterns in [0,1]. Set False to allow unconstrained grayscale values.

    Returns:
        tuple: (optimized_pattern, loss_history) where optimized_pattern has shape
            [Layers, Nx, Ny] on CPU.
    """
    from dflat.rcwa import RCWA_Solver

    device = "cuda" if torch.cuda.is_available() else "cpu"
    solver = RCWA_Solver(wavelength_set_m=wavelength_set_m, **rcwa_kwargs).to(device)

    # Target field
    target_amp = torch.tensor(target_amp, dtype=torch.float32, device=device)
    target_phase = torch.tensor(target_phase, dtype=torch.float32, device=device)
    zero = torch.tensor(0.0, dtype=target_amp.dtype, device=device)
    target_field = torch.complex(target_amp, zero) * torch.exp(
        torch.complex(zero, target_phase)
    )

    # Design variables
    logits = torch.zeros(
        (solver.Nlayers, solver.Nx, solver.Ny),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    optimizer = optim.Adam([logits], lr=lr)
    err_list = []
    for _ in range(max_iter):
        optimizer.zero_grad()
        pattern = torch.sigmoid(logits) if binarize else logits
        pred_field = solver(pattern, ref_field=True)
        loss = torch.mean(torch.abs(pred_field - target_field))
        loss.backward()
        optimizer.step()

        err = loss.item()
        err_list.append(err)
        if err < err_thresh:
            break

    final_pattern = torch.sigmoid(logits).detach().cpu().numpy() if binarize else logits.detach().cpu().numpy()
    return final_pattern, err_list
