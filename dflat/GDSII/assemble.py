import numpy as np
from tqdm.auto import tqdm
import gdspy
import time
import uuid

from .gds_utils import add_marker_tag, upsample_block


def assemble_cylinder_gds(
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    gds_unit=1e-6,
    gds_precision=1e-9,
    marker_size=250e-6,
    number_of_points=9,
):
    """Generate a GDS file for nanocylinder metasurfaces.

    Args:
        params (numpy.ndarray): Nanocylinder radii across the lens, shape [H, W, 1].
        mask (numpy.ndarray): Boolean mask indicating whether to write a shape (True) or skip it (False), shape [H, W].
        cell_size (list): Cell sizes holding the nanocylinder [dy, dx].
        block_size (list): Block sizes to repeat the nanocylinders [dy', dx']. Resizing may be applied.
        savepath (str): Path to save the GDS file (including .gds extension).
        gds_unit (float, optional): GDSPY units. Defaults to 1e-6.
        gds_precision (float, optional): GDSPY precision. Defaults to 1e-9.
        marker_size (float, optional): Size of alignment markers. Defaults to 250e-6.
        number_of_points (int, optional): Number of points to represent the circular shape. Defaults to 9.

    Raises:
        ValueError: If params.shape[-1] != 1.
    """
    if params.shape[-1] != 1:
        raise ValueError("Shape dimension D encoding radius should be equal to 1.")

    assemble_standard_shapes(
        gdspy.Round,
        params,
        mask,
        cell_size,
        block_size,
        savepath,
        gds_unit,
        gds_precision,
        marker_size,
        number_of_points,
    )
    return


def assemble_ellipse_gds(
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    gds_unit=1e-6,
    gds_precision=1e-9,
    marker_size=250e-6,
    number_of_points=9,
):
    """Generate a GDS file for nano-ellipse metasurfaces.

    Args:
        params (numpy.ndarray): Ellipse radii across the lens, shape [H, W, 2] where [:,:,0] is x-radius and [:,:,1] is y-radius.
        mask (numpy.ndarray): Boolean mask indicating whether to write a shape (True) or skip it (False), shape [H, W].
        cell_size (list): Cell sizes holding the nano-ellipse [dy, dx].
        block_size (list): Block sizes to repeat the nano-ellipses [dy', dx']. Resizing may be applied.
        savepath (str): Path to save the GDS file (including .gds extension).
        gds_unit (float, optional): GDSPY units. Defaults to 1e-6.
        gds_precision (float, optional): GDSPY precision. Defaults to 1e-9.
        marker_size (float, optional): Size of alignment markers. Defaults to 250e-6.
        number_of_points (int, optional): Number of points to represent the elliptical shape. Defaults to 9.

    Raises:
        ValueError: If params.shape[-1] != 2.
    """
    if params.shape[-1] != 2:
        raise ValueError("Shape dimension D encoding radii (x,y) should be equal to 2.")

    assemble_standard_shapes(
        gdspy.Round,
        params,
        mask,
        cell_size,
        block_size,
        savepath,
        gds_unit,
        gds_precision,
        marker_size,
        number_of_points,
    )
    return


def assemble_fin_gds(
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    gds_unit=1e-6,
    gds_precision=1e-9,
    marker_size=250e-6,
    number_of_points=9,
):
    """Generate a GDS file for nanofin metasurfaces.

    Args:
        params (numpy.ndarray): Nanofin dimensions across the lens, shape [H, W, 2] where [:,:,0] is width and [:,:,1] is length.
        mask (numpy.ndarray): Boolean mask indicating whether to write a shape (True) or skip it (False), shape [H, W].
        cell_size (list): Cell sizes holding the nanofin [dy, dx].
        block_size (list): Block sizes to repeat the nanofins [dy', dx']. Resizing may be applied.
        savepath (str): Path to save the GDS file (including .gds extension).
        gds_unit (float, optional): GDSPY units. Defaults to 1e-6.
        gds_precision (float, optional): GDSPY precision. Defaults to 1e-9.
        marker_size (float, optional): Size of alignment markers. Defaults to 250e-6.

    Raises:
        ValueError: If params.shape[-1] != 2.
    """
    if params.shape[-1] != 2:
        raise ValueError(
            "Shape dimension D encoding width and length should be equal to 2."
        )

    assemble_standard_shapes(
        gdspy.Rectangle,
        params,
        mask,
        cell_size,
        block_size,
        savepath,
        gds_unit,
        gds_precision,
        marker_size,
        number_of_points,
    )
    return


def assemble_standard_shapes(
    cell_fun,
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    gds_unit=1e-6,
    gds_precision=1e-9,
    marker_size=250e-6,
    number_of_points=9,
):
    """
    Assemble standard shapes for GDS files based on given parameters.

    This function creates a GDS file containing a metasurface pattern of standard shapes
    (e.g., circles, ellipses, rectangles) based on the provided parameters and mask.

    Args:
        cell_fun (callable): GDSPY function to create the shape (e.g., gdspy.Round, gdspy.Rectangle).
        params (numpy.ndarray): Shape parameters across the lens, shape [H, W, D] where D depends on the shape type.
        mask (numpy.ndarray): Boolean mask indicating whether to write a shape (True) or skip it (False), shape [H, W].
        cell_size (list): Cell sizes holding the shape [dy, dx].
        block_size (list): Block sizes to repeat the shapes [dy', dx']. Resizing may be applied.
        savepath (str): Path to save the GDS file (including .gds extension).
        gds_unit (float, optional): GDSPY units. Defaults to 1e-6.
        gds_precision (float, optional): GDSPY precision. Defaults to 1e-9.
        marker_size (float, optional): Size of alignment markers. Defaults to 250e-6.
        number_of_points (int, optional): Number of points to represent curved shapes. Defaults to 9.

    Raises:
        ValueError: If input dimensions are incorrect or inconsistent.

    Returns:
        None
    """
    # Input validation
    if len(cell_size) != 2 or len(block_size) != 2:
        raise ValueError("cell_size and block_size must be lists of length 2.")
    if not np.all(np.greater_equal(block_size, cell_size)):
        raise ValueError("block_size must be greater than or equal to cell_size.")
    if len(params.shape) != 3 or len(mask.shape) != 2:
        raise ValueError("params must be 3D and mask must be 2D.")
    if mask.shape != params.shape[:2]:
        raise ValueError("mask shape must match the first two dimensions of params.")

    # Upsample the params to match the target blocks
    params_, mask = upsample_block(params, mask, cell_size, block_size)
    mask = mask.astype(bool)
    print(params_.shape)
    H, W, D = params_.shape

    # Write to GDS
    unique_id = str(uuid.uuid4())[:8]
    lib = gdspy.GdsLibrary(unit=gds_unit, precision=gds_precision)
    main_cell = lib.new_cell(f"MAIN_{unique_id}")
    cell_cache = {}

    print("Writing metasurface shapes to GDS File with cell reuse...")
    start = time.time()
    dy_unit = cell_size[0] / gds_unit
    dx_unit = cell_size[1] / gds_unit

    for yi in range(H):
        for xi in range(W):
            if not mask[yi, xi]:
                continue

            # Quantize shape parameters (1 nm)
            shape_params = tuple(int(round(val * 1e9)) for val in params_[yi, xi])

            if shape_params not in cell_cache:
                shape_cell = lib.new_cell(
                    f"SHAPE_{'_'.join(map(str, shape_params))}_{uuid.uuid4().hex[:4]}"
                )
                shape_params_unit = [val * 1e-9 / gds_unit for val in shape_params]

                if cell_fun == gdspy.Round:
                    if len(shape_params_unit) == 1:
                        radius = shape_params_unit[0]
                        shape = cell_fun(
                            (0, 0), radius, number_of_points=number_of_points
                        )
                    elif len(shape_params_unit) == 2:
                        rx, ry = shape_params_unit
                        shape = cell_fun(
                            (0, 0), (rx, ry), number_of_points=number_of_points
                        )
                    else:
                        raise ValueError(
                            "Unsupported shape parameter length for Round."
                        )
                elif cell_fun == gdspy.Rectangle:
                    w, h = shape_params_unit
                    shape = cell_fun((-w / 2, -h / 2), (w / 2, h / 2))
                else:
                    raise ValueError("Unsupported cell function.")

                shape_cell.add(shape)
                cell_cache[shape_params] = shape_cell

            x = xi * dx_unit
            y = yi * dy_unit
            main_cell.add(gdspy.CellReference(cell_cache[shape_params], (x, y)))

    # Wrap with top-level cell
    top_cell = lib.new_cell(f"TOP_CELL_{unique_id}")
    top_cell.add(gdspy.CellReference(main_cell))

    lib.write_gds(savepath)
    print(f"GDS file saved to {savepath}")
    print(f"Total placed shapes: {np.count_nonzero(mask)}")
    print(f"Unique shapes: {len(cell_cache)}")
    print(f"Elapsed time: {time.time() - start:.2f} seconds")


def _shape_to_polygons(shape):
    if hasattr(shape, "polygons"):
        return shape.polygons
    if hasattr(shape, "points"):
        return [shape.points]
    raise ValueError("Unsupported GDSPY shape type for DXF export.")


def gds_to_dxf(
    gds_path,
    dxf_path,
    cell_name=None,
    flatten=True,
):
    """Convert a GDS file to a DXF file for COMSOL import.

    Args:
        gds_path (str): Input GDS file path.
        dxf_path (str): Output DXF file path.
        cell_name (str, optional): Specific cell name to export. Defaults to top-level cell.
        flatten (bool, optional): Flatten references before export. Defaults to True.
    """
    try:
        import ezdxf
    except ImportError as exc:
        raise ImportError(
            "The `ezdxf` package is required to export DXF. Install it via pip."
        ) from exc

    lib = gdspy.GdsLibrary()
    lib.read_gds(gds_path)

    if cell_name is None:
        top_cells = lib.top_level()
        if not top_cells:
            raise ValueError("No top-level cells found in GDS.")
        cell = top_cells[0]
    else:
        if cell_name not in lib.cells:
            raise ValueError(f"Cell '{cell_name}' not found in GDS.")
        cell = lib.cells[cell_name]

    if flatten:
        cell = cell.copy(f"{cell.name}_FLATTENED")
        cell.flatten()

    doc = ezdxf.new()
    msp = doc.modelspace()
    polygons_by_spec = cell.get_polygons(by_spec=True)
    for (layer, _datatype), polygons in polygons_by_spec.items():
        layer_name = f"L{layer}"
        if layer_name not in doc.layers:
            doc.layers.new(layer_name)
        for pts in polygons:
            msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer_name})

    doc.saveas(dxf_path)


def assemble_standard_shapes_dxf(
    cell_fun,
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    dxf_unit=1e-6,
    marker_size=250e-6,
    number_of_points=9,
    layer="L0",
):
    """Assemble standard shapes and save directly to DXF (for COMSOL import)."""
    try:
        import ezdxf
    except ImportError as exc:
        raise ImportError(
            "The `ezdxf` package is required to export DXF. Install it via pip."
        ) from exc

    if len(cell_size) != 2 or len(block_size) != 2:
        raise ValueError("cell_size and block_size must be lists of length 2.")
    if not np.all(np.greater_equal(block_size, cell_size)):
        raise ValueError("block_size must be greater than or equal to cell_size.")
    if len(params.shape) != 3 or len(mask.shape) != 2:
        raise ValueError("params must be 3D and mask must be 2D.")
    if mask.shape != params.shape[:2]:
        raise ValueError("mask shape must match the first two dimensions of params.")

    params_, mask = upsample_block(params, mask, cell_size, block_size)
    mask = mask.astype(bool)
    H, W, _ = params_.shape

    doc = ezdxf.new()
    msp = doc.modelspace()
    if layer not in doc.layers:
        doc.layers.new(layer)

    dy_unit = cell_size[0] / dxf_unit
    dx_unit = cell_size[1] / dxf_unit

    shape_cache = {}
    print("Writing metasurface shapes to DXF File...")
    start = time.time()

    for yi in range(H):
        for xi in range(W):
            if not mask[yi, xi]:
                continue

            shape_params = tuple(int(round(val * 1e9)) for val in params_[yi, xi])
            if shape_params not in shape_cache:
                shape_params_unit = [val * 1e-9 / dxf_unit for val in shape_params]
                if cell_fun == gdspy.Round:
                    if len(shape_params_unit) == 1:
                        radius = shape_params_unit[0]
                        shape = cell_fun(
                            (0, 0), radius, number_of_points=number_of_points
                        )
                    elif len(shape_params_unit) == 2:
                        rx, ry = shape_params_unit
                        shape = cell_fun(
                            (0, 0), (rx, ry), number_of_points=number_of_points
                        )
                    else:
                        raise ValueError(
                            "Unsupported shape parameter length for Round."
                        )
                elif cell_fun == gdspy.Rectangle:
                    w, h = shape_params_unit
                    shape = cell_fun((-w / 2, -h / 2), (w / 2, h / 2))
                else:
                    raise ValueError("Unsupported cell function.")

                shape_cache[shape_params] = _shape_to_polygons(shape)

            x = xi * dx_unit
            y = yi * dy_unit
            for pts in shape_cache[shape_params]:
                pts_xy = pts + np.array([x, y])
                msp.add_lwpolyline(
                    pts_xy,
                    close=True,
                    dxfattribs={"layer": layer},
                )

    doc.saveas(savepath)
    print(f"DXF file saved to {savepath}")
    print(f"Total placed shapes: {np.count_nonzero(mask)}")
    print(f"Unique shapes: {len(shape_cache)}")
    print(f"Elapsed time: {time.time() - start:.2f} seconds")


def assemble_cylinder_dxf(
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    dxf_unit=1e-6,
    marker_size=250e-6,
    number_of_points=9,
    layer="L0",
):
    """Generate a DXF file for nanocylinder metasurfaces."""
    if params.shape[-1] != 1:
        raise ValueError("Shape dimension D encoding radius should be equal to 1.")

    assemble_standard_shapes_dxf(
        gdspy.Round,
        params,
        mask,
        cell_size,
        block_size,
        savepath,
        dxf_unit,
        marker_size,
        number_of_points,
        layer,
    )
    return


def assemble_ellipse_dxf(
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    dxf_unit=1e-6,
    marker_size=250e-6,
    number_of_points=9,
    layer="L0",
):
    """Generate a DXF file for nano-ellipse metasurfaces."""
    if params.shape[-1] != 2:
        raise ValueError("Shape dimension D encoding radii (x,y) should be equal to 2.")

    assemble_standard_shapes_dxf(
        gdspy.Round,
        params,
        mask,
        cell_size,
        block_size,
        savepath,
        dxf_unit,
        marker_size,
        number_of_points,
        layer,
    )
    return


def assemble_fin_dxf(
    params,
    mask,
    cell_size,
    block_size,
    savepath,
    dxf_unit=1e-6,
    marker_size=250e-6,
    number_of_points=9,
    layer="L0",
):
    """Generate a DXF file for nanofin metasurfaces."""
    if params.shape[-1] != 2:
        raise ValueError(
            "Shape dimension D encoding width and length should be equal to 2."
        )

    assemble_standard_shapes_dxf(
        gdspy.Rectangle,
        params,
        mask,
        cell_size,
        block_size,
        savepath,
        dxf_unit,
        marker_size,
        number_of_points,
        layer,
    )
    return

    return


if __name__ == "__main__":
    assemble_cylinder_gds(
        np.random.rand(10, 10, 1) * 250e-9,
        np.ones((10, 10)),
        # np.random.choice([True, False], size=(10, 10)),
        [500e-9, 500e-9],
        [1e-6, 1e-6],
        "/home/deanhazineh/Research/DFlat/dflat/GDSII/out.gds",
        gds_unit=1e-6,
        gds_precision=1e-9,
        marker_size=250e-6,
        number_of_points=9,
    )
