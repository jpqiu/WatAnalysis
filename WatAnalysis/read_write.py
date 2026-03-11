import warnings
import MDAnalysis as mda
from ase import Atoms
from ectoolkits.structures.slab import Slab

def universe_to_ase_atoms(u):
    """Convert MDAnalysis Universe to ASE Atoms object (for first frame)."""
    u.trajectory[0]  # take only the first frame
    positions = u.atoms.positions
    symbols = u.atoms.names
    if hasattr(u, 'dimensions') and u.dimensions is not None:
        cell = u.dimensions[:3]
        pbc = [True, True, True]
    else:
        cell = None
        pbc = None
    return Atoms(positions=positions, symbols=symbols, cell=cell, pbc=pbc)

def prepare_universe(
    traj_path: str,
    fmt: str = "LAMMPSDUMP",
    element_map: dict = None,
    dt: float = 0.0005,
    cell_size: list = None,  # [lx, ly, lz]
    top_ids: list = None,
    bottom_ids: list = None,
    auto_detect_surface: bool = True,
    surface_element: str = "Cu",
    verbose: bool = True
):
    """
    Prepare MDAnalysis Universe and surface atom IDs.

    Parameters
    ----------
    traj_path : str
        Path to the trajectory file
    fmt : str
        File format, e.g., 'LAMMPSDUMP', 'XYZ'
    element_map : dict, optional
        Mapping from atom type (int) to element symbol (str)
    dt : float
        Time step for trajectory
    cell_size : list, optional
        Box dimensions for XYZ files [lx, ly, lz]
    top_ids, bottom_ids : list, optional
        Manual surface atom IDs
    auto_detect_surface : bool
        If True, automatically detect top/bottom surface atoms using Slab
    surface_element : str
        Element to use for surface detection
    verbose : bool
        Print info
    """
    # --- 1️⃣ Load Universe ---
    try:
        if fmt.upper() in ["LAMMPS", "LAMMPSDUMP"]:
            u = mda.Universe(traj_path, format="LAMMPSDUMP", dt=dt)
            if element_map is None:
                element_map = {1: "Cu", 2: "H", 3: "O", 4: "Cl", 5: "K"}
            names = [element_map.get(int(t), "X") for t in u.atoms.types]
            u.add_TopologyAttr("names", names)

        elif fmt.upper() in ["XYZ", "PDB"]:
            u = mda.Universe(traj_path, format=fmt, dt=dt)
            if cell_size is not None:
                u.dimensions = cell_size
            else:
                warnings.warn("Format is XYZ but 'cell_size' is None. PBC handling may fail!")

        else:
            raise ValueError(f"Unsupported format: {fmt}")

    except Exception as e:
        raise RuntimeError(f"Failed to load universe from {traj_path}") from e

    # --- 2️⃣ Determine Surface Indices ---
    final_surf_ids = []

    # Method A: Manual specification
    if top_ids is not None and bottom_ids is not None:
        if verbose: print(f"[Info] Using manually provided surface IDs.")
        final_surf_ids = [bottom_ids, top_ids]

    # Method B: Auto detection using ASE + Slab (only first frame)
    elif auto_detect_surface:
        if verbose: print(f"[Info] Auto-detecting surface atoms for element: {surface_element}...")
        atoms_ase = universe_to_ase_atoms(u)
        slab = Slab(atoms_ase)
        # dw: the botton slab after being wrapped or the botton slab of the upper electrode without being wrapped. 
        idx_top = slab.find_surf_idx(
            element=surface_element, tolerance=1.5, dsur="dw", check_cross_boundary=True
        )
        idx_bottom = slab.find_surf_idx(
            element=surface_element, tolerance=1.5, dsur="up", check_cross_boundary=True
        )
        ids_bottom = [atoms_ase[i].index + 1 for i in idx_bottom]  # convert to 1-based ID if needed
        ids_top = [atoms_ase[i].index + 1 for i in idx_top]
        final_surf_ids = [ids_bottom, ids_top]
        if verbose:
            print(f" -> Found Top surface atoms: {len(ids_top)}")
            print(f" -> Found Bottom surface atoms: {len(ids_bottom)}")

    else:
        raise ValueError("Must provide top_ids/bottom_ids or enable auto_detect_surface=True.")

    return u, final_surf_ids

