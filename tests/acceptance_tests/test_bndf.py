import os
from fdsreader import Simulation

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test_bndf():
    sim = Simulation(os.path.join(TEST_DIR, "../cases/bndf_data"))
    obst = sim.obstructions.get_nearest(-0.8, 1, 1)
    face = obst.get_global_boundary_data_arrays("Wall Temperature")[1]

    assert len(face[-1]) == 68


def test_get_nearest_patch():
    """Test that get_nearest_patch returns correct face based on distance."""
    sim = Simulation(os.path.join(TEST_DIR, "../cases/bndf_data"))
    obst = sim.obstructions[0]

    # Get patches for different points around the obstruction
    # Bounding box: x=[-1, 2], y=[-1.2, 2.4], z=[-0.1, 0]
    # Use z slightly inside the obstruction for side faces to avoid edge/face ties.
    patch_x_plus = obst.get_nearest_patch(3, 0.5, -0.05)
    patch_x_minus = obst.get_nearest_patch(-2, 0.5, -0.05)
    patch_y_plus = obst.get_nearest_patch(0.5, 3, -0.05)
    patch_y_minus = obst.get_nearest_patch(0.5, -2, -0.05)
    patch_z_plus = obst.get_nearest_patch(0.5, 0.5, 1)

    # Orientations: 1=X+, 2=Y+, 3=Z+, -1=X-, -2=Y-, -3=Z-
    assert patch_x_plus.orientation == 1
    assert patch_x_minus.orientation == -1
    assert patch_y_plus.orientation == 2
    assert patch_y_minus.orientation == -2
    assert patch_z_plus.orientation == 3

    # Note: no -3 (z-minus) patch exists in this test data, so -z is not tested
