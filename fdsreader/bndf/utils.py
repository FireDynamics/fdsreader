from typing import List

from fdsreader.bndf import Patch


def sort_patches_cartesian(patches_in: List[Patch]):
    """Returns all patches (of same orientation!) sorted in cartesian coordinates.
    """
    patches = patches_in.copy()
    if len(patches) != 0:
        patches_cart = [[patches[0]]]
        orientation = abs(patches[0].orientation)
        if orientation == 1:  # x
            patches.sort(key=lambda p: (p.extent.y_start, p.extent.z_start))
        elif orientation == 2:  # y
            patches.sort(key=lambda p: (p.extent.x_start, p.extent.z_start))
        elif orientation == 3:  # z
            patches.sort(key=lambda p: (p.extent.x_start, p.extent.y_start))

        if orientation == 1:
            for patch in patches[1:]:
                if patch.extent.y_start == patches_cart[-1][-1].extent.y_start:
                    patches_cart[-1].append(patch)
                else:
                    patches_cart.append([patch])
        else:
            for patch in patches[1:]:
                if patch.extent.x_start == patches_cart[-1][-1].extent.x_start:
                    patches_cart[-1].append(patch)
                else:
                    patches_cart.append([patch])
        return patches_cart
    return patches