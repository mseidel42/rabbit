import numpy as np

def _normalize_cdf(hist):
    cdf = np.cumsum(hist)
    total = cdf[-1]
    if total > 0:
        cdf = cdf / total
    else:
        cdf = np.zeros_like(cdf)
    return cdf, total

def morph_two(h_A, h_B, alpha):
    """
    Interpolate between two 1D histograms h_A and h_B by a factor alpha.
    alpha=0 -> h_A
    alpha=1 -> h_B
    """
    n_bins = len(h_A)

    cdf_A, total_A = _normalize_cdf(h_A)
    cdf_B, total_B = _normalize_cdf(h_B)

    if total_A == 0 or total_B == 0:
        h_morphed = h_A + alpha * (h_B - h_A)
        return np.maximum(0, h_morphed)

    total_morphed = total_A + alpha * (total_B - total_A)
    total_morphed = max(0, total_morphed)

    y_common = np.linspace(0, 1, max(1000, n_bins * 10))
    x_edges = np.arange(1, n_bins + 1)

    def get_inv_cdf(cdf):
        cdf_strict = cdf + np.arange(n_bins) * 1e-12
        cdf_strict = cdf_strict / cdf_strict[-1]
        p_vals = np.insert(cdf_strict, 0, 0)
        x_vals = np.insert(x_edges, 0, 0)
        return np.interp(y_common, p_vals, x_vals)

    x_A_inv = get_inv_cdf(cdf_A)
    x_B_inv = get_inv_cdf(cdf_B)

    x_morphed_inv = x_A_inv + alpha * (x_B_inv - x_A_inv)
    x_morphed_inv_strict = np.maximum.accumulate(x_morphed_inv)

    cdf_morphed = np.interp(x_edges, x_morphed_inv_strict, y_common)

    h_morphed = np.diff(np.insert(cdf_morphed, 0, 0))
    h_morphed = h_morphed * total_morphed

    return h_morphed

def horizontal_morph(h_nom, h_up, h_down, alpha):
    """
    Perform horizontal morphing (integral morphing / A.L. Read method) for 1D histograms.
    h_nom, h_up, h_down: 1D numpy arrays representing bin contents.
    alpha: interpolation parameter (0=nom, 1=up, -1=down)
    Returns: morphed 1D numpy array.
    """
    if alpha >= 0:
        return morph_two(h_nom, h_up, alpha)
    else:
        return morph_two(h_nom, h_down, -alpha)

def horizontal_morph_nd(h_nom, h_up, h_down, alpha, axis=0):
    """
    Perform horizontal morphing along a specific axis for N-dimensional histograms.
    We apply the 1D morphing along the specified axis for every combination of the other axes.
    """
    if h_nom.ndim == 1:
        return horizontal_morph(h_nom, h_up, h_down, alpha)

    h_morphed = np.zeros_like(h_nom)

    # Move the target axis to the last dimension
    nom_swapped = np.moveaxis(h_nom, axis, -1)
    up_swapped = np.moveaxis(h_up, axis, -1)
    down_swapped = np.moveaxis(h_down, axis, -1)
    morphed_swapped = np.zeros_like(nom_swapped)

    # Flatten all other dimensions
    flat_shape = (-1, nom_swapped.shape[-1])
    nom_flat = nom_swapped.reshape(flat_shape)
    up_flat = up_swapped.reshape(flat_shape)
    down_flat = down_swapped.reshape(flat_shape)
    morphed_flat = morphed_swapped.reshape(flat_shape)

    for i in range(nom_flat.shape[0]):
        morphed_flat[i] = horizontal_morph(nom_flat[i], up_flat[i], down_flat[i], alpha)

    morphed_swapped = morphed_flat.reshape(nom_swapped.shape)
    h_morphed = np.moveaxis(morphed_swapped, -1, axis)
    return h_morphed
