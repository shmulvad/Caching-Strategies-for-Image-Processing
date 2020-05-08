def convolve_pixel(picture, kernel, kernel_dim, pad, x, y, cval):
    """Performs spatial convolution at a single pixel in a 2D image"""
    pixel = 0.0
    for i in range(kernel_dim):
        for j in range(kernel_dim):
            x_val, y_val = x + i - pad, y + j - pad
            if picture.valid_index(x_val, y_val):
                pixel += kernel[i, j] * picture[x_val, y_val]
            else:
                pixel += kernel[i, j] * cval
    return pixel


def convolve_voxel(picture, kernel, kernel_dim, pad, x, y, z, cval):
    """Performs spatial convolution at a single voxel in a 3D image"""
    voxel = 0.0
    for i in range(kernel_dim):
        for j in range(kernel_dim):
            for h in range(kernel_dim):
                x_val, y_val, z_val = x + i - pad, y + j - pad, z + h - pad
                if picture.valid_index(x_val, y_val, z_val):
                    voxel += kernel[i, j, h] * picture[x_val, y_val, z_val]
                else:
                    voxel += kernel[i, j, h] * cval
    return voxel


def convolution_2d(picture, kernel, cval=0.0):
    """Performs convolution for a 2D image. Pads borders with cval"""
    assert picture.dim == 2
    ret_data = picture.empty_of_same_shape()
    kernel_dim = kernel.shape[0]
    pad = kernel_dim // 2
    for (x, y) in picture.iter_keys():
        ret_data[x, y] = convolve_pixel(picture, kernel, kernel_dim,
                                        pad, x, y, cval)
    return ret_data


def convolution_3d(picture, kernel, cval=0.0):
    """Performs convolution for a 3D image. Pads borders with cval"""
    assert picture.dim == 3
    ret_data = picture.empty_of_same_shape()
    kernel_dim = kernel.shape[0]
    pad = kernel_dim // 2
    for (x, y, z) in picture.iter_keys():
        ret_data[x, y, z] = convolve_voxel(picture, kernel, kernel_dim,
                                           pad, x, y, z, cval)
    return ret_data


def convolution(picture, kernel, cval=0.0):
    """Performs convolution for a 2D or 3D image. Pads borders with cval"""
    assert picture.dim in [2, 3], \
        f"Convolution for pictures of dim {picture.dim} not implemented"
    return convolution_2d(picture, kernel, cval) \
        if picture.dim == 2 \
        else convolution_3d(picture, kernel, cval)
