def should_erode_pixel(picture, kernel, kernel_dim, pad, x, y):
    for i in range(kernel_dim):
        for j in range(kernel_dim):
            for h in range(kernel_dim):
                if kernel[i, j, h] > picture[x + i - pad, y + j - pad]:
                    return False
    return True


def should_erode_voxel(picture, kernel, kernel_dim, pad, x, y, z):
    for i in range(kernel_dim):
        for j in range(kernel_dim):
            for h in range(kernel_dim):
                if kernel[i, j, h] > picture[x + i - pad, y + j - pad, z + h - pad]:
                    return False
    return True


def erosion_2d(picture, kernel):
    assert picture.dim == 2
    ret_data = picture.empty_of_same_shape()
    kernel_dim = kernel.shape[0]
    pad = kernel_dim // 2
    for (x, y) in picture.iter_keys():
        if not picture.valid_index(x, y, pad=pad):
            continue
        should_erode = should_erode_pixel(picture, kernel, kernel_dim, pad, x, y)
        ret_data[x, y] = 0 if should_erode else 1
    return ret_data


def erosion_3d(picture, kernel):
    assert picture.dim == 3
    ret_data = picture.empty_of_same_shape()
    kernel_dim = kernel.shape[0]
    pad = kernel_dim // 2
    for (x, y, z) in picture.iter_keys():
        if not picture.valid_index(x, y, z, pad=pad):
            continue
        should_erode = should_erode_voxel(picture, kernel, kernel_dim, pad, x, y, z)
        ret_data[x,y,z] = 0 if should_erode else 1
    return ret_data
