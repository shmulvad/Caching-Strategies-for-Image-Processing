def convolution_pixel(picture, kernel, kernel_dim, pad, x, y):
    pixel = 0.0
    for i in range(kernel_dim):
        for j in range(kernel_dim):
            pixel += kernel[i, j] * picture[x + i - pad, y + j - pad]
    return pixel


def convolution_voxel(picture, kernel, kernel_dim, pad, x, y, z):
    voxel = 0.0
    for i in range(kernel_dim):
        for j in range(kernel_dim):
            for h in range(kernel_dim):
                voxel += kernel[i, j, h] * picture[x + i - pad, y + j - pad, z + h - pad]
    return voxel


def convolution_2d(picture, kernel):
    assert picture.dim == 2
    ret_data = picture.empty_of_same_shape()
    kernel_dim = kernel.shape[0]
    pad = kernel_dim // 2
    for (x, y) in picture.iter_keys():
        if not picture.valid_index(x, y, pad=pad):
            continue
        ret_data[x, y] = convolution_pixel(picture, kernel, kernel_dim, pad, x, y)
    return ret_data


def convolution_3d(picture, kernel):
    assert picture.dim == 3
    ret_data = picture.empty_of_same_shape()
    kernel_dim = kernel.shape[0]
    pad = kernel_dim // 2
    for (x, y, z) in picture.iter_keys():
        if not picture.valid_index(x, y, z, pad=pad):
            continue
        ret_data[x, y, z] = convolution_voxel(picture, kernel, kernel_dim, pad, x, y, z)
    return ret_data

def convolution(picture, kernel):
    return convolution_2d(picture, kernel) \
           if picture.dim == 2 \
           else convolution_3d(picture, kernel)
