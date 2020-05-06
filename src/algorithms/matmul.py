def matmul(A, B, C, A_left_top, B_left_top, C_left_top, n):
    """
    Does standard 3-nested for-loop matrix multiplication at a particular
    location in array. Returns nothing, but submatrix C is modified when
    function returns
    """
    a_left, a_top = A_left_top
    b_left, b_top = B_left_top
    c_left, c_top = C_left_top

    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[c_left+i, c_top+j] += A[a_left+i, a_top+k] * B[b_left+k, b_top+j]


def left_top_submatrices(left_top_tuple, n_half):
    """
    Calculates the coordinates of the top left corner of the four submatrices
    and returns these as four tuples of two coordinates
    """
    left, top = left_top_tuple
    M00 = (left, top)
    M01 = (left+n_half, top)
    M10 = (left, top+n_half)
    M11 = (left+n_half, top+n_half)
    return (M00, M01, M10, M11)


def mm_recursive(A, B, C, A_left_top, B_left_top, C_left_top, n, r):
    """
    Recursively does matrix multiplication in the different blocks until recursion
    depth r is reached.
    """
    if (r <= 0):  # Standard matrix multiplication
        matmul(A, B, C, A_left_top, B_left_top, C_left_top, n)
    else:
        n_half = int(n / 2)
        A00, A01, A10, A11 = left_top_submatrices(A_left_top, n_half)
        B00, B01, B10, B11 = left_top_submatrices(B_left_top, n_half)
        C00, C01, C10, C11 = left_top_submatrices(C_left_top, n_half)

        r1 = r - 1
        mm_recursive(A, B, C, A00, B00, C00, n_half, r1)
        mm_recursive(A, B, C, A01, B10, C00, n_half, r1)
        mm_recursive(A, B, C, A00, B01, C01, n_half, r1)
        mm_recursive(A, B, C, A01, B11, C01, n_half, r1)
        mm_recursive(A, B, C, A10, B00, C10, n_half, r1)
        mm_recursive(A, B, C, A11, B10, C10, n_half, r1)
        mm_recursive(A, B, C, A10, B01, C11, n_half, r1)
        mm_recursive(A, B, C, A11, B11, C11, n_half, r1)


def matmul_rec(A, B, r):
    """
    Entry point for calling the recursive matrix multiplication to some recursion
    depth r. A and B should both be n x n matrices where n = 2^t and implement
    empty_of_same_shape(). Returns the result of the matrix multiplication
    """
    C = B.empty_of_same_shape()
    n = B.shape[0]
    mmRecursive(A, B, C, (0, 0), (0, 0), (0, 0), n, r)
    return C
