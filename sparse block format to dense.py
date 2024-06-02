def blocks_to_full_matrix(M, block_sizes):
    """
    Converts a dictionary representing a block matrix into a full numpy array.

    Parameters:
      M: Dictionary where key (i, j) corresponds to the block of the matrix at position (i, j).
      block_sizes: Array where block_sizes[i] specifies the dimensions of blocks in the i-th row/column.

    Returns:
      Two-dimensional numpy array composed of the blocks.
    """

    # Determine the dimensions of the full matrix
    total_rows = sum(block_sizes)
    total_cols = total_rows

    full_matrix = np.zeros((total_rows, total_cols))
    row_start = 0
    for i in range(len(block_sizes)):
        col_start = 0
        for j in range(len(block_sizes)):
            # If the block exists in the dictionary, insert it into the full matrix
            if (i, j) in M:
                block = M[(i, j)]

                # Check that the size of the block matches the expected size
                if block.shape != (block_sizes[i], block_sizes[j]):
                    raise ValueError(f"Block size at position {(i, j)} does not match the expected size.")

                full_matrix[row_start:row_start + block_sizes[i], col_start:col_start + block_sizes[j]] = block

            # Update the starting position for the next column
            col_start += block_sizes[j]
        # Update the starting position for the next row
        row_start += block_sizes[i]

    return full_matrix


def inverse_permutation(p):
    """
    Computes the inverse of a permutation array.

    :param p: A one-dimensional numpy array where p[i] = j indicates that element i moves to position j.
    :return: A numpy array representing the inverse permutation.
    """
    # Initialize the inverse permutation array with the same size as p
    p_inv = np.zeros_like(p)

    # Populate the inverse permutation array
    for i in range(len(p)):
        p_inv[p[i]] = i

    return p_inv


def apply_permutation(matrix, p):
    """
    Applies a permutation p to both the rows and columns of the matrix.

    :param matrix: A two-dimensional numpy array.
    :param p: A one-dimensional array of permutations, where p[i] = j
              means that element i should move to position j.
    :return: The matrix with the permutation applied to both rows and columns.
    """
    # Check that the permutation has the correct size
    if len(p) != matrix.shape[0]:
        raise ValueError("The size of the permutation must match the dimensions of the matrix.")

    # Apply the permutation to the rows
    permuted_matrix = matrix[p, :]

    # Apply the permutation to the columns
    permuted_matrix = permuted_matrix[:, p]

    return permuted_matrix
