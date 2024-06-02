def compress(
    M: Dict[Tuple[int, int], np.array],
    M_size: int,
    iter: int,
    eps: float,
    close_blocks: List[Set],
    nonzero_line: List[int],
    nonzero_col: List[int],
) -> Tuple[np.array, Dict[Tuple[int, int], np.array]]:
    """
    Compresses the iter-th column and iter-th row.

    Args:
        M: Matrix (of size M_size) in sparse block format.
        M_size: The number of block rows.
        iter: The number of the current iteration.
        eps: Singular value relative tolerance for truncated SVD
        close_blocks: List of sets with close blocks indexes:
          close_blocks[line] has column indexes of nonzero blocks.
        nonzero_line: Indexes of nonzero elements in the iter-th line (iter, col).
        nonzero_col: Indexes of nonzero elements in the iter-th column (line, iter).

    Returns:
        U: Matrix for Q = diag(I_(b0 + ... + b_iter), U, 
          I_(b_iter + 2 + ... + b_(M_size))) and Q^T M Q.
        M: Updated matrix in sparse block format.
        r: Rank of far blocks
    """

    nonzero_blocks = set(M.keys())
    A = M[iter, iter]

    # Determine far line and column indices
    far_line_ind = set(nonzero_line) - set(close_blocks[iter])
    far_col_ind = set(nonzero_col) - set(close_blocks[iter])

    # Stack far blocks
    if far_line_ind:
      far_line = np.hstack([M[iter, col] for col in far_line_ind])
    if far_col_ind:
      far_col = np.vstack([M[line, iter] for line in far_col_ind]).T

    if far_line_ind and far_col_ind:
      far_blocks = np.hstack([far_line, far_col])
    elif far_line_ind:
      far_blocks = far_line
    elif far_col_ind:
      far_blocks = far_col
    else:
      return np.identity(len(A)), M, 0, close_blocks

    # Compute SVD of the far blocks
    U, S, _ = np.linalg.svd(far_blocks, full_matrices=False)

    '''
    Obtaining Q^T M Q where
    Q[0, 0] = np.identity(bl_ar[0] + ... + bl_ar[iter])
    Q[iter + 1, iter + 1] = U
    Q[i, i] = np.identity(bl_ar[i]) for i > iter
    '''

    # Update matrix M
    M[iter, iter] = U.T @ M[iter, iter] @ U
    for line in nonzero_col:
      M[line, iter] = M[line, iter] @ U
      M, close_blocks = check_zero(M, line, iter, close_blocks)

    for col in nonzero_line:
      M[iter, col] = U.T @ M[iter, col]
      M, close_blocks = check_zero(M, iter, col, close_blocks)

    # Compress far blocks
    r = len(S[S/S[0] > eps])

    for col in far_line_ind:
      M[iter, col] = M[iter, col][:r, :]
      M, close_blocks = check_zero(M, iter, col, close_blocks)

    for line in far_col_ind:
      M[line, iter] = M[line, iter][:, :r]
      M, close_blocks = check_zero(M, line, iter, close_blocks)

    return U, M, r, close_blocks
