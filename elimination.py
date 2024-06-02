def eliminate(
    M: Dict, iter: int, size_i: int, r: int, M_size: int,
    close_blocks: List, nonzero_line: List[int], nonzero_col: List[int],
    ) -> Tuple[Dict, Dict, Dict]:
    '''
    Eliminates the compressed parts of the iter-th row and iter-th column.

    Args:
      M: Matrix in sparse block format.
      iter: Current iteration number.
      size_i: Size of the blocks.
      r: Rank of blocks.
      M_size: the number of block rows in matrix M.
      nonzero_line: Indexes j of nonzero elements in the iter-th row: M[iter, j] != 0
      nonzero_col: Indexes i of nonzero elements in the iter-th column: M[i, iter] != 0

    Returns decomposition Q^T M_(iter - 1)^C Q = M_L M_iter M_R:
      M_L_col: M_L[iter][r:], block column in sparse block format
      M: Matrix in sparse block format
      M_R_row: M_R[iter][:, r:], block row in sparse block format
    '''

    nonzero_blocks = M.keys()
    A_3 = M[iter, iter][r:, r:]

    # raising error if diagonal block is not invertible
    try:
      A_3_inv = np.linalg.inv(A_3)
    except np.linalg.LinAlgError:
      raise ValueError('Diagonal block is not invertible')

    L, U = lu(A_3, permute_l=True)

    '''
    Matrix M_L
    We store only a column M_L[:, iter][r:]. M_L is:
    - M(i, iter)[:, r:] U^(-1) for i <= iter - 1 (here on the left is identical matrix, on the right --- zero)
    - M(iter, iter)[:r, r:] U^(-1) (same)
    - L (here both on the left and right are zero)
    - M(i, iter)[:, r:] U^(-1) for i >= iter + 1  (here on the left is zero, on the left is identical matrix)

    M(iter, iter)[:r, r:] and L are stored as one block of the size size_i
    '''

    M_L_col = {}
    U_inv = np.linalg.inv(U)
    for i in close_blocks[iter]:
      ''' far blocks in column iter are cropped so there is no [:, r:] for them
      only close blocks are nonzero at [:, r:] '''
      if i == iter:
        diag_block = np.vstack([M[iter, iter][:r, r:] @ U_inv, L])
        M_L_col[iter, iter] = diag_block.copy()
      else:
        M_L_col[i, iter] = M[i, iter][:, r:] @ U_inv

      M_L_col, _ = check_zero(M_L_col, i, iter, None)


    # Matrix M_R
    M_R_row = {}
    L_inv = np.linalg.inv(L)
    for j in close_blocks[iter]:
      if j != iter:
        M_R_row[iter, j] = L_inv @ M[iter, j][r:, :]
      else:
        diag_block = np.hstack([L_inv @ M[iter, iter][r:, :r], U])
        M_R_row[iter, iter] = diag_block.copy()

      M_R_row, _ = check_zero(M_R_row, iter, j, None)

    '''
      Eliminating blocks
      Note that some M[iter, col] with col > iter are far blocks so they are cropped
    and have M[iter, col][r:, :] = 0
      Also note that some M[line, iter] with line > iter are far blocks so they are
    cropped and have M[line, iter][:, r:] = 0
    '''

    M_C = M.copy()
    for (line, col) in itertools.product(close_blocks[iter], close_blocks[iter]):
      if (line, col) == (iter, iter):
        M_C[iter, iter] = (M[iter, iter][:r, :r] - M[iter, iter][:r, r:] @
                         A_3_inv @ M[iter, iter][r:, :r])
      elif line == iter:
        if (iter, col) in nonzero_blocks:
          M_C[iter, col] = (M[iter, col][:r, :] - M[iter, iter][:r, r:] @
                          A_3_inv @ M[iter, col][r:, :])
        else:
          M_C[iter, col] = -M[iter, iter][:r, r:] @ A_3_inv @ M[iter, col][r:, :]
      elif col == iter:
        if (iter, col) in nonzero_blocks:
          M_C[line, iter] = (M[line, iter][:, :r] - M[line, iter][:, r:] @
                           A_3_inv @ M[iter, iter][r:, :r])
        else:
          M_C[line, iter] = -(M[line, iter][:, r:] @ A_3_inv @
                            M[iter, iter][r:, :r])
      else:
        if (line, col) in nonzero_blocks:
          M_C[line, col] -= M[line, iter][:, r:] @ A_3_inv @ M[iter, col][r:, :]
        else:
          M_C[line, col] = -M[line, iter][:, r:] @ A_3_inv @ M[iter, col][r:, :]

      M_C, close_blocks = check_zero(M_C, line, col, close_blocks)

    '''
    M[iter, iter] has size[:r, :r], after it goes diagonal identity matrix
    for the rest of the block, we don't store it
    '''

    return M_L_col, M_C, M_R_row, close_blocks
