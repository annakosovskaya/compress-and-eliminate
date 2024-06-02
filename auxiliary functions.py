def check_zero(
    M: Dict[Tuple[int, int], np.array], 
    line: int, 
    col: int, 
    close_blocks: List[Set[int]]
) -> Tuple[Dict[Tuple[int, int], np.array], List[Set[int]]]:
    """
    Checks if the matrix block at position (line, col) is zero and deletes it if true.
    Also updates the close_blocks structure if the column is in the set of close blocks for the line.

    Args:
        M (dict): The matrix in a sparse block format where keys are tuples (line, col).
        line (int): The row index in the matrix M.
        col (int): The column index in the matrix M.
        close_blocks (list): List of sets, where each set contains indices of close blocks for each row.

    Returns:
        tuple: Updated matrix M and close_blocks after removing zero blocks.
    """
    is_zero = np.all(M[line, col] == 0)
    if is_zero:
        del M[line, col]
        if close_blocks and col in close_blocks[line]:
            close_blocks[line].remove(col)

    return M, close_blocks

def get_nonzero_line_and_column_indexes(
    M: Dict[Tuple[int, int], np.array], iter: int
) -> Tuple[List[int], List[int]]:
    """
    Args:
      M: Matrix in sparse block format
      iter: Index of the current line and column to get indexes of nonzero
      elements in it

    Returns:
      nonzero_line: List of indexes of nonzero elements in the iter-th line
        -> M[iter, x] != 0 for x in nonzero_line, without block (iter, iter)
      nonzero_col: List of indexes of nonzero elements in the iter-th column,
        without block (iter, iter)
    """

    nonzero_blocks = M.keys()
    nonzero_line = [
        col for (line, col) in nonzero_blocks if line == iter and col != iter
    ]  # (iter, col) is nonzero - iter-th column
    nonzero_col = [
        line for (line, col) in nonzero_blocks if col == iter and line != iter
    ]  # (line, iter) is nonzero - iter-th row

    return nonzero_line, nonzero_col
