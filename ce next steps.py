def ce_next(
    M: Dict[Tuple[int, int], np.array], 
    block_sizes_new: List[int], 
    close_blocks: List[Set[int]]
) -> Tuple[
    List[int], 
    int, 
    List[Set[int]], 
    List[Dict[Tuple[int, int], np.array]], 
    List[Dict[Tuple[int, int], np.array]], 
    List[np.array], 
    Dict[Tuple[int, int], np.array]
]:
    """
    Perform the next iteration of the algorithm.

    Parameters:
    M (dict): The matrix in sparse block format.
    block_sizes_new (list): The new sizes of the blocks.
    close_blocks (list): The close blocks.

    Returns:
    tuple: Contains the following elements:
        - block_sizes_new (list): The new sizes of the blocks.
        - M_size (int): The number of block rows.
        - close_blocks (list): The close blocks.
        - M_L_array (list): The M_L matrices with only non-zero and non-unit elements.
        - M_R_array (list): The M_R matrices with only non-zero and non-unit elements.
        - Q_U (list): The Q matrices with only non-zero and non-unit elements.
        - M (dict): The updated matrix.
    """
    M_L_array = []
    M_R_array = []
    Q_U = []

    M_size = len(block_sizes_new)
    block_sizes_new = []

    for iter in tqdm(range(M_size)):
        nonzero_line, nonzero_col = get_nonzero_line_and_column_indexes(M, iter)

        # Compression
        U, M, r, close_blocks = compress(
            M, M_size, iter, 10**(-6), close_blocks, nonzero_line, nonzero_col
        )
        Q_U.append(U)
        block_sizes_new.append(r)

        # Elimination
        M_L, M, M_R, close_blocks = eliminate(
            M, iter, block_sizes[iter], r, M_size, close_blocks,
            nonzero_line, nonzero_col
        )
        M_L_array.append(M_L)
        M_R_array.append(M_R)

    return block_sizes_new, M_size, close_blocks, M_L_array, M_R_array, Q_U, M
