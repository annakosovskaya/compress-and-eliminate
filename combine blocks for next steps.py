def combine_blocks(
    M: Dict[Tuple[int, int], np.array], 
    block_sizes: List[int], 
    step: int = 2
) -> Tuple[Dict[Tuple[int, int], np.array], List[int], List[List[int]]]:
    """
    Combine adjacent blocks into larger blocks in a given matrix.

    Args:
        M (dict): Dictionary with keys as (line, col) tuples for nonzero blocks.
        block_sizes (list): List of integers defining the size of blocks in each dimension.
        step (int): Number of blocks to combine in one dimension (default is 2 for 2x2).

    Returns:
        tuple: Contains the following elements:
            - M_new (dict): A new dictionary with combined blocks.
            - new_block_sizes (list): List of integers defining the size of combined blocks.
            - close_blocks (list): List of lists containing indices of close blocks.
    """
    M_new = {}
    # Precompute new block sizes for combined blocks
    new_block_sizes = [sum(block_sizes[i:i + step]) for i in range(0, len(block_sizes), step)]

    # Iterate over the blocks to combine them
    for i in range(0, len(block_sizes), step):
        for j in range(0, len(block_sizes), step):
            new_block = None
            # Set the size of the new combined block using precomputed sizes
            new_block_row_size = new_block_sizes[i // step]
            new_block_col_size = new_block_sizes[j // step]
            for di in range(step):
                for dj in range(step):
                    block_index = (i + di, j + dj)
                    if block_index in M:
                        block = M[block_index]
                        if new_block is None:
                            new_block = np.zeros((new_block_row_size, new_block_col_size))
                        # Calculate starting indices within the new block
                        row_start = sum(block_sizes[i:i + di])
                        col_start = sum(block_sizes[j:j + dj])
                        # Place the block at the correct position within the new larger block
                        new_block[row_start:row_start + block.shape[0], col_start:col_start + block.shape[1]] = block
            if new_block is not None:
                M_new[(i // step, j // step)] = new_block

    close_blocks = [[] for _ in range(len(new_block_sizes))]
    for line, col in M_new.keys():
        close_blocks[line].append(col)
    
    return M_new, new_block_sizes, close_blocks
