def generate_symmetric_pairs(n: int, nonzero_blocks: int) -> List[Tuple[int, int]]:
    """
    Generate symmetric pairs for the given matrix size and number of nonzero blocks.

    Args:
        n (int): The size of the matrix.
        nonzero_blocks (int): The number of nonzero blocks.

    Returns:
        list: A list of symmetric pairs (tuples).
    """
    # Generate all diagonal pairs
    diagonal_pairs = [(i, i) for i in range(n)]

    # Initialize a set for symmetric pairs
    symmetric_pairs = set(diagonal_pairs)

    # Generate additional pairs until we have nonzero_blocks pairs in total
    while len(symmetric_pairs) < nonzero_blocks:
        a = np.random.choice(n, 2, replace=False)
        pair1 = (a[0], a[1])
        pair2 = (a[1], a[0])

        # Add the pairs symmetrically if they are not already included
        if pair1 not in symmetric_pairs and pair2 not in symmetric_pairs:
            symmetric_pairs.add(pair1)
            symmetric_pairs.add(pair2)

    return list(symmetric_pairs)

def generate_matrix(
    matrix_size: int = 100, 
    sparsity: float = 0.2
) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[int], List[Set[int]], int, List[Tuple[int, int]]]:
    """
    Generate a sparse matrix in block format.

    Args:
        matrix_size (int): The size of the matrix.
        sparsity (float): The sparsity of the matrix.

    Returns:
        tuple: Contains the following elements:
            - M (dict): The matrix in sparse block format.
            - block_sizes (list): The sizes of the blocks.
            - close_blocks (list): The close blocks.
            - block_num (int): The number of blocks.
            - pairs (list): The list of block pairs.
    """
    nonzero_elements = max(1, int(matrix_size * sparsity))
    B = int(nonzero_elements ** 0.5)
    block_num = math.ceil(matrix_size / B)
    nonzero_blocks = sparsity * block_num ** 2
    pairs = generate_symmetric_pairs(block_num, nonzero_blocks)

    block_sizes = np.random.randint(
        max(1, int(B * 0.9)), int(B * 1.1) + 1, block_num
    )

    close_blocks = [set() for _ in range(block_num)]
    M = {}
    for (line, col) in pairs:
        close_blocks[line] |= {col}
        M[line, col] = np.random.rand(block_sizes[line], block_sizes[col]) * 100

    return M, block_sizes, close_blocks, block_num, pairs
