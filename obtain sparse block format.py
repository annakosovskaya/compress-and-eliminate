def make_dense_blocks(
    csr: ss.coo_matrix, B: int = 10
) -> Tuple[List[int], List[int], Dict[Tuple[int, int], np.array], List[Set], int]:
    """
    Obtain a sparse block format.

    Args:
        csr: Matrix in COO sparse format.
        B: Desired approximate block size (height or width).

    Returns:
        prm: Permutation (the same for rows and columns, so new matrix block_matrix = P csr P)
        block_sizes: Block (i, j) has the size block_sizes[i] x block_sizes[j]
        block_matrix: Matrix in sparse block format (dictionary with the key (i, j) existing if np.array block (i, j) is nonzero)
        nonzero_blocks: List of sets representing non-zero blocks indexes
                        in the block_matrix line (nonzero_blocks[line] contains cols: (line, col) is nonzero)
        nparts: The number of block rows
    """

    # Get the size of the csr matrix
    csr_size = csr.shape[0]
    nparts = max(2, csr_size // B)

    # Create a graph from the CSR matrix
    G = nx.from_scipy_sparse_array(csr)

    # Partition the graph using METIS
    _, l = metis.part_graph(G, nparts=nparts, recursive=True)

    # Initialize parts_cont and block_sizes
    parts_cont = [[] for _ in range(nparts)]
    block_sizes = [0 for _ in range(nparts)]
    for node, part in enumerate(l):
        parts_cont[part].append(node)
        block_sizes[part] += 1

    # Create the permutation list
    prm = [node for part in parts_cont for node in part]

    # Identify non-zero blocks
    nonzero_blocks = [set() for _ in range(nparts)]
    for node in range(csr_size):
        block = l[node]
        neighbors_blocks = {l[neighbor] for neighbor in G.adj[node]}
        nonzero_blocks[block].update(neighbors_blocks)

    # Initialize block_matrix and close_blocks
    block_matrix = {}
    for line in range(nparts):
        for col in nonzero_blocks[line]:
            block_matrix[line, col] = np.zeros((block_sizes[line], block_sizes[col]))

    # Extract row indices, column indices, and data from the CSR matrix
    row_idx, col_idx, data = csr.row, csr.col, csr.data

    # Create a mapping of node positions within their blocks
    row_pos = {node: i for part in parts_cont for i, node in enumerate(part)}

    # Populate block_matrix with actual values from the CSR matrix
    for r, c, v in zip(row_idx, col_idx, data):
        line, col = l[r], l[c]
        if (line, col) in block_matrix:
            block_matrix[(line, col)][row_pos[r], row_pos[c]] = v

    return prm, block_sizes, block_matrix, nonzero_blocks, nparts
