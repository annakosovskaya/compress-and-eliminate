# check obtaining sparse block format

prm, block_sizes, M, close_blocks, M_size = make_dense_blocks(B)
prm_inv = inverse_permutation(prm)

B_dense = B.todense()
M_dense = blocks_to_full_matrix(M, block_sizes)

M_to_B = apply_permutation(M_dense, prm_inv)
print(np.sum(M_to_B - B_dense))
