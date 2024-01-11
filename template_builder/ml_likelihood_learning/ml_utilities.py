import numpy as np


def shuffle_table(table, shuffle_columns):
    rng = np.random.default_rng()

    indices = np.arange(len(table))
    rng.shuffle(indices)
    rand_table = table[indices]

    n_shuffle = int(len(rand_table) / 2)
    vals_to_be_shuffled = np.column_stack(
        [rand_table[shuffle_column][:n_shuffle] for shuffle_column in shuffle_columns]
    )
    vals_not_to_be_shuffled = np.column_stack(
        [rand_table[shuffle_column][n_shuffle:] for shuffle_column in shuffle_columns]
    )
    rng.shuffle(vals_to_be_shuffled)

    after_shuffle_columns = [
        "shuffled_" + shuffle_column for shuffle_column in shuffle_columns
    ]

    after_shuffle_vals = np.concatenate((vals_to_be_shuffled, vals_not_to_be_shuffled))

    for i, col in enumerate(after_shuffle_columns):
        rand_table[col] = after_shuffle_vals[:, i]
    y_column = np.ones(len(rand_table))
    y_column[:n_shuffle] = 0
    rand_table.add_column(y_column, name="shuffled")
    rng.shuffle(indices)
    rand_table = rand_table[indices]
    return rand_table
