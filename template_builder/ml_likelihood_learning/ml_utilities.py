import numpy as np


def shuffle_table(table, shuffle_column):
    rng = np.random.default_rng()

    indices = np.arange(len(table))
    rng.shuffle(indices)
    rand_table = table[indices]

    n_shuffle = int(len(rand_table) / 2)
    vals_to_be_shuffled = np.array(rand_table[shuffle_column][:n_shuffle])
    rng.shuffle(vals_to_be_shuffled)

    after_shuffle_column = "shuffled_" + shuffle_column

    after_shuffle_vals = np.concatenate(
        (vals_to_be_shuffled, np.array(rand_table[shuffle_column][n_shuffle:]))
    )

    rand_table[after_shuffle_column] = after_shuffle_vals
    y_column = np.ones(len(rand_table))
    y_column[:n_shuffle] = 0
    rand_table.add_column(y_column, name="shuffled")
    rng.shuffle(indices)
    rand_table = rand_table[indices]
    return rand_table
