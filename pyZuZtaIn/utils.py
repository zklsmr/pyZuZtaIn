def extract_biomarker_sequences(samples_sequence, biomarker_labels=None, subtype_order=None):
    """
    Extract biomarker progression sequences from samples.

    Parameters:
    -----------
    samples_sequence : ndarray
        3D array of shape (n_subtypes, n_samples, sequence_length)
        Contains the sampled sequences for each subtype
    biomarker_labels : list, optional
        Labels for each biomarker
    subtype_order : array-like, optional
        Order to process subtypes in

    Returns:
    --------
    dict : Dictionary containing for each subtype:
        - probabilities: 2D array showing probability of each biomarker at each position
        - most_likely_sequence: List of biomarker indices in their most likely order
        - sequence_labels: If labels provided, sequence with biomarker names
    """
    import numpy as np
    N_S = samples_sequence.shape[0]  # number of subtypes
    N = samples_sequence.shape[2]    # sequence length

    if subtype_order is None:
        subtype_order = np.arange(N_S)

    if biomarker_labels is None:
        biomarker_labels = [f"Biomarker {i}" for i in range(N)]

    sequences = {}

    for i in range(N_S):
        subtype_idx = subtype_order[i]

        # Get sequences for this subtype and transpose
        this_samples_sequence = samples_sequence[subtype_idx,:,:].T

        # Calculate probability of each biomarker at each position
        position_probabilities = (this_samples_sequence==np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[1]

        # Get most likely position for each biomarker
        most_likely_positions = np.argmax(position_probabilities, axis=1)

        # Sort biomarkers by their most likely position
        sequence_order = np.argsort(most_likely_positions)

        # Store results for this subtype
        sequences[f"Subtype_{i+1}"] = {
            "probabilities": position_probabilities,
            "most_likely_sequence": sequence_order,
            "sequence_labels": [biomarker_labels[idx] for idx in sequence_order],
            "position_certainties": np.max(position_probabilities, axis=1)[sequence_order]
        }

    return sequences







def plot_and_sequence(samples_sequence, samples_f, n_samples, score_vals, biomarker_labels=None, ml_f_EM=None, cval=False,
                       subtype_order=None, biomarker_order=None, title_font_size=12, stage_font_size=10,
                       stage_label='SuStaIn Stage', stage_rot=0, stage_interval=1, label_font_size=10,
                       label_rot=0, cmap="original", biomarker_colours=None, figsize=None,
                       subtype_titles=None, separate_subtypes=False, save_path=None, save_kwargs={}):
    # Get the number of subtypes
    N_S = samples_sequence.shape[0]
    # Get the number of features/biomarkers
    N_bio = score_vals.shape[0]

    # Initialize dictionary to store sequences
    sequence_data = {}

    # [Previous setup code remains the same until the plotting loop]

    if biomarker_labels is None:
        biomarker_labels = [f"Biomarker {i}" for i in range(N_bio)]
    else:
        biomarker_labels = [biomarker_labels[i] for i in biomarker_order]

    # [Color setup code remains the same]

    # Flag to plot subtypes separately
    if separate_subtypes:
        nrows, ncols = 1, 1
    else:
        if N_S == 1:
            nrows, ncols = 1, 1
        elif N_S < 3:
            nrows, ncols = 1, N_S
        elif N_S < 7:
            nrows, ncols = 2, int(np.ceil(N_S / 2))
        else:
            nrows, ncols = 3, int(np.ceil(N_S / 3))

    total_axes = nrows * ncols
    if separate_subtypes:
        subtype_loops = N_S
    else:
        subtype_loops = 1

    figs = []

    for i in range(subtype_loops):
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        figs.append(fig)

        for j in range(total_axes):
            if not separate_subtypes:
                i = j
            if isinstance(axs, np.ndarray):
                ax = axs.flat[i]
            else:
                ax = axs

            if i not in range(N_S):
                ax.set_axis_off()
                continue

            # Get sequences for this subtype
            this_samples_sequence = samples_sequence[subtype_order[i],:,:].T
            N = this_samples_sequence.shape[1]

            # Calculate position probabilities
            position_probabilities = np.zeros((N_bio, N))
            for pos in range(N):
                counts = np.sum(this_samples_sequence == pos, axis=1)
                position_probabilities[:, pos] = counts / this_samples_sequence.shape[0]

            # Get most likely positions and sequence order
            most_likely_positions = np.argmax(position_probabilities, axis=1)
            sequence_order = np.argsort(most_likely_positions)

            # Store sequence data
            sequence_data[f"Subtype_{i+1}"] = {
                "probabilities": position_probabilities[biomarker_order],
                "most_likely_sequence": sequence_order,
                "sequence_labels": [biomarker_labels[idx] for idx in sequence_order],
                "position_certainties": np.max(position_probabilities, axis=1)[sequence_order]
            }

            # Calculate confusion matrix for plotting
            confus_matrix = position_probabilities

            # [Rest of plotting code remains the same]

            # Create the colored confusion matrix
            confus_matrix_c = np.ones((N_bio, N, 3))
            for j, z in enumerate(num_scores):
                alter_level = colour_mat[j] == 0
                confus_matrix_score = confus_matrix[(stage_score==z)[0]]
                confus_matrix_c[
                    np.ix_(
                        stage_biomarker_index[(stage_score==z)[0]], range(N),
                        alter_level
                    )
                ] -= np.tile(
                    confus_matrix_score.reshape((stage_score==z).sum(), N, 1),
                    (1, 1, alter_level.sum())
                )

            # [Rest of plotting code remains the same]

        if save_path is not None:
            if separate_subtypes:
                save_name = f"{save_path}_subtype{i}"
            else:
                save_name = f"{save_path}_all-subtypes"
            if "format" in save_kwargs:
                file_format = save_kwargs.pop("format")
            else:
                file_format = "png"
            fig.savefig(f"{save_name}.{file_format}", **save_kwargs)

    return figs, axs, sequence_data
