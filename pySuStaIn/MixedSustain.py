
from operator import index

from pySuStaIn.ZscoreSustain import ZscoreSustain
from pySuStaIn.AbstractSustain import AbstractSustain, AbstractSustainData
from scipy import stats
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm

import pdb

class MixedSustainData(AbstractSustainData):

    def __init__(self, zdata, prob_nl, prob_score, numStages):
        self.zdata = zdata
        self.prob_nl = prob_nl
        self.prob_score = prob_score
        self.__numStages = numStages
        self.__numSamples = self._validate_and_get_num_samples()

    def _validate_and_get_num_samples(self):
        num_samples = None
        if self.zdata is not None:
            num_samples = self.zdata.shape[0]

        if self.prob_nl is None and self.prob_score is not None:
            raise ValueError("prob_score provided without prob_nl.")
        if self.prob_nl is not None and self.prob_score is None:
            raise ValueError("prob_nl provided without prob_score.")

        if self.prob_nl is not None:
            if num_samples is None:
                num_samples = self.prob_nl.shape[0]
            else:
                assert num_samples == self.prob_nl.shape[0], "zdata and prob_nl must have same number of samples"
        if self.prob_score is not None:
            if num_samples is None:
                num_samples = self.prob_score.shape[0]
            else:
                assert num_samples == self.prob_score.shape[0], "zdata/prob_nl and prob_score must have same number of samples"

        if num_samples is None:
            raise ValueError("MixedSustainData requires zdata or prob_nl/prob_score.")
        return num_samples

    def getNumSamples(self):
        return self.__numSamples
    
    def getNumBiomarkers(self):
        num_biomarkers = 0
        if self.zdata is not None:
            num_biomarkers += self.zdata.shape[1]
        if self.prob_nl is not None:
            num_biomarkers += self.prob_nl.shape[1]
        return num_biomarkers
    
    def getNumStages(self):
        return self.__numStages
    
    def reindex(self, index):
        zdata = self.zdata[index,] if self.zdata is not None else None
        prob_nl = self.prob_nl[index,] if self.prob_nl is not None else None
        prob_score = self.prob_score[index,] if self.prob_score is not None else None
        return MixedSustainData(zdata, prob_nl, prob_score, self.__numStages)
        
class MixedSustain(AbstractSustain):

    def __init__(
            self, 
            zscore_data, z_vals, z_max, zscore_biomarker_labels, # zscore parameters
            prob_nl, prob_score, score_vals, ordinal_biomarker_labels, # ordinal parameters
            N_startpoints, N_S_max, N_iterations_MCMC,
            output_folder, dataset_name, use_parallel_startpoints, seed=None
            ):
                
        if zscore_data is not None and not np.all(zscore_data == 0):
            num_zscore_biomarkers = zscore_data.shape[1]
        else:
            num_zscore_biomarkers = 0
            z_vals = []
        if prob_nl is not None:
            num_ordinal_biomarkers = prob_nl.shape[1]
        else:
            num_ordinal_biomarkers = 0
            score_vals = []

        assert len(zscore_biomarker_labels) == num_zscore_biomarkers, "number of zscore biomarkers does not match with biomarker labels"
        assert len(ordinal_biomarker_labels) == num_ordinal_biomarkers, "number of ordinal biomarkers does not match with biomarker labels"

        if zscore_data is not None and prob_nl is not None:
            num_subjects_zdata = zscore_data.shape[0]
            num_subjects_ordinal = prob_nl.shape[0]
            assert num_subjects_zdata == num_subjects_ordinal, "number of subjects in zscore and ordinal data does not match"
            num_subjects = num_subjects_zdata
        elif zscore_data is not None:
            num_subjects = zscore_data.shape[0]
        elif prob_nl is not None:
            num_subjects = prob_nl.shape[0]

        num_biomarkers = num_zscore_biomarkers + num_ordinal_biomarkers
        max_score_value = int(max(
            score_vals.shape[1] if num_ordinal_biomarkers > 0 else 0, 
            z_vals.shape[1] if num_zscore_biomarkers > 0 else 0
        ))
        
        # create mixed_data_vals # maybe useful to create a function out of this = clearer? 
        bool_zscore_biomarkers = [True]*num_zscore_biomarkers + [False]*num_ordinal_biomarkers 
        bool_zscore_biomarkers = np.array(bool_zscore_biomarkers)

        zscore_indices = iter(z_vals)
        ordinal_indices = iter(score_vals)

        mixed_data_vals = np.zeros((num_biomarkers, max_score_value))
        for i in range(num_biomarkers):
            if bool_zscore_biomarkers[i]: # biomarker is zscore
                row = next(zscore_indices)
                mixed_data_vals[i, :len(row)] = row
            else: # biomarker is ordinal 
                row = next(ordinal_indices)
                mixed_data_vals[i, :len(row)] = row

        stage_score = mixed_data_vals.T.flatten()
        IX_select = stage_score > 0
        stage_score = stage_score[IX_select]
        stage_score = stage_score.reshape(1, len(stage_score))
        # extract which biomarkers have which stages
        stage_biomarker_index = np.tile(np.arange(num_biomarkers), (max_score_value,))
        stage_biomarker_index = stage_biomarker_index[IX_select]
        stage_biomarker_index = stage_biomarker_index.reshape(1, len(stage_biomarker_index))

        # initialise parameters
        # - zscore
        self.zscore_data = zscore_data
        self.z_vals = z_vals
        self.zscore_biomarker_labels = zscore_biomarker_labels
        self.min_biomarker_zscore = [0] * num_zscore_biomarkers
        self.max_biomarker_zscore = z_max
        self.std_biomarker_zscore = [1] * num_zscore_biomarkers # TODO: check if this one is used
        # - ordinal
        self.prob_nl = prob_nl
        self.prob_score = prob_score
        self.score_vals = score_vals
        self.ordinal_biomarker_labels = ordinal_biomarker_labels
        # - data combined
        self.mixed_data_vals = mixed_data_vals
        self.stage_score = stage_score
        self.stage_biomarker_index = stage_biomarker_index
        self.bool_zscore_biomarkers = bool_zscore_biomarkers
        self.num_biomarkers = num_biomarkers
        self.num_stages = stage_biomarker_index.shape[1]
        self.num_subjects = num_subjects
        self.biomarker_labels = zscore_biomarker_labels + ordinal_biomarker_labels
        # - model
        self.N_startpoints = N_startpoints
        self.N_S_max = N_S_max
        self.N_iterations_MCMC = N_iterations_MCMC
        # - general 
        self.output_folder = output_folder
        self.dataset_name = dataset_name
        self.use_parallel_startpoints = use_parallel_startpoints

        # initialise sustain data
        # TODO: prob_score is flattened in OrdinalSustain so should do that here as well I think
        self.__sustainData = MixedSustainData(zscore_data, prob_nl, prob_score, self.num_stages)

        # initialise abstract sustain
        super().__init__(self.__sustainData, N_startpoints, N_S_max, N_iterations_MCMC, output_folder, dataset_name, use_parallel_startpoints, seed)

        print("Init done for MixedSustain")

    def _initialise_sequence(self, sustainData, rng): # done :)
        """
        Randomly initialise a sequence ensuring that biomarkers are monotonically increasing 

        OUTPUTS:
        S - a random mixed input data model under the condition that each biomarker
            is monotonically increasing
        """
        N = self.num_stages
        S = np.zeros(N)

        for i in range(N):

            IS_min_stage_score = np.array([False] * N)
            possible_biomarkers = np.unique(self.stage_biomarker_index)
            for j in range(len(possible_biomarkers)):

                IS_unselected = [False] * N

                for k in set(range(N)) - set(S[:i]):
                    IS_unselected[k] = True
                    
                this_biomarkers = np.array(
                    [(np.array(self.stage_biomarker_index)[0] == possible_biomarkers[j]).astype(int) +
                     (np.array(IS_unselected) == 1).astype(int)]) == 2
                
                if not np.any(this_biomarkers):
                    this_min_stage_score = 0
                else:
                    this_min_stage_score = min(self.stage_score[this_biomarkers])
                if (this_min_stage_score):
                    temp = ((this_biomarkers.astype(int) + (self.stage_score == this_min_stage_score).astype(int)) == 2).T
                    temp = temp.reshape(len(temp), )
                    IS_min_stage_score[temp] = True

            events = np.array(range(N))
            possible_events = np.array(events[IS_min_stage_score])
            this_index = np.ceil(rng.random() * ((len(possible_events)))) - 1
            S[i] = possible_events[int(this_index)]

        S = S.reshape(1, len(S))

        return S

    def _calculate_likelihood_stage(self, sustainData, S):

        # TODO: maybe compute stage_value_zscore and stage_value_ordinal only for the relevant biomarkers?
        # then you do not need the boolean 
        # TODO: use sustainData instead of self.zscore etc. 

        if sustainData.prob_nl is not None:
            # prob_nl = self.prob_nl.copy()
            # prob_score = self.prob_score.copy()
            prob_nl = sustainData.prob_nl.copy()
            prob_score = sustainData.prob_score.copy()
        else: # in case there are no ordinal biomarkers in dataset
            prob_nl = []
            prob_score = []

        M = sustainData.getNumSamples() # fluctuates for multiple clusters
        
        # p perm k becomes a 3d array
        p_perm_k_biomarkers = np.zeros((M, self.num_stages + 1, self.num_biomarkers))

        # ---- z-score biomarkers calculation 
        # compute expected value with stage_value_zscore 
        stage_value_zscore = self.stage_value_zscore(S) # shape (num_biomarkers, num_stages + 1)
        stage_value_zscore = stage_value_zscore[self.bool_zscore_biomarkers]
        if self.zscore_data is not None:
            zscored_data = np.array(sustainData.zdata[:, :, None], dtype=np.float64)

            
            # zscored_data = np.array(self.zscore_data[:, :, None], dtype=np.float64)

            # x = (observed_value - expected_value)
            # shape x: (num_subjects, num_zscored_biomarkers, num_stages + 1)
            x = (zscored_data - stage_value_zscore) 
            x = np.transpose(x, (0, 2, 1)) # reshape to (num_subjects, num_stages + 1, num_zscored_biomarkers)

            # transform to likelihoods > with stats.norm.pdf() this is similar as to manually computing it (method in ZscoreSustain)
            x_prob = stats.norm.pdf(x) 

            p_perm_k_biomarkers[:, :, self.bool_zscore_biomarkers] = x_prob

        # ---- ordinal biomarkers calculation
        stage_value_ordinal = self.stage_value_ordinal(S) # shape = (num_stages + 1, num_ordinal_biomarkers)
        stage_value_ordinal = stage_value_ordinal[:, ~self.bool_zscore_biomarkers]

        # initialise first stage with normal probability
        p_perm_k_biomarkers[:, 0, ~self.bool_zscore_biomarkers] = prob_nl

        # prob_nl = shape = (num_subjects, num_ordinal_biomarkers)
        # prob_score = shape = (num_subjects, num_ordinal_biomarkers, num_scores)

        # probability_stage = np.zeros((prob_nl.shape))

        # loop over all stages and update p_perm_k with probabilities of biomarkers at that stages
        for stage in range(self.num_stages):
            # initialise with normal scores
            probability_stage = prob_nl.copy()    # shape = (num_subjects, num_ordinal_biomarkers)

            stage_value_justreached = stage_value_ordinal[stage + 1] # CHECK: should be plus one right? because we skip stage = 0?
             
            # change value of probability stage score if abnormal value in stage_value_justreached
            for i, value in enumerate(stage_value_justreached):
                if value > 0: # abnormal score reached
                    idx_value = value.astype(int) - 1    # using as an index: value 2 is position 1 in prob_score
                    idx_biomarker = i

                    probability_stage[:, idx_biomarker] = prob_score[:, idx_biomarker, idx_value]

            # update p_perm_k for this stage 
            p_perm_k_biomarkers[:, stage + 1, ~self.bool_zscore_biomarkers] = probability_stage

        # ---- calculate p_perm_k
        coeff = 1. / float(self.num_stages + 1)    # normalisation factor 
        p_perm_k = np.prod(p_perm_k_biomarkers, 2)    # multiply over all biomarkers
        p_perm_k = coeff * p_perm_k

        return p_perm_k

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):

        # TODO do not use self.N_S_max cus clusters change throughout 

        N_S = S_init.shape[0]
        M = sustainData.getNumSamples() # subjects fluctuates for multiple clusters
        
        S_opt = S_init.copy() # copy otherwise S_init will be overriden
        f_opt = np.array(f_init).reshape(N_S, 1, 1)
        f_val_mat = np.tile(f_opt, (1, self.num_stages + 1, M))
        f_val_mat = np.transpose(f_val_mat, (2, 1, 0))
        p_perm_k = np.zeros((M, self.num_stages + 1, N_S))

        for s in range(N_S):
            p_perm_k[:, :, s] = self._calculate_likelihood_stage(sustainData, S_opt[s])

        p_perm_k_weighted = p_perm_k * f_val_mat
        p_perm_k_norm = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)
        f_opt = (np.squeeze(np.sum(p_perm_k_norm, axis=(0, 1))) / np.sum(p_perm_k_norm)).reshape(N_S, 1, 1)

        f_val_mat = np.tile(f_opt, (1, self.num_stages + 1, M))
        f_val_mat = np.transpose(f_val_mat, (2, 1, 0))
        order_seq = rng.permutation(N_S)  # this will produce different random numbers to Matlab

        for s in order_seq:
            order_bio = rng.permutation(self.num_stages)  # this will produce different random numbers to Matlab
            for i in order_bio:
                current_sequence = S_opt[s]
                current_location = np.array([0] * len(current_sequence))
                current_location[current_sequence.astype(int)] = np.arange(len(current_sequence))

                selected_event = i

                move_event_from = current_location[selected_event]

                this_stage_zscore = self.stage_score[0, selected_event]
                selected_biomarker = self.stage_biomarker_index[0, selected_event]
                possible_zscores_biomarker = self.stage_score[self.stage_biomarker_index == selected_biomarker]

                # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                min_filter = possible_zscores_biomarker < this_stage_zscore
                max_filter = possible_zscores_biomarker > this_stage_zscore
                events = np.array(range(self.num_stages))
                if np.any(min_filter):
                    min_zscore_bound = max(possible_zscores_biomarker[min_filter])
                    min_zscore_bound_event = events[((self.stage_score[0] == min_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
                else:
                    move_event_to_lower_bound = 0
                if np.any(max_filter):
                    max_zscore_bound = min(possible_zscores_biomarker[max_filter])
                    max_zscore_bound_event = events[((self.stage_score[0] == max_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                    move_event_to_upper_bound = current_location[max_zscore_bound_event]
                else:
                    move_event_to_upper_bound = self.num_stages
                    # FIXME: hack because python won't produce an array in range (N,N), while matlab will produce an array (N)... urgh
                if move_event_to_lower_bound == move_event_to_upper_bound:
                    possible_positions = np.array([0])
                else:
                    possible_positions = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)
                possible_sequences = np.zeros((len(possible_positions), self.num_stages))
                possible_likelihood = np.zeros((len(possible_positions), 1))
                possible_p_perm_k = np.zeros((M, self.num_stages + 1, len(possible_positions)))
                for index in range(len(possible_positions)):
                    current_sequence = S_opt[s]

                    #choose a position in the sequence to move an event to
                    move_event_to = possible_positions[index]

                    # move this event in its new position
                    current_sequence = np.delete(current_sequence, move_event_from, 0)  # this is different to the Matlab version, which call current_sequence(move_event_from) = []
                    new_sequence = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, self.num_stages - 1)]])
                    possible_sequences[index, :] = new_sequence

                    possible_p_perm_k[:, :, index] = self._calculate_likelihood_stage(sustainData, new_sequence)

                    p_perm_k[:, :, s] = possible_p_perm_k[:, :, index]
                    total_prob_stage = np.sum(p_perm_k * f_val_mat, 2)
                    total_prob_subj = np.sum(total_prob_stage, 1)
                    possible_likelihood[index] = np.sum(np.log(total_prob_subj + 1e-250))

                possible_likelihood = possible_likelihood.reshape(possible_likelihood.shape[0])
                max_likelihood = max(possible_likelihood)
                this_S = possible_sequences[possible_likelihood == max_likelihood, :]
                this_S = this_S[0, :]
                S_opt[s] = this_S
                this_p_perm_k = possible_p_perm_k[:, :, possible_likelihood == max_likelihood]
                p_perm_k[:, :, s] = this_p_perm_k[:, :, 0]

            S_opt[s] = this_S

        p_perm_k_weighted = p_perm_k * f_val_mat
        p_perm_k_norm = p_perm_k_weighted / np.sum(p_perm_k_weighted + 1e-250, axis=(1, 2), keepdims=True)

        f_opt = (np.squeeze(np.sum(p_perm_k_norm, axis=(0, 1))) / np.sum(p_perm_k_norm)).reshape(N_S, 1, 1)
        f_val_mat = np.tile(f_opt, (1, self.num_stages + 1, M))
        f_val_mat = np.transpose(f_val_mat, (2, 1, 0))

        f_opt = f_opt.reshape(N_S)
        total_prob_stage = np.sum(p_perm_k * f_val_mat, 2)
        total_prob_subj = np.sum(total_prob_stage, 1)

        likelihood_opt = np.sum(np.log(total_prob_subj + 1e-250))

        return S_opt, f_opt, likelihood_opt

    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma):

        # TODO write the code a little neater

        N                                   = self.stage_score.shape[1]
        N_S                                 = seq_init.shape[0]

        if isinstance(f_sigma, float):  # FIXME: hack to enable multiplication
            f_sigma                         = np.array([f_sigma])

        samples_sequence                    = np.zeros((N_S, N, n_iterations))
        samples_f                           = np.zeros((N_S, n_iterations))
        samples_likelihood                  = np.zeros((n_iterations, 1))
        samples_sequence[:, :, 0]           = seq_init  # don't need to copy as we don't write to 0 index
        samples_f[:, 0]                     = f_init

        # Reduce frequency of tqdm update to 0.1% of total for larger iteration numbers
        tqdm_update_iters = int(n_iterations/1000) if n_iterations > 100000 else None 

        for i in tqdm(range(n_iterations), "MCMC Iteration", n_iterations, miniters=tqdm_update_iters):
            if i > 0:
                seq_order                   = self.global_rng.permutation(N_S)  # this function returns different random numbers to Matlab
                for s in seq_order:
                    move_event_from         = int(np.ceil(N * self.global_rng.random())) - 1
                    current_sequence        = samples_sequence[s, :, i - 1]

                    current_location        = np.array([0] * N)
                    current_location[current_sequence.astype(int)] = np.arange(N)

                    selected_event          = int(current_sequence[move_event_from])
                    this_stage_zscore       = self.stage_score[0, selected_event]
                    selected_biomarker      = self.stage_biomarker_index[0, selected_event]
                    possible_zscores_biomarker = self.stage_score[self.stage_biomarker_index == selected_biomarker]

                    # slightly different conditional check to matlab version to protect python from calling min,max on an empty array
                    min_filter              = possible_zscores_biomarker < this_stage_zscore
                    max_filter              = possible_zscores_biomarker > this_stage_zscore
                    events                  = np.array(range(N))
                    if np.any(min_filter):
                        min_zscore_bound            = max(possible_zscores_biomarker[min_filter])
                        min_zscore_bound_event      = events[((self.stage_score[0] == min_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                        move_event_to_lower_bound   = current_location[min_zscore_bound_event] + 1
                    else:
                        move_event_to_lower_bound   = 0

                    if np.any(max_filter):
                        max_zscore_bound            = min(possible_zscores_biomarker[max_filter])
                        max_zscore_bound_event      = events[((self.stage_score[0] == max_zscore_bound).astype(int) + (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
                        move_event_to_upper_bound   = current_location[max_zscore_bound_event]
                    else:
                        move_event_to_upper_bound   = N

                    # FIXME: hack because python won't produce an array in range (N,N), while matlab will produce an array (N)... urgh
                    if move_event_to_lower_bound == move_event_to_upper_bound:
                        possible_positions          = np.array([0])
                    else:
                        possible_positions          = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)

                    distance                = possible_positions - move_event_from

                    if isinstance(seq_sigma, int):  # FIXME: change to float
                        this_seq_sigma      = seq_sigma
                    else:
                        this_seq_sigma      = seq_sigma[s, selected_event]

                    # use own normal PDF because stats.norm is slow
                    weight                  = AbstractSustain.calc_coeff(this_seq_sigma) * AbstractSustain.calc_exp(distance, 0., this_seq_sigma)
                    weight                  /= np.sum(weight)
                    index                   = self.global_rng.choice(range(len(possible_positions)), 1, replace=True, p=weight)  # FIXME: difficult to check this because random.choice is different to Matlab randsample

                    move_event_to           = possible_positions[index]

                    current_sequence        = np.delete(current_sequence, move_event_from, 0)
                    new_sequence            = np.concatenate([current_sequence[np.arange(move_event_to)], [selected_event], current_sequence[np.arange(move_event_to, N - 1)]])
                    samples_sequence[s, :, i] = new_sequence

                new_f                       = samples_f[:, i - 1] + f_sigma * self.global_rng.standard_normal()
                new_f                       = (np.fabs(new_f) / np.sum(np.fabs(new_f)))
                samples_f[:, i]             = new_f

            S                               = samples_sequence[:, :, i]
            f                               = samples_f[:, i]
            likelihood_sample, _, _, _, _   = self._calculate_likelihood(sustainData, S, f)
            samples_likelihood[i]           = likelihood_sample

            if i > 0:
                ratio                           = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
                if ratio < self.global_rng.random():
                    samples_likelihood[i]       = samples_likelihood[i - 1]
                    samples_sequence[:, :, i]   = samples_sequence[:, :, i - 1]
                    samples_f[:, i]             = samples_f[:, i - 1]

        perm_index                          = np.where(samples_likelihood == max(samples_likelihood))
        perm_index                          = perm_index[0]
        ml_likelihood                       = max(samples_likelihood)
        ml_sequence                         = samples_sequence[:, :, perm_index]
        ml_f                                = samples_f[:, perm_index]

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood
    
    def _plot_sustain_model(self, *args, **kwargs):
        # TODO: decide whether to port ZscoreSustain.plot_positional_var or delegate to another model.
        raise NotImplementedError("TODO: implement MixedSustain plotting")

    @staticmethod
    def plot_positional_var():
        # TODO: implement plotting (or delegate) for MixedSustain.
        pass

    @staticmethod
    def generate_data():
        # TODO: implement data generation for MixedSustain.
        pass

    @staticmethod
    def generate_random_model():
        # TODO: implement random model generation for MixedSustain.
        pass
    
    @classmethod
    def test_sustain(cls):
        # TODO: implement test helper for MixedSustain.
        pass

    def subtype_and_stage_individuals_newData(self, zscore_data_new, prob_nl_new, prob_score_new, samples_sequence, samples_f, N_samples):

        numStages_new = self.__sustainData.getNumStages()
        sustainData_newData = MixedSustainData(zscore_data_new, prob_nl_new, prob_score_new, numStages_new)

        # TODO: implement subtype/stage inference for new data in MixedSustain.
        pass

    def compute_likelihood(self, sustainData, N_S, model_samples_sequence, model_samples_f):

        n_iterations = model_samples_sequence.shape[2]
        samples_likelihood = np.zeros((n_iterations, 1))

        for i in tqdm(range(n_iterations)):
            # check how N_S fits into this
            S = model_samples_sequence[:, :, i]
            f = model_samples_f[:, i]

            likelihood_sample, _, _, _, _   = self._calculate_likelihood(self.__sustainData, S, f)
            samples_likelihood[i]           = likelihood_sample

        return samples_likelihood

    # new functions
    def stage_value_zscore(self, S, return_point_value=False):
        
        # S = np.array(S).reshape(-1)
        S_inv = np.array([0]*self.num_stages)
        S_inv[S.astype(int)] = np.arange(self.num_stages)

        point_value = np.zeros((self.num_biomarkers, self.num_stages + 2))
        arange_N = np.arange(self.num_stages + 2)

        possible_biomarkers = np.unique(self.stage_biomarker_index)

        idx_zscore = 0
        for i in range(self.num_biomarkers):
            b = possible_biomarkers[i]
            if self.bool_zscore_biomarkers[b]:
                event_location = np.concatenate([[0], S_inv[(self.stage_biomarker_index == b)[0]], [self.num_stages]])
                event_value = np.concatenate([[self.min_biomarker_zscore[idx_zscore]], self.stage_score[self.stage_biomarker_index == b], [self.max_biomarker_zscore[idx_zscore]]])
                
                for j in range(len(event_location) - 1):
                    if j == 0:
                        temp = arange_N[event_location[j]:(event_location[j + 1] + 2)]
                        N_j = event_location[j + 1] - event_location[j] + 2
                        point_value[b, temp] = ZscoreSustain.linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])
                    else:
                        temp = arange_N[(event_location[j] + 1):(event_location[j + 1] + 2)]
                        N_j = event_location[j + 1] - event_location[j] + 1
                        point_value[b, temp] = ZscoreSustain.linspace_local2(event_value[j], event_value[j + 1], N_j, arange_N[0:N_j])

                idx_zscore += 1

        stage_value = 0.5 * point_value[:, :point_value.shape[1] - 1] + 0.5 * point_value[:, 1:]

        if return_point_value:
            return point_value

        return stage_value
    
    def stage_value_ordinal(self, S):

        stage_value_ordinal = np.zeros((self.num_stages + 1, self.num_biomarkers))

        for stage in range(self.num_stages):

            index_justreached = int(S[stage])
            biomarker_justreached = int(self.stage_biomarker_index[0][index_justreached])
            stage_value_justreached = int(self.stage_score[0][index_justreached])

            if not self.bool_zscore_biomarkers[biomarker_justreached]:
                index = stage + 1
                stage_value_ordinal[index:, biomarker_justreached] = stage_value_justreached

        return stage_value_ordinal
    
    def visualise_stage_values(self, S, save_name, gt_stages=None, ordinal_data=None):

        plt.figure(figsize=(6,4), constrained_layout=True)
        plt.rcParams['text.usetex'] = True # TeX rendering
        plt.rcParams['font.family'] = 'serif'  # Use LaTeX default serif font
        plt.rcParams['font.serif'] = ['Computer Modern Roman']  # Explicitly set font

        # TODO: if multiple clusters, loop over S

        # prepare inputs 
        stage_value_zscore = self.stage_value_zscore(S[0], return_point_value=True)
        stage_value_zscore = np.transpose(stage_value_zscore, (1, 0)) # reshape to shape (num_stages + 2, num_biomarkers)
        stage_value_zscore = stage_value_zscore[:, self.bool_zscore_biomarkers]
        stage_value_ordinal = self.stage_value_ordinal(S[0]) # shape = (num_stages + 1, num_biomarkers)
        stage_value_ordinal = stage_value_ordinal[:, ~self.bool_zscore_biomarkers]
        stage_value_ordinal = np.vstack((stage_value_ordinal, np.full((1, stage_value_ordinal.shape[1]), np.nan)))

        stage_value = np.hstack((stage_value_zscore, stage_value_ordinal))
        stage_value = pd.DataFrame(stage_value, index=np.arange(self.num_stages + 2), columns=np.arange(self.num_biomarkers))

        legend = []
        idx_zscore = 0
        idx_ordinal = 0
        for i in range(self.num_biomarkers):
            if self.bool_zscore_biomarkers[i]:
                name_biomarker = self.zscore_biomarker_labels[idx_zscore] # in case it is not simulated data
                z_value = self.z_vals[idx_zscore]
                legend.append(f'{i} zscore {z_value}')
                idx_zscore += 1
            else:
                name_biomarker = self.ordinal_biomarker_labels[idx_ordinal]
                ordinal_value = self.score_vals[idx_ordinal].astype(int)
                legend.append(f'{i} ordinal {ordinal_value}')
                idx_ordinal += 1

        x = stage_value.index
        arange_biomarkers = np.arange(self.num_biomarkers)
        continuous_cols = arange_biomarkers[self.bool_zscore_biomarkers]
        ordinal_cols = arange_biomarkers[~self.bool_zscore_biomarkers]

        # also plot the data values as scatter plot 
        # zscored data valeus
        if self.zscore_data is not None:
            z_scored = self.zscore_data.copy()
        else:
            z_scored = []

        colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        colours = ['#A52A2A', '#A52A2A', '#E69F00', '#228B22', '#228B22']

        if gt_stages is not None:
            # simulation data is plotted - this cannot be done for real data cuz then we do not know the GT stage

            max_samples_plot = 25 if self.num_subjects > 25 else self.num_subjects

            for i in range(max_samples_plot):
                gt_stage = gt_stages[i]

                idx_zscore = 0
                idx_ordinal = 0
                for j in range(self.num_biomarkers):
                    
                    if self.bool_zscore_biomarkers[j]:
                        color = colours[j]
                        # print(gt_stage, z_scored[idx_zscore, i])
                        plt.scatter(gt_stage, z_scored[i, idx_zscore], color=color, alpha=0.6, s=10)
                        idx_zscore += 1

                    else:
                        color = colours[j]
                        plt.scatter(gt_stage, ordinal_data[i, idx_ordinal], color=color, alpha=0.6, s=10)
                        idx_ordinal += 1

        # plot biomarker values in a different way for z-scored and ordinal biomarkers
        for idx_z in continuous_cols:
            plt.plot(x, stage_value[idx_z], color=colours[idx_z], linewidth=1.7)
        for idx_o in ordinal_cols:
            plt.step(x, stage_value[idx_o], where='post', color=colours[idx_o], linewidth=1.7)

        # plt.ylim(0, np.max(self.mixed_data_vals)+0.5)
        # # plt.ylim(0, np.max(self.max_biomarker_zscore))
        # plt.yticks(np.arange(np.max(self.mixed_data_vals) + 1))
        # plt.xlim(0, self.num_stages+ 0.1)
        # plt.xlim(0, self.num_stages + 0.5)
        # plt.xticks(np.arange(0, self.num_stages + 1, 1))
        # # plt.title("Sequence: " + str(S[0]))
        # plt.xlabel("Stages", fontsize=12)
        # plt.ylabel("Scores", fontsize=12)
        # # plt.legend(legend, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

        # for publication 
        
        plt.ylim(0, np.max(self.mixed_data_vals)+1)
        plt.yticks(np.arange(np.max(self.mixed_data_vals) + 2), fontsize=16)
        plt.xlim(0, self.num_stages+ 1)
        plt.xticks(np.arange(0, self.num_stages + 1, 1), fontsize=16)
        plt.xlabel(r'Stages', fontsize=16)
        plt.ylabel(r'Scores', fontsize=16)
        plt.grid(linestyle='dotted')

        plt.tight_layout()  # Adjust layout to prevent overlap

        # for stage in range(self.num_stages):
        #     stage = stage + 1
        #     plt.axvline(x=stage, ls='--', c='grey', alpha=0.4)

        save_to = os.path.join(self.output_folder, save_name)
        
        # plt.savefig(f'{save_to}.eps', format='eps', dpi=300)
        plt.savefig(f'{save_to}.png', dpi=300)
        plt.savefig(f'{save_to}.pdf', bbox_inches='tight')

        print("plot saved to " + f'{save_to}.eps')
