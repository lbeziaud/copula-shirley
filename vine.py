import numpy as np
import pandas as pd
import pyvinecopulib as pv

from .transform import TransformToNatualScale

def GetVineStructure(pseudo_observations, vine_family_set='all', vine_par_method="mle", vine_nonpar_method="constant", vine_selcrit="aic", vine_trunc_lvl=None, vine_tree_crit="tau", vine_cores=1):
    """Estimate the vine-copula structure from the pseudo-observations. Uses the R library rvinecopulib.
    
    See https://vinecopulib.github.io/rvinecopulib/reference/vinecop.html for more informations on the vine-copula parameters.
    Note: For our implementation, trunc_lvl = 0 means no truncation.

    Args:
        pseudo_observations (n-D array): An array containing pseudo-observations to estimate the model from.
        vine_family_set (str, optional): The copulas families to fit to. Defaults to 'all'.
        vine_par_method (str, optional): The estimation method for parametric models. Defaults to 'mle'.
        vine_nonpar_method (str, optional): The estimation method for nonparametric models. Defaults to 'constant'.
        vine_selcrit (str, optional): The criterion for family selection. Defaults to 'aic'.
        vine_trunc_lvl (int, optional): The truncation level of the vine copula. Defaults to 'NA'.
        vine_tree_crit (str, optional): The criterion for tree selection. Defaults to 'tau'.
        vine_cores (int, optional): The number of cores to use. Defaults to 1.

    Returns:
        Vinecop: An R vine-copula structure estimated from the pseudo-observations.
    """
    if vine_trunc_lvl is None:
        kwargs_trunc = dict(select_trunc_lvl=True)
    elif vine_trunc_lvl == 0:
        kwargs_trunc = dict(select_trunc_lvl=False)
    else:
        kwargs_trunc = dict(select_trunc_lvl=False, trunc_level=vine_trunc_lvl)

    if isinstance(vine_family_set, str):
        vine_family_set = getattr(pv, vine_family_set)

    controls = pv.FitControlsVinecop(family_set=vine_family_set,
                                     parametric_method=vine_par_method,
                                     nonparametric_method=vine_nonpar_method,
                                     **kwargs_trunc,
                                     tree_criterion=vine_tree_crit,
                                     selection_criterion=vine_selcrit,
                                     num_threads=vine_cores
                                     )
    cop = pv.Vinecop(pseudo_observations, controls=controls)
    return cop
    
def GetSamplesFromVine(vine_struct, n_sample, col_names, dp_ecdfs, vine_cores=1):
    """Generate samples from the vine-copula model.

    Args:
        vine_struct (Vinecop): An R vine-copula structure.
        n_sample (int): The numer of sample to generate.
        col_names (list): The column names of the original data.
        dp_ecdfs (dict): A dictionary of ECDFs.
        vine_cores (int, optional): The number of cores to use. Defaults to 1.

    Returns:
        DataFrame: A DataFrame with newly generated (pseudo-observations) samples from the vine-copula model.
    """
    samp = pd.DataFrame(np.asarray(vine_struct.simulate(n_sample, num_threads=vine_cores)), columns=col_names)
    return TransformToNatualScale(samp, dp_ecdfs)

def GetSamplesFromVineOHE(vine_struct, n_sample, col_names, decoder, dp_ecdfs, constant_cols, constant_vals, vine_cores=1):
    """Generate samples from the vine-copula model for Dummy Encoded Variables.
    Given the highly sparse domain, vine copulas tend to create empty one-hot encoded variables (all equals to 0 given a categorical attributes) 
    which results in NaN values when decoded.

    Args:
        vine_struct (object): An R vine-copula structure.
        n_sample (int): The numer of sample to generate.
        col_names (list): The column names of the original data.
        decoder (object): The decoder for the One-Hot Encoder.
        dp_ecdfs (dict): A dictionary of ECDFs.
        vine_cores (int, optional): The number of cores to use. Defaults to 1.

    Returns:
        DataFrame: A DataFrame with newly generated (pseudo-observations) samples from the vine-copula model.
    """
    vine_samples = pd.DataFrame(columns=col_names)

    while vine_samples.shape[0] < n_sample:
        samp = GetSamplesFromVine(vine_struct, vine_cores*1000, col_names=col_names, dp_ecdfs=dp_ecdfs, vine_cores=vine_cores)
        for i in range(len(constant_cols)): samp[constant_cols[i]] = constant_vals[i]
        dec_samp = decoder(samp).dropna()
        if dec_samp.shape[0]:
            samp = samp.drop(constant_cols, axis=1).iloc[dec_samp.index]
            vine_samples = vine_samples.append(samp)

    return vine_samples.sample(n_sample).reset_index(drop=True)