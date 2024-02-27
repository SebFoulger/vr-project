import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_lm

class LinearSegmentation:
    """
    Linear segmentation procedure
    """    
    def __init__(self):
        """Initialiser function for class.
        """     
        pass

    def segment(self,
                x: pd.Series, 
                y: pd.Series,
                window_size: int = 10, 
                sig_level: float = 0.01):
        """Apply segmentation procedure.

        Args:
            x (pd.Series): x values. Must be same length as y.
            y (pd.Series): y values. Must be same length as x.
            window_size (int, optional): Size for window. Defaults to 10.
            sig_level (float, optional): Significance level for statistical difference. Defaults to 0.01.
            x.name (str, optional): Name of the x data
        Returns:
            pd.Series, list: Predictions of fitted model and list of breakpoints for segments
        """
        assert(len(x) == len(y))
        # Initialise variables
        breakpoints = []
        predictions = []
        i = 0
        j = window_size
        q = 1
        prev_pred = 0
        # final_pred will be the next prediction of the left segment - the right window will be forced to go through
        # this point.
        while j + window_size <= len(x):
            # Special case for first linear regression
            if i == 0:
                # Regression with constant
                left_model = sm.OLS(y[:j], sm.add_constant(x[:j]))
                left_results = left_model.fit(use_t=True)
                final_pred = left_results.params[x.name]*x[j]+left_results.params['const']
            else:
                # Regression forced through previous section
                left_model = sm.OLS(y[i:j] - prev_pred, x[i:j]-x[i-1])
                left_results = left_model.fit(use_t=True)
                final_pred = left_results.params[x.name]*(x[j]-x[i-1]) + prev_pred
            # Right window regression forced through intersection
            right_model = sm.OLS(y[j:j+window_size] - final_pred, x[j:j+window_size]-x[j])
            right_results = right_model.fit(use_t=True)
            # Perform t-test
            pvalue = right_results.t_test(f'{x.name} = {left_results.params[x.name]}').pvalue
            # If the t-test is significant but less significant than the previous section then return that section
            if pvalue < sig_level and pvalue > q:
                # Add breakpoint and predictions
                breakpoints.append(j - 2)
                predictions = np.append(predictions, prev_predictions)
                # Update variables to indicate new segment being created
                prev_pred = prev_predictions[-1]
                i = j - 1
                j = j + window_size - 1
                q = 1
            else:
                # Move section
                j += 1
                q = pvalue
                # Keep track of predictions of previous section
                prev_predictions = left_results.predict() + prev_pred

        # Build model on final section of data
        if j<len(x):
            final_model = sm.OLS(y[i:] - prev_pred, x[i:]-x[i-1])
            final_results = final_model.fit(use_t=True)
            predictions = np.append(predictions, final_results.predict() + prev_pred)

        return {'predictions': predictions, 'breakpoints': breakpoints}