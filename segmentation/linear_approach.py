import pandas as pd
import numpy as np
import statsmodels.api as sm

class LinearSegmentation:
    """
    Class for applying linear segmentation to data.
    """    
    def __init__(self,
                 x: pd.Series, 
                 y: pd.Series):
        """Initialiser function for class.

        Args:
            x (pd.Series): x values.
            y (pd.Series): y values.
        """        
        self.x = x[:-1]
        self.y = y[:-1]

    def segment(self,
                init_segment_size: int = 10,
                window_size: int = 10, 
                step: int = 1, 
                sig_level: float = 0.05):
        """Apply segmentation procedure.

        Args:
            init_segment_size (int, optional): Initial size for left segments. Defaults to 10.
            window_size (int, optional): Size for window. Defaults to 10.
            step (int, optional): Step size for moving window. Defaults to 1.
            sig_level (float, optional): Significance level for statistical difference. Defaults to 0.05.

        Returns:
            pd.Series, list: Predictions of fitted model and list of breakpoints for segments
        """        
        cur_x = self.x.copy()
        cur_y = self.y.copy()
        i, predictions = self._ind_segment(cur_x, cur_y, init_segment_size=init_segment_size, 
                                                           window_size=window_size, step=step, sig_level = sig_level)
        breakpoints = [i]
        while len(cur_x)>i:
            cur_x = cur_x[i:]
            cur_y = cur_y[i:]

            i, left_predictions = self._ind_segment(cur_x, cur_y, 
                                                    init_segment_size=init_segment_size, window_size=window_size, 
                                                    step=step, sig_level = sig_level)
            if left_predictions is not None:
                predictions = np.concatenate([predictions,left_predictions])
                breakpoints.append(i)
            else:
                break
        
        if len(cur_x)>0:
            final_model = sm.OLS(cur_y,sm.add_constant(cur_x))
            final_results = final_model.fit()
            final_predictions = final_results.predict()
            predictions = np.concatenate([predictions,final_predictions])

        for i in range(1,len(breakpoints)):
            breakpoints[i]+=breakpoints[i-1]

        return predictions, breakpoints

    def _ind_segment(self, 
                     x: pd.Series,
                     y: pd.Series, 
                     init_segment_size: int = 10, 
                     window_size: int = 10, 
                     step: int = 1, 
                     sig_level: float = 0.05):
        """Finds one individual segment.

        Args:
            x (pd.Series): x values.
            y (pd.Series): y values.
            init_segment_size (int, optional): Initial size for left segments. Defaults to 10.
            window_size (int, optional): Size for window. Defaults to 10.
            step (int, optional): Step size for moving window. Defaults to 1.
            sig_level (float, optional): Significance level for statistical difference. Defaults to 0.05.

        Returns:
            int, pd.Series: breakpoint for new segment, and a series of predictions for this segment.
        """        
        for i in range(len(y)):
            if i*step+init_segment_size>=len(x)-1:
                return i, None
            left_model = sm.OLS(y[:i*step+init_segment_size],sm.add_constant(x[:i*step+init_segment_size]))
            left_results = left_model.fit()
            left_params = left_results.params

            right_model = sm.OLS(y[i*step+init_segment_size:i*step+init_segment_size+window_size],
                                sm.add_constant(x[i*step+init_segment_size:i*step+init_segment_size+window_size]))
            right_results = right_model.fit()

            if right_results.t_test(left_params).pvalue <= sig_level:
                break

        return i*step+init_segment_size, left_results.predict()
