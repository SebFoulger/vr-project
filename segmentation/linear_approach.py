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
                sig_level: float = 0.05,
                beta_bool: bool = True,
                force_intersection: bool = False):
        """Apply segmentation procedure.

        Args:
            init_segment_size (int, optional): Initial size for left segments. Defaults to 10.
            window_size (int, optional): Size for window. Defaults to 10.
            step (int, optional): Step size for moving window. Defaults to 1.
            sig_level (float, optional): Significance level for statistical difference. Defaults to 0.05.
            beta_bool (bool, optional): Indicates whether statistical significance should be based only on betas, or
            also on intercepts. Defaults to True.
            force_intersection (bool, optional): Forces each segment to connect to the last if True. Note this only 
            affects the left segment and the right window regression still doesn't force intersection when deciding a 
            new section. Defaults to False.

        Returns:
            pd.Series, list: Predictions of fitted model and list of breakpoints for segments
        """        
        cur_x = self.x.copy()
        cur_y = self.y.copy()
        i, predictions = self._ind_segment(cur_x, cur_y, init_segment_size=init_segment_size, 
                                                           window_size=window_size, step=step, sig_level = sig_level,beta_bool=beta_bool)
        breakpoints = [i]
        while len(cur_x)>i:
            last_x = cur_x[list(cur_x.index)[0]+i-1]
            cur_x = cur_x[i:]
            cur_y = cur_y[i:]
            if force_intersection:
                intersection = (last_x,predictions[len(predictions)-1])
                
            else:
                intersection=None
            i, left_predictions = self._ind_segment(cur_x, cur_y, 
                                                    init_segment_size=init_segment_size, window_size=window_size, 
                                                    step=step, sig_level = sig_level, beta_bool=beta_bool,
                                                    intersection = intersection)

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
                     sig_level: float = 0.05,
                     beta_bool: bool = True,
                     intersection = None):
        """Finds one individual segment.

        Args:
            x (pd.Series): x values.
            y (pd.Series): y values.
            init_segment_size (int, optional): Initial size for left segments. Defaults to 10.
            window_size (int, optional): Size for window. Defaults to 10.
            step (int, optional): Step size for moving window. Defaults to 1.
            sig_level (float, optional): Significance level for statistical difference. Defaults to 0.05.
            beta_bool (bool, optional): Set to true to have statistical significance based on gradients only. Defaults 
            to True.
            intersection (None/tuple, optional): Set to None if no intersection forcing is desired. Set to (x,y) to
            force the left segment to go through (x,y). Defaults to None.

        Returns:
            int, pd.Series: breakpoint for beta_bool segment, and a series of predictions for this segment.
        """
        if intersection is not None:
            x = x-intersection[0]
            y = y-intersection[1]
        for i in range(len(y)):
            if i*step+init_segment_size>=len(x)-1:
                return i, None
            if intersection is None:
                left_model = sm.OLS(y[:i*step+init_segment_size],sm.add_constant(x[:i*step+init_segment_size]))
            else:
                left_model = sm.OLS(y[:i*step+init_segment_size],x[:i*step+init_segment_size])
            left_results = left_model.fit()
            left_predictions = left_results.predict()
            left_params = left_results.params
            right_model = sm.OLS(y[i*step+init_segment_size:i*step+init_segment_size+window_size],
                                sm.add_constant(x[i*step+init_segment_size:i*step+init_segment_size+window_size]))
            right_results = right_model.fit()
            _break = i*step+init_segment_size
            if beta_bool and right_results.t_test(f'time_exp = {left_params["time_exp"]}').pvalue <= sig_level:
                
                break
            elif not beta_bool and right_results.t_test(left_params).pvalue <= sig_level:
                break
        if intersection is not None:
            return _break, left_predictions+intersection[1]
        else:
            return _break, left_predictions
