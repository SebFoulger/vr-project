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
                force_left_intersection: bool = False,
                force_right_intersection: bool = False):
        """Apply segmentation procedure.

        Args:
            init_segment_size (int, optional): Initial size for left segments. Defaults to 10.
            window_size (int, optional): Size for window. Defaults to 10.
            step (int, optional): Step size for moving window. Defaults to 1.
            sig_level (float, optional): Significance level for statistical difference. Defaults to 0.05.
            beta_bool (bool, optional): Indicates whether statistical significance should be based only on betas, or
            also on intercepts. Defaults to True.
            force_left_intersection (bool, optional): Forces each left segment to connect to the last if True. beta_bool
            must be True if this is True. Defaults to False. 
            force_right_intersection (bool, optional): Forces each right window to connect to the left segment if True. 
            beta_bool must be True if this is True. Defaults to False. 

        Returns:
            pd.Series, list: Predictions of fitted model and list of breakpoints for segments
        """  
        # Will be the remaining data to segment throughout the process
        cur_x = self.x.copy()
        cur_y = self.y.copy()
        # STEP 1: find initial leftmost segment
        i, predictions = self._ind_segment(cur_x, cur_y, init_segment_size=init_segment_size, 
                                           window_size=window_size, step=step, sig_level = sig_level,beta_bool=beta_bool,
                                           right_intersection = force_right_intersection)
        breakpoints = [i]
        # STEP 2: find intermediate segments
        if predictions is not None:
            while len(cur_x)>i:
                # Represents the last x value in the current segment
                last_x = cur_x[list(cur_x.index)[0]+i-1]
                # Remove new segment data from current considered data
                cur_x = cur_x[i:]
                cur_y = cur_y[i:]
                if force_left_intersection:
                    # Intersection is the last point of the previous prediction
                    left_intersection = (last_x, predictions[len(predictions)-1])
                else:
                    left_intersection=None
                # Get new left segment and breakpoint
                i, left_predictions = self._ind_segment(cur_x, cur_y, 
                                                        init_segment_size=init_segment_size, window_size=window_size, 
                                                        step=step, sig_level = sig_level, beta_bool=beta_bool,
                                                        left_intersection = left_intersection, right_intersection=force_right_intersection)
                # Checking if the new segment is not empty
                if left_predictions is not None:
                    predictions = np.concatenate([predictions,left_predictions])
                    breakpoints.append(i)
                else:
                    # Finish if it is empty
                    break
        else:
            predictions = np.array([])
        # STEP 3: find final segment (if there is any data left)
        if len(cur_x)>0:
            final_model = sm.OLS(cur_y,sm.add_constant(cur_x))
            final_results = final_model.fit(use_t=True)
            final_predictions = final_results.predict()
            predictions = np.concatenate([predictions,final_predictions])
        # Turn distances between breakpoints into breakpoints by accumulating
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
                     left_intersection = None,
                     right_intersection: bool = False):
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
            left_intersection (None/tuple, optional): Set to None if no intersection forcing is desired. Set to (x,y) to
            force the left segment to go through (x,y). Defaults to None.
            right_intersection (bool, optional): Set to True to enforce right window to intersect left segment. Defaults
            to False. 

        Returns:
            int, pd.Series: breakpoint for beta_bool segment, and a series of predictions for this segment.
        """
        prev_pvalue = 1
        # Iterate moving window size
        for i in range(0,len(y),step):
            # If there isn't enough data left then finish
            if i+init_segment_size>=len(x)-1:
                return i, None
            # STEP 1: find left segment
            # If we don't want to force a left intersect
            if left_intersection is None:
                left_model = sm.OLS(y[:i+init_segment_size],sm.add_constant(x[:i+init_segment_size]))
            else:
                # Shift data to pass through left_intersection
                left_model = sm.OLS(y[:i+init_segment_size]-left_intersection[1],
                                    x[:i+init_segment_size]-left_intersection[0])
            # Fit left segment
            left_results = left_model.fit(use_t=True)
            left_predictions = left_results.predict()
            if left_intersection is not None:
                left_predictions += left_intersection[1]
            left_params = left_results.params
            
            next_prediction = x.iloc[i+init_segment_size]*left_params['time_exp']
            # STEP 2: find right window
            # If we are forcing the right window to intersect the left segment
            if right_intersection:
                # Shift data to pass through the end of the left segment
                right_model = sm.OLS(y[i+init_segment_size:i+init_segment_size+window_size]-next_prediction,
                                     x[i+init_segment_size:i+init_segment_size+window_size]-x.iloc[i+init_segment_size])
            else:
                right_model = sm.OLS(y[i+init_segment_size:i+init_segment_size+window_size],
                                sm.add_constant(x[i+init_segment_size:i+init_segment_size+window_size]))
            right_results = right_model.fit(use_t=True)
            # Breakpoint
            _break = i+init_segment_size
            if beta_bool:
                pvalue = right_results.t_test(f'time_exp = {left_params["time_exp"]}').pvalue
            else:
                pvalue = right_results.t_test(left_params).pvalue
            # STEP 3: check for statistical significance
            if pvalue <= sig_level and pvalue>prev_pvalue:
                #print(x.iloc[prev_return[0]],np.var(prev_return[1]-y[:prev_return[0]]))
                return prev_return
            # If we are enforcing a left intersection we need to shift the predictions
            prev_return = _break, left_predictions
            prev_pvalue = pvalue

