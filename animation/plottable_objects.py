import numpy as np
import pandas as pd

class Scatter:
    """
    Class to handle dynamic scatter plot (used for head-to-eye)
    """    
    def __init__(self, xss: np.array,yss: np.array,zss: np.array,
                 ax, init_marker: str, init_linestyle: str, init_label: str):
        """Initialiser function for class

        Args:
            xss (np.array): Numpy array where each entry is a pandas series containing the x-coordinates of one of the
            scatter points. Each series must be the same length.
            yss (np.array): As above but for y-coordinates. Must have the same length as above and the same length of
            each series.
            zss (np.array): As above but for z-coordinates.
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): Axes for plotting.
            init_marker (str): Marker for scatter points.
            init_linestyle (str): Linestyle between scatter points.
            init_label (str): Label for scatter.
        """        
        self.xss=xss
        self.yss=yss
        self.zss=zss
        self.ax=ax
        self.prev_marker=init_marker
        self.prev_linestyle=init_linestyle
        self.prev_label=init_label
        self.points, = ax.plot(xss[:,0],yss[:,0],zss[:,0], 
                               marker=init_marker, linestyle=init_linestyle, label=init_label, zorder=3)

    """Public functions"""

    def update(self,t: int, marker: str = None, linestyle: str = None, label: str = None):
        """Update function for scatter plot.

        Args:
            t (int): index of data to update with.
            marker (str, optional): marker to update scatter points with. Set to None if no update is desired. Defaults 
            to None.
            linestyle (str, optional): linestyle to update scatter with. Set to None if no update is desired. Defaults 
            to None.
            label (str, optional): label to update scatter with. Set to None if no update is desired. Defaults to None.
        """        
        if marker==None:
            marker=self.prev_marker
        else:
            self.prev_marker=marker

        if linestyle==None:
            linestyle=self.prev_linestyle
        else:
            self.prev_linestyle=linestyle

        if label==None:
            label=self.prev_label
        else:
            self.prev_label=label

        if self.xss[1, t] == 0 and self.yss[1, t] == 0 and self.zss[1, t] == 0:

            self.points.set_data_3d(self.xss[0,t],self.yss[0,t], self.zss[0,t])
        else:
            self.points.set_data_3d(self.xss[:,t],self.yss[:,t], self.zss[:,t])

        self.points.set_marker(marker)
        self.points.set_linestyle(linestyle)
        self.points.set_label(label)

class Arrow:
    """
    Class to handle dynamic arrow (used for controller).
    """    
    def __init__(self, xs: pd.Series,ys: pd.Series,zs: pd.Series,
                 yaws: pd.Series,pitches: pd.Series,rolls: pd.Series,
                 init_length: float,init_color: str,init_label: str, ax, delta: float):
        """Initialiser function for class.

        Args:
            xs (pd.Series): series of x-coordinates for base of arrow.
            ys (pd.Series): series of y-coordinates for base of arrow.
            zs (pd.Series): series of z-coordinates for base of arrow.
            yaws (pd.Series): series of yaws for arrow direction.
            pitches (pd.Series): series of pitches for arrow direction.
            rolls (pd.Series): series of rolls for arrow direction.
            init_length (float): initial length of arrow.
            init_color (str): initial colour of arrow.
            init_label (str): initial name for arrow.
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): axes for plotting.
            delta (float): size of arrow in calculation.
        """        
        self.xs=xs
        self.ys=ys
        self.zs=zs
        self.yaws=yaws
        self.pitches=pitches
        self.rolls=rolls
        self.ax=ax
        self.delta=delta

        self.prev_length = init_length
        self.prev_color = init_color
        self.prev_label = init_label

        self.quiver = ax.quiver(*self._get_arrow(xs[0], ys[0],zs[0],yaws[0],pitches[0],rolls[0]), length=init_length, 
                                color=init_color, label=init_label, zorder=2)

    """Public functions"""

    def update(self,t: int,length: float = None, color: str = None, label: str = None):
        """Update function for arrow.

        Args:
            t (int): index of data to update with.
            length (float, optional): length of arrow to update with. Set to None if no update is desired. Defaults to 
            None.
            color (str, optional): colour of arrow to update with. Set to None if no update is desired. Defaults to 
            None.
            label (str, optional): name of arrow to update with. Set to None if no update is desired. Defaults to None.
        """        
        if length==None:
            length=self.prev_length
        else:
            self.prev_length=length

        if color==None:
            color=self.prev_color
        else:
            self.prev_color=color

        if label==None:
            label=self.prev_label
        else:
            self.prev_label=label

        self.quiver.remove()
        self.quiver = self.ax.quiver(*self._get_arrow(self.xs[t], self.ys[t], self.zs[t], self.yaws[t],self.pitches[t],
                                                      self.rolls[t]), length=length, color=color, label=label, zorder=2)

    """Private functions"""

    def _get_arrow(self,x: float,y: float,z: float,yaw: float,pitch: float,roll: float):
        """_summary_

        Args:
            x (float): _description_
            y (float): _description_
            z (float): _description_
            yaw (float): _description_
            pitch (float): _description_
            roll (float): _description_

        Returns:
            _type_: _description_
        """
        arrow_return = (x,y,z,x+self.delta*np.cos(yaw)*np.cos(pitch),
                        y+self.delta*np.sin(yaw)*np.cos(pitch),z+self.delta*np.sin(pitch))
        return arrow_return
    