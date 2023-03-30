import numpy as np

# Object for animated scatter plot of points:
class Scatter:
    def __init__(self, xss,yss,zss,ax, init_marker, init_linestyle, init_label):
        self.xss=xss
        self.yss=yss
        self.zss=zss

        self.ax=ax

        self.prev_marker=init_marker
        self.prev_linestyle=init_linestyle
        self.prev_label=init_label
     
        self.points, = ax.plot(xss[:,0],yss[:,0],zss[:,0], marker=init_marker, linestyle=init_linestyle, label=init_label)

    def update(self,t,marker=None,linestyle=None,label=None):

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

        self.points.set_data(self.xss[:,t],self.yss[:,t])
        self.points.set_3d_properties(self.zss[:,t], 'z')

        self.points.set_marker(marker)
        self.points.set_linestyle(linestyle)
        self.points.set_label(label)

# Object for directional point
class Arrow:
    def __init__(self, xs,ys,zs,yaws,pitches,rolls,init_length,init_color,init_label, ax, delta):
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

        self.quiver = ax.quiver(*self.get_arrow(xs[0], ys[0],zs[0],yaws[0],pitches[0],rolls[0]), length=init_length, color=init_color, label=init_label)

    def get_arrow(self,x,y,z,yaw,pitch,roll):
        return x,y,z,x+self.delta*np.cos(yaw)*np.cos(pitch),y+self.delta*np.sin(yaw)*np.cos(pitch),z+self.delta*np.sin(pitch)

    def update(self,t,length=None, color=None, label=None):
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
        self.quiver = self.ax.quiver(*self.get_arrow(self.xs[t], self.ys[t], self.zs[t], self.yaws[t],self.pitches[t],self.rolls[t]), length=length, color=color, label=label)
