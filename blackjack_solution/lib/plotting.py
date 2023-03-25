 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_policy(policy, plot_filename="plot.png"):



    def get_Z(player_hand, dealer_showing, usable_ace):

        if (player_hand, dealer_showing, usable_ace) in policy:

            return policy[player_hand, dealer_showing, usable_ace]

        else:

            return 1



    def get_figure(usable_ace, ax):

        x_range = np.arange(1, 11)

        y_range = np.arange(12, 22)

        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([[get_Z(player_hand, dealer_showing, usable_ace) for dealer_showing in x_range] for player_hand in range(21, 11, -1)])

        surf = ax.imshow(Z, cmap=plt.get_cmap('Accent', 2), vmin=0, vmax=1, extent=[0.5, 10.5, 10.5, 21.5])

        plt.xticks(x_range, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))

        plt.yticks(y_range)

        ax.set_xlabel('Dealer Showing')

        ax.set_ylabel('Player Hand')

        ax.grid(color='black', linestyle='-', linewidth=1)

        divider = make_axes_locatable(ax)

        cax = divider.append_axes("right", size="5%", pad=0.1)

        cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)

        cbar.ax.set_yticklabels(['0 (HIT)','1 (STICK)'])

        #cbar.ax.invert_yaxis() 

            

    fig = plt.figure(figsize=(15, 15))

    ax = fig.add_subplot(121)

    ax.set_title('Usable Ace', fontsize=16)

    get_figure(True, ax)

    ax = fig.add_subplot(122)

    ax.set_title('No Usable Ace', fontsize=16)

    get_figure(False, ax)

    plt.savefig(plot_filename)

    plt.show()

    
    
def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = 21
    min_y = min(k[1] for k in V.keys())
    max_y = 10

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)
    
     
    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))
#     print( Z_noace)
    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=V[min(V, key=V.get)], vmax=V[max(V, key=V.get)])
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))