from __future__ import print_function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from VisPose import Pose, Sequence, PosePlt, init

fig = plt.figure(figsize=(7.2, 7.2), dpi=100)
ax = fig.add_subplot(111, projection='3d')
init(ax)
lim = 15
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])

####  Now plot your poses  #####
camplt = PosePlt(ax)
p1 = Pose([0, 0, 0], [0, 0, 5])
p2 = Sequence([p1, Pose([0, 0, 120], [0, 0, 5])])
print (p2)
print (p2.inv())
print (p2)
print (p2.inv())
camplt.nplot([p1, p2])
###########  End  ##############

ax.view_init(elev=-90, azim=-90)
plt.show()