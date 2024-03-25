he only python libraries used in this assignment are :

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

to download:

pip install numpy
pip install matplotlib

To run each simulation via command line:

Case1.py:
python Case1.py

Case2.py:
python Case2.py

Case3.py:
python Case3.py

Case4.py:
python Case4.py


For Case1 and Case2:
hitting run will simulate 600 iterations, each plot point will be separated by 100
iterations each.
After 6 captures, the simulation will plot a trajectory graph, then a velocity magnitude
graph, then a connection graph. Closing each pop-up window will prompt the simulation to
continue.

Case3 and Case4 are the same as 1 and 2, but with 540 iterations and 
adding a center of mass plot at the end.


NOTE: "nodes" parameter is set to 100 at the top of each file, the time between each capture
with 100 nodes takes around 25 seconds on average. It is highly suggested to change "nodes" parameter to 
20 ~ 30 for testing before executing.