# GradeRegular
## Summary
Converts images to JFLAP files (.jff) or Verilog. Use JFLAP for equivalence checking between multiple diagrams.
*NOTE: detction accuracy is not super great. Any help will be appriciated.

## Setup Procedures
1. *Install MiniConda3*: https://conda.io/miniconda.html
2. *Install Python Libraries* (in Anaconda Prompt):
  * **conda install pip**
  * **conda install scipy scikit-image ipython**
  * **pip install numpy matplotlib opencv-python nnabla tarjan**
3. *Clone Git Repo*: **git clone [this repo url]**
4. Using the programs:
  * Make Verilog from image: **python Image2V.py [input image file]** # this has visual output
  * Make JFLAP File (.jff) from image: **python Image2JFLAP.py [input image file]** # this has no visual output but outputs .jff file to ./Workspace/imgResult.jff
  
**Sample Visual Output (Examples/epsilonEnds1.jpg)**
![Run Sample](https://github.com/YoshikiTakashima/GradeRegular/blob/master/Wiki/goodRun.jpeg "Good Run Sample")

## Drawing Rules
**RULES**: leftmost node is the start state, accept states are double circles, transitions are 0, 1, or Epsilon 
1. State circles must be 10 to 40% of the smaller of the two sides of image (if landscape, then it is the height)
2. All transitions represent one and only one value out of 0, 1, or epsilon
3. For self-loops, the transition label should be written within the loop. For non-self-loops, the transition label should be written close to the destination.
4. Transition line should not cross with another transition line
5. No labels or letters within states.
6. When having 2 lines in parallel, make sure they are not curved and reduce the distance between the two to prevent it from being detected as circle.
7. For accept states (double circle), make sure the inner circle is clearly separated with good distance (10-20% of circle radius).

## Drawing tips (not absolute, but helps detect)
1. Diagram should be written neatly and with thick strokes.
2. Keep the number of nodes low (2-4) so that these do not form near-closed regions.
3. Keep transition labels from touching transition lines.

## General Program Structure and Specific Algorithms/Tools Used
![Flowchart](https://github.com/YoshikiTakashima/GradeRegular/blob/master/Wiki/Flowchart.JPG "Logo Title Text 1")

1. *Detect Circles*: For each point at equal intervals in X,Y, go in 4 directions until you hit a black point. If these are symmetric and fit profiles of size and skew, then it is a circle.
2. *Detect Lines*: Around each node, For each black point, trace such unit you hit a node. Form traces and transitions from such.
3. *Text Region Identification*: [OpenCV Implementation of MSER](https://docs.opencv.org/3.4/d3/d28/classcv_1_1MSER.html)
4. *Machine Learning Text Classifcation*: Sony [NNABLA](https://github.com/sony/nnabla) and [Neural Network Console](https://dl.sony.com/)
5. *Join Lines*: If 2 lines converge, join them.
6. *Transition and State*: For each self loop, find label closed to it. For each remaining label, find transition endpoint closest to it.
7. *Synthesize Verilog, Epsilon-Reachable State Detection*: [Tarjan's Algorithm, Transitive closure](https://pypi.org/project/tarjan/) 
