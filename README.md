# Gradient-Descent-and-XGBoost

Exploratory work on the Gradient Descent algorithm and Extreme Gradient Boosting machine learning model.

The 'Basic Gradient Descent' file is based on the following cost function

<a href="http://www.codecogs.com/eqnedit.php?latex=2.9(x&space;-&space;7)^2&space;&plus;&space;1.5" target="_blank"><img src="http://latex.codecogs.com/gif.latex?2.9(x&space;-&space;7)^2&space;&plus;&space;1.5" title="2.9(x - 7)^2 + 1.5" /></a>

with the following derivative

<a href="http://www.codecogs.com/eqnedit.php?latex=5.8(x-7)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?5.8(x-7)" title="5.8(x-7)" /></a>



The Gradient Descent algorithm has the update formulation

<a href="http://www.codecogs.com/eqnedit.php?latex=x^{(1)}&space;=&space;x^{(0)}&space;-&space;\epsilon&space;*&space;\triangledown&space;f(x)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?x^{(1)}&space;=&space;x^{(0)}&space;-&space;\epsilon&space;*&space;\triangledown&space;f(x)" title="x^{(1)} = x^{(0)} - \epsilon * \triangledown f(x)" /></a>

where

<a href="http://www.codecogs.com/eqnedit.php?latex=\epsilon&space;=&space;\text{&space;Learning&space;Rate&space;}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\epsilon&space;=&space;\text{&space;Learning&space;Rate&space;}" title="\epsilon = \text{ Learning Rate }" /></a>

and

<a href="http://www.codecogs.com/eqnedit.php?latex=\triangledown&space;f(x)&space;=&space;\text{&space;Gradient&space;of&space;Cost&space;Function&space;}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\triangledown&space;f(x)&space;=&space;\text{&space;Gradient&space;of&space;Cost&space;Function&space;}" title="\triangledown f(x) = \text{ Gradient of Cost Function }" /></a>



In the function running Gradient Descent, there is an option that evaluates an adaptive learning rate. This particular adaptive learning rate is from Barzilai and Borwein. It takes the following form

<a href="http://www.codecogs.com/eqnedit.php?latex=\epsilon&space;=&space;\frac{\Delta&space;g(x)^{T}&space;\Delta&space;x}{\Delta&space;g(x)^{T}&space;\Delta&space;g(x)}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\epsilon&space;=&space;\frac{\Delta&space;g(x)^{T}&space;\Delta&space;x}{\Delta&space;g(x)^{T}&space;\Delta&space;g(x)}" title="\epsilon = \frac{\Delta g(x)^{T} \Delta x}{\Delta g(x)^{T} \Delta g(x)}" /></a>

where

<a href="http://www.codecogs.com/eqnedit.php?latex=\fn_cm&space;\Delta&space;g(x)&space;=&space;\triangledown&space;f(x^{(i+1)})&space;-&space;\triangledown&space;f(x^{(i)})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\fn_cm&space;\Delta&space;g(x)&space;=&space;\triangledown&space;f(x^{(i+1)})&space;-&space;\triangledown&space;f(x^{(i)})" title="\Delta g(x) = \triangledown f(x^{(i+1)}) - \triangledown f(x^{(i)})" /></a>

and

<a href="http://www.codecogs.com/eqnedit.php?latex=\Delta&space;x&space;=&space;x^{(i&plus;1)}&space;-&space;x^{(i)}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\Delta&space;x&space;=&space;x^{(i&plus;1)}&space;-&space;x^{(i)}" title="\Delta x = x^{(i+1)} - x^{(i)}" /></a>


