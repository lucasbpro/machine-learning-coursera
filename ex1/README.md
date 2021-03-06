# Exercise 1 - Linear Regression

This exercise explores the power of linear regression for obtaining linear models to two-dimensional and multi-dimensional datasets. Models allow us to obtain estimations for an unknown variable given information known variables.

## Linear Regression fundamentals

Linear Regression is a mathematical process which aims to obtain a linear model of the type <img src="https://render.githubusercontent.com/render/math?math=h_{\theta}(x)=\theta^T%20x">, with <img src="https://render.githubusercontent.com/render/math?math=\theta"> and <img src="https://render.githubusercontent.com/render/math?math=x"> being n-dimensional vectors, which is the closest as possible to the actual data (x,y). In other words, we aim to have <img src="https://render.githubusercontent.com/render/math?math=h_{\theta}(x)"> the closest as possible to <img src="https://render.githubusercontent.com/render/math?math=y">.

In order to translate this definition to mathematical terms, we define a cost function <img src="https://render.githubusercontent.com/render/math?math=J({\theta})"> which measures the distance from the linear model <img src="https://render.githubusercontent.com/render/math?math=h_{\theta}(x)"> to the actual data points (x,y). Then, we run an optimization algorithm which finds the optimal parameter vector <img src="https://render.githubusercontent.com/render/math?math=\theta"> such that, for the specific data we have, the cost function is minimized.

In the following sections, there will be presented examples of 2D Linear Regression, where *x* and *y* are single variables, and Multi-dimensional Linear Regression, where *x* is a n-dimensional variable (that is, a set of variables).

## Two-dimensional Linear Regression

For the 2D Linear Regression, this exercise uses a dataset which relates the profit *x* of a food truck business over several diferents cities of population *y*. The plot below shows the data available in file *ex1data1.txt*. Obtaining a model for the data below allow us to obtain means for estimating the profit of the business in a city where the food truck is not present, given this city's population.

<img src="./img/data_visualization.png" width="500">

### Defining the cost function

As we aim to have <img src="https://render.githubusercontent.com/render/math?math=h_{\theta}(x)"> the closest as possible to <img src="https://render.githubusercontent.com/render/math?math=y">, the cost function can be defined as the sum of squared errors for all data points in the dataset. Assuming *m* is the number of datapoints in the dataset, we have:

<img src="https://render.githubusercontent.com/render/math?math=J(\theta)=\frac{1}{2m}\sum%20(y-h_{\theta}(x))^2">

If we plot <img src="https://render.githubusercontent.com/render/math?math=J(\theta)"> as a function of <img src="https://render.githubusercontent.com/render/math?math=\theta_{1}"> and <img src="https://render.githubusercontent.com/render/math?math=\theta_{2}"> , we obtain a graph like the one below:

<img src="./img/cost_function_3D.png" width="500">

In order to find the optimal <img src="https://render.githubusercontent.com/render/math?math=\theta"> vector, we shall use optimization!

### Obtaining a linear model for the data

There are tons of optimization algorithms, but Gradient Descent usually stands out because of its simplicity and low-computational cost. By running 500 iterations of the gradient descent, we find the optimal <img src="https://render.githubusercontent.com/render/math?math=\theta"> below (see the red cross on the left), which yelds the the model repsented by the blue curve (on the right).

<p float="left">
  <img src="./img/cost_function_countour_lines.png" width="450">
  <img src="./img/linear_model.png" width="450">
</p>



