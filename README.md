# model-selection-via-cross-validation
The objective of this project was to perform Linear Regression and Ridge Regression on a U.S.A Working-Age Population Data set using polynomial curve fitting and root mean squared error (RMSE) as the performance metric. Cross-validation was employed for selecting the best model.

For more information about me, please visit my LinkedIn:

[![LinkedIn][LinkedIn.js]][LinkedIn-url]

<div align="center">
  <h3 align="center">A brief Read.me introducing the project and its contents</h3>
    <br />
  </p>
</div>

<!-- ABOUT THE PROJECT -->

## Backgroud Terminology: 

To understand what Linear Regression and Ridge Regression (L2 Regularization) are, please visit the following links: 

Linear Regression: 
https://www.spiceworks.com/tech/artificial-intelligence/articles/what-is-linear-regression/#:~:text=Linear%20regression%20is%20an%20algorithm,machine%20learning%20for%20predictive%20analysis.

Ridge Regression:
https://vitalflux.com/ridge-regression-concepts-python-example/


## About The Project

  In this project, machine learning is used to find the best polynomial model to fit a given dataset. The project is divided into two parts:
  
<b> 1. Finding the optimal values d* and  λ*  for each part </b>:
  
<b> a. Part 1: Using Linear Regression to find the optimal Degree d </b>:

we will use 6-fold cross-validation on the training data to select the optimal polynomial degree value d from the set {0, 1, 2, 3, 4, 5, ..., 12}. In linear regression there is regularization parameter λ, so it is set equal to zero. 

<b> b. Part 2: Using Ridge Regression to find the optimal regularization parameter λ </b>:

we will use 6-fold cross-validation on the training data to select the optimal regularization parameter λ from the set {0, exp(−25), exp(−20), exp(−14), exp(−7), exp(−3), 1, exp(3), exp(7)} with the degree d set to 12.

<b> 2. Learning the polynomials' weights "coefficients" using all the training data:</b>

Once the optimal values d* and λ* for each part have been found, we will learn the regularized coefficient weights of the polynomials found in each part using all the training data.
 
 <b> 3. Testing the polynomials' accuracy using the testing data </b>
 
 Finally, we will evaluate the resulting polynomial model on the training and test data, and report the training and test RMSE.

   
### Dataset

1. The dataset represents an indicator that measures the proportion of the working-age population within the total population of the United States of America over time
note: "indicator" refers to a measure or metric used to track a specific aspect or phenomenon.
3. The working-age population is between ages 15 to 64
4. The indicator measures the working-age population proportion for non-consecutive years between 1970 and 2021.
5. The only input attribute is the year
6. The output is the (numerical) indicator of the working-age population for the given input year
7. The files train.dat and test.dat contain the training and test datasets, respectively
8. Each dataset consists of two columns, the first corresponding to the calendar-year values (input) and the second column is the indicator of the working-age population (output)

### Scaling

Data is normalized using the "z-score" normalization.  This will help the learning algorithm to output a hypothesis in a more numerically robust and accurate way. Z-score normalization involves applying a simple linear transformation to the input and output values separately, resulting in the average of the values being 0 and the standard deviation being 1. 

The formula to calculating the z-score of a point, x, is as follows:

![image](https://github.com/SaliElloh/model-selection-via-cross-validation/assets/112829375/a9cead32-c7a6-41e0-87a2-939383d73d90)

### PseudoCode used:

<b> Pseudocode for perform cross validation to find the optimal parameter d and lambda (used in file “CVfor0to12.py” and “CVfor12Only.py”)</b> 
<ol>
  <li>Set the number of folds, k, to 6.</li>
  <li>Initialize a KFold object with k number of splits and shuffle=False.</li>
  <li>Define a range of degrees from 0 to 12., or store the lambda values in a list</li>
  <li>Initialize an empty list called test_rmse.</li>
  <li>Loop over the degrees/lamdas:
    <ol type="a">
      <li>Initialize an empty list called test_rmse_fold.</li>
      <li>Loop over the k folds in the KFold object.
        <ol type="i">
          <li>Split the data into training and testing sets for the current fold.</li>
          <li>Define scaler functions for scaling the data for the current fold.</li>
          <li>Scale the training and testing data for the current fold.</li>
          <li>Import the Ridge regression class on the training data for the current fold.</li>
          <li>Transform the training and testing data using PolynomialFeatures with the current degree.</li>
          <li>Fit the model to the training data for the current fold</li>
          <li>Compute the testing RMSE for the current fold and append it to test_rmse_fold.</li>
          <li>Print the coefficients for the current degree.</li>
        </ol>
      </li>
      <li>Compute the testing RMSE for the current parameter and append it to test_rmse.</li>
      <li>Determine the optimal degree by finding the index of the minimum value in test_rmse.</li>
      <li>Print the optimal degree.</li>
    </ol>
  </li>
  <li>Compute the average testing RMSE for all degrees.</li>
  <li>Plot the average testing RMSE for each degree..</li>
</ol>

<b> Pseudocode for Training the model on the optimal parameter d* and lambda* (used in files “Optimal_Degree_trained.py” and “Optimal_alpha_trained.py”: </b>
1. Load the training and test data.
2. Separate the input and output variables from the training and test sets.
3. Define scaler functions for scaling the input variables.
4. Scale the training and test input variables using the scaler functions.
5. Define the degree of the polynomial regression model and create a PolynomialFeatures object,
include the optimal parameter found as part of the ridge regression model
6. Transform the scaled input variables using the PolynomialFeatures object.
7. Fit the Ridge regression model to the training data.
8. Make predictions on the training and test data using the fitted Ridge regression model.
9. Descale the predicted y train and test data (y_pred)
10. Evaluate the model's training RMSE and testing RMSE
11. print the coefficients for the optimal parameter model
12. Plot the training and test data, the fitted Ridge regression model, and the resulting polynomial
Curve.

## Results: 
the results of this project include: 

1. Averages of the RMSE values obtained during the 6-fold CV for each case plotted against the optimal degree d* and regularization parameter  λ∗ obtained via the 6-fold CV, separately:






2. the coefficient-weights of the d∗-degree polynomial and the λ∗-regularized 12-degree learned on all the training data

Coefficients of Optimal Degree (6):


Coefficients of Optimal Alpha (e^-3):




 4. the training and test RMSE of that final, learned polynomials, using the training and testing data:

Optimal Alpha e^-3:


Optimal Degree 6:


 5.  the 2 plots containing all the training data along with the resulting polynomial curves for d∗ and λ∗, for the range of years 1968-2023 as input:

Polynomial plots of the optimal degrees: 
Optimal degree (6):

 Optimal alpha e^-3:



<!-- Dataset -->


### Built With
The frameworks and libraries used within this project are:
* [![Scikit-learn][scikit-learn.js]][scikit-learn-url]
* [![TensorFlow][Tensorflow.js]][Tensorflow-url]
* [![NumPy][NumPy.js]][NumPy-url]
* [![Matplotlib][Matplotlib.js]][Matplotlib-url]
* [![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-yellowgreen?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/)


<!-- GETTING STARTED -->
## Getting Started

To get the project running, there's a couple of programs and steps needed. here are the steps: 

### Prerequisites

Python: PyCharm is an IDE for Python development. 
PyCharm:  you can download Pycahrm from JetBrains website.
Operating system: PyCharm is available for Windows, macOS, and Linux. 
Hardware requirements: PyCharm has minimum hardware requirements, including a multi-core processor, at least 4 GB of RAM, and a minimum screen resolution of 1024x768.
 Tensorflow, Scikit-learn, NumPy, Matplotlib installed on your computer

### Steps to run the code:

1. Download the “model-selection-via-cross-validation” zip file. 
2. Extract all files found in the zip file:
    In the file, you’ll find 6 files, titled:

     a. CVfor0to12:  performs the k fold cv using the given training data to select the optimal
degree.
      b. Optimal Degree_trained: after finding d*, perform linear regression for  all the data using the optimal degree found: finds the polynomial coefficients, and fit
      
      c. CVfor12Only: performs k fold cv to select the optimal regularizer value, λ*
      
      d. Optimal alpha_trained: performs ridge regression for λ* selected using all the data
      
      e. Test.csv: contains all the test data used in the project
      
      f. Train.csv: contains all the training data used in the project
      
3. Run each file separately to get the final results of the project

<!-- NEW LABELS IMAGE EXAMPLES -->
## Image examples


<!-- LICENSE -->
## License

No License used.

<!-- CONTACT -->
## Contact

Sali E-loh - [@Sali El-loh](https://www.linkedin.com/in/salielloh12/) - ellohsali@gmail.com


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments


* [TensoFlow: Clothes Image Classification Tutorial](https://www.tensorflow.org/tutorials/keras/classification)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[LinkedIn.js]: https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[LinkedIn-url]: https://www.linkedin.com/in/salielloh12/
[Tensorflow.js]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[Tensorflow-url]: https://www.tensorflow.org/
[NumPy.js]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Matplotlib.js]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/
[scikit-learn.js]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]:https://scikit-learn.org/










