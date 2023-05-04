# model-selection-via-cross-validation
The objective of this project was to perform regression learning on a U.S.A Working-Age Population Data set using polynomial curve fitting and root mean squared error (RMSE) as the performance metric. Additionally, cross-validation was employed for selecting the best model.

For more information about me, please visit my LinkedIn:

[![LinkedIn][LinkedIn.js]][LinkedIn-url]


<div align="center">
  <h3 align="center">A brief Read.me introducing the project and its contents</h3>
    <br />
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

  In thi project, machine learning is used to find the best polynomial model to fit a given dataset. Specifically, we will use 6-fold cross-validation on the training data to select the optimal polynomial degree value d from the set {0, 1, 2, 3, 4, 5, ..., 12}. We will also use 6-fold cross-validation on the training data to select the optimal regularization parameter λ from the set {0, exp(−25), exp(−20), exp(−14), exp(−7), exp(−3), 1, exp(3), exp(7)}. Once the optimal values of d and λ have been found, we will learn the regularized coefficient-weights of a 12-degree polynomial using all the training data. Finally, we will evaluate the resulting polynomial model on the training and test data, and report the training and test RMSE.


### PseudoCode used:

Pseudocode for calculating the optimal Parameter
1. Set the number of folds, k, to 6.
2. Initialize a KFold object with k number of splits and shuffle=False.
3. Define a range of degrees from 0 to 12., or store the lambda values in a list
4. Initialize an empty list called test_rmse.
5. Loop over the degrees/lamdas:
a. Initialize an empty list called test_rmse_fold.
b. Loop over the k folds in the KFold object.
i. Split the data into training and testing sets for the current fold.
ii. Define scaler functions for scaling the data for the current fold.
iii. Scale the training and testing data for the current fold.
iv. Import the Ridge regression class on the training data for the current fold.
v. Transform the training and testing data using PolynomialFeatures with the current
degree.
vi. Fit the model to the training data for the current fold
vii. Compute the testing RMSE for the current fold and append it to test_rmse_fold.
viii. Print the coefficients for the current degree.
c. Compute the testing RMSE for the current parameter and append it to test_rmse.
d.. Determine the optimal degree by finding the index of the minimum value in test_rmse.
e. Print the optimal degree.
6. Compute the average testing RMSE for all degrees.
7. Plot the average testing RMSE for each degree.
Pseudocode for Training the model on the optimal parameter found:
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
curve.
Results:
(1) the averages of the RMSE values obtained during the 6-fold CV for each case and
(2) the optimal degree d∗ and regularization parameter λ∗ obtained via the 6-fold CV;



### Scaling


The frameworks and libraries used within this project are:
* [![Scikit-learn][scikit-learn.js]][scikit-learn-url]
* [![TensorFlow][Tensorflow.js]][Tensorflow-url]
* [![NumPy][NumPy.js]][NumPy-url]
* [![Matplotlib][Matplotlib.js]][Matplotlib-url]
* [![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-yellowgreen?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/)


<!-- Dataset -->
### Dataset

Here's a summary of the Dataset used in the project:

1. Given a dataset of the indicator of working-age population in the U.S.A. through time
2. The working-age population is defined as those aged 15 to 64
3. The indicator measures the share of the working-age population in the total population for a number of years between 1970 and 2021, not necessarily consecutive
4. The only input attribute is the year
5. The output is the (numerical) indicator of the working-age population for the given input year
6. The files train.dat and test.dat contain the training and test datasets, respectively
7. Each dataset consists of two columns, the first corresponding to the calendar-year values (input) and the second column is the indicator of the working-age population (output)


<!-- GETTING STARTED -->
## Getting Started

To get the project running, there's a couple of programs and steps needed. here are the steps: 

### Prerequisites

If using Google colab to test the project, you need:

1. a Google colab account
2. Access to a GPU
3. Internet/Wi-Fi

If you plan on running it on python, you need to install on your computer the following:

1. Python
2. pip 
3. PyCharm 
4. Tensorflow, Keras, NumPy, Matplotlib


### Steps to run the code:

1. Download zip file
2. In the file, you’ll find 6 files, titled:

     a. CVfor0to12: performs the k fold cv using the given training data to select the optimal
degree

      b. Optimal Degree_trained: after finding d*, this trains all the data on the optimal degree found
      
      c. CVfor12Only: performs k fold cv to select the optimal regularizer value, λ*
      
      d. Optimal alpha_trained: performs ridge regression for λ* selected using all the data
      
      e. Test.csv: test data
      
      f. Train.csv: training data
      
3.Download the files. All program code is in python


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


