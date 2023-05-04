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


### What are Hijabs, Abayas, and Thobes? 


### Scaling


The frameworks and libraries used within this project are:
* [![Scikit-learn][scikit-learn.js]][scikit-learn-url]
* [![TensorFlow][Tensorflow.js]][Tensorflow-url]
* [![Keras][Keras.js]][Keras-url]
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

1.


2. 
3. 
4. 


   

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


