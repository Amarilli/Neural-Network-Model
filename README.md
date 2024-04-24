# Neural Network Model

## Background

The nonprofit foundation Alphabet Soup wants a tool to help it select the applicants for funding with the best chance of success in their ventures. 
Using the features in the provided dataset, I created a binary classifier with machine learning and neural networks to predict the success rate of Alphabet Soup's funding applicants.

From Alphabet Soup’s business team, I received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

## Processing the data

I read the `charity_data.csv` to a Pandas DataFrame

I dropped the EIN and NAME columns.

I determined the number of unique values for each column.

I determined the number of data points for each unique value in columns with more than 10 unique values.

I used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then checked if the binning was successful.

I used pd.get_dummies() to encode categorical variables.

I split the preprocessed data into a features array, X, and a target array, y. I used these arrays and the train_test_split function to split the data into training and testing datasets.

I scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, and then using the transform function.

## Compile, Train, and Evaluate the Model

Using TensorFlow, I design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. I reflected on how many inputs there were before determining the number of neurons and layers in the model. Once I completed that step, I compiled, trained, and evaluated the binary classification model to calculate the model’s loss and accuracy.

I created a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

I created the first hidden layer and chose an appropriate activation function.

I added a second hidden layer with an appropriate activation function.

I created an output layer with an appropriate activation function.

I checked the structure of the model.

I compiled and trained the model.

I created a callback that saves the model's weights every five epochs.

I evaluated the model using the test data to determine the loss and accuracy.

I saved and exported your results to an HDF5 file. Named the file AlphabetSoupCharity.h5.

## Optimize the Model
Using TensorFlow, I optimized the model to achieve a target predictive accuracy higher than 75%.

I adjusted the input data to ensure that no variables or outliers were confusing the model

I added more neurons to a hidden layer and added more hidden layers.
I used different activation functions for the hidden layers.
I reduced the number of epochs in the training regimen.

## Final Report on the Neural Network Model

The following is a comprehensive report and analysis of the Neural Network Model, along with answers to the questions posed in the assignment. The model's primary objective was to design an algorithm that could facilitate Alphabet Soup in predicting the success of funding applicants. The binary classifier model demonstrated a relatively high accuracy in determining whether the funding would succeed.

Concerning data preprocessing, the model's target variable was the IS_SUCCESSFUL column. In contrast, the following columns were the model's features: NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT. The EIN column, which merely served as an identifier for the applicant organization and had no impact on the model's behavior, was removed.

For model optimization, I introduced three hidden layers with multiple neurons, significantly improving accuracy from 72% to over 74%. It's worth noting that the Initial model had only two layers. Additionally, I reduced the number of epochs between the Initial and the Optimized Models, a change that, coupled with the third layer, played a crucial role in enhancing the model's accuracy. 

To further boost the model's performance, I implemented the following steps:

- Instead of dropping both the EIN and Name columns, I dropped only the EIN column. 
- I added a 3rd Activation Layer 
- I imported Keras Tuner, a library that helped me pick the optimal set of hyperparameters for my TensorFlow model. Hyperparameters are the configuration settings used to tune how the model learns and are set before the training process. Examples of hyperparameters include learning rate, the number of hidden layers in a neural network, the number of nodes in each layer, and activation functions. The proper set of hyperparameters can significantly impact the performance of a model, making the difference between a mediocre model and a highly accurate one.

Overall, the model's optimization resulted in an accuracy of above 74%. It implies that each point in the test data can be correctly classified 74% of the time. 

While the current model has demonstrated good performance and achieved a high level of accuracy, it's worth exploring an alternative approach - the Random Forest model. This model, also suitable for classification problems, has the potential to achieve higher than 80% accuracy rates, offering a competitive alternative to consider.





