# Import datasets, classifiers and performance metrics
from sklearn import svm,datasets,metrics
from sklearn.model_selection import train_test_split

#read gigits
# def read_digits():
#     digits = datasets.load_digits()
#     x = digits.images
#     y = digits.target 
#     return x,y

#preprocess
def preprocess_data(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data

# Split data into 50% train and 50% test subsets
def split_data(x,y,test_size,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
     x,y, test_size=0.5, shuffle=False,random_state=random_state
    )
    return X_train,X_test,y_train,y_test
# train the model of choice with the model params
def train_model(x,y,model_params,model_type="svm"):
    # Create a classifier: a support vector classifier
    if model_type=="svm":
        clf = svm.SVC
    model=clf(**model_params)
    #pdb.set_trace()
    # train the model
    model.fit(x,y)
    return model

def split_train_dev_test(X, y, test_size, dev_size, random_state=1):
    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Calculate the remaining size for the development set
    remaining_size = 1.0 - test_size
    dev_relative_size = dev_size / remaining_size
    
    # Split the train data into train and development subsets
    X_train_final, X_dev, y_train_final, y_dev = train_test_split(
        X_train, y_train, test_size=dev_relative_size, random_state=random_state
    )
    
    return X_train_final, X_dev, X_test, y_train_final, y_dev, y_test

def predict_and_eval(model, y_predict, y_true):
    print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_predict, y_true)}\n"
    )


