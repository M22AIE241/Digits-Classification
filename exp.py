"""
Author: SG ,m22aie241 

"""

#utils import
from utils import load_dataset, data_preprocessing, split_train_dev_test,predict_and_eval
from utils import get_list_of_param_comination, tune_hparams
import pandas as pd 
import sys

total_run = int(sys.argv[1])
dev_size = [float(sys.argv[2])]
test_size = [float(sys.argv[3])]
model_type = sys.argv[4]


###########################################################################################
#1.get/load the dataset
X,y = load_dataset()

print("total no of images in datasat",y.shape[0])
print("size of each image in datasat ", X[0].shape)

#2.Sanity check of data

################################################################################################
#taking different combinations of train dev and test and reporting results

# test_size =  [0.2]
# dev_size = [0.2]
# total_run = 5
results = []
for run_num in range(total_run):
    for ts in test_size:
        for ds in dev_size:
            #3. Spliting the data
            X_train, y_train, X_test, y_test, X_dev, y_dev = split_train_dev_test(X, y, test_size=ts, dev_size=ds)  

            #################################################################################################
            #4. Preprocessing the data
            X_train = data_preprocessing(X_train)
            X_test= data_preprocessing(X_test)
            X_dev =data_preprocessing(X_dev)

            #################################################################################################
            #5. Classification model training
            #5.1 SVM
            #hyper parameter tuning for gamma and C
            if model_type == 'svm':
                gamma_values = [0.001, 0.002, 0.005, 0.01, 0.02]
                C_values = [0.1, 0.2, 0.5, 0.75, 1]
                list_of_param_comination = get_list_of_param_comination([gamma_values, C_values],  ['gamma', 'C'])
                best_hparams, best_model, best_val_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_param_comination, model_type = model_type)

                #get training accuracy of this best model:
                train_accuracy, _ = predict_and_eval(best_model, X_train, y_train)

                ################################################################################################
                #6. Prediction and evaluation on test sat
                # test accuracy
                test_accuracy, _ = predict_and_eval(best_model, X_test, y_test)

                #print for github actions
                print('svm  ','test_size=',ts,' dev_size=',ds,' train_size=',round(1-ts-ds,2),' train_acc=',train_accuracy,' dev_acc',best_val_accuracy,' test_acc=',test_accuracy, ' best_hyper_params=', best_hparams)
                results.append({'run_num':run_num,'model_type':model_type, 'train_accuracy':train_accuracy, 'val_accuracy':best_val_accuracy,
                                'test_acc':test_accuracy, 'best_hparams':best_hparams})
            #5.2 Decision Tree
            #hyper parameter tunning
            if model_type == 'tree':
                max_depth = [5,10,20,30,50,75,100]
                list_of_param_comination = get_list_of_param_comination([max_depth],  ['max_depth'])
                best_hparams, best_model, best_val_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_param_comination,model_type = model_type)

                #get training accuracy of this best model:
                train_accuracy, _ = predict_and_eval(best_model, X_train, y_train)

                ################################################################################################
                #6. Prediction and evaluation on test sat
                # test accuracy
                test_accuracy, _ = predict_and_eval(best_model, X_test, y_test)

                #print for github actions
                print('tree ','test_size=',ts,' dev_size=',ds,' train_size=',round(1-ts-ds,2),' train_acc=',train_accuracy,' dev_acc',best_val_accuracy,' test_acc=',test_accuracy, ' best_hyper_params=', best_hparams)
                results.append({'run_num':run_num,'model_type':model_type, 'train_accuracy':train_accuracy, 'val_accuracy':best_val_accuracy,
                                'test_acc':test_accuracy, 'best_hparams':best_hparams})

result_df = pd.DataFrame(results)
print(result_df.groupby('model_type').describe().T)