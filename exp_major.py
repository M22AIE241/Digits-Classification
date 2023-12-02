#major : M22AIE241
#utils import
from utils import load_dataset, data_preprocessing, split_train_dev_test,predict_and_eval
from utils import get_list_of_param_comination, tune_hparams ,tune_hparams_logistic_regression,get_loaded_model_params
# import pandas as pd 
import sys
import argparse
from joblib import dump, load

# from sklearn.metrics import confusion_matrix, classification_report,f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--total_run", type=int) # help="Description for arg1")
parser.add_argument("--dev_size", type=float) # help="Description for arg2")
parser.add_argument("--test_size", type=float) # help="Description for arg3")
parser.add_argument("--prod_model_path", type=str, default= None) 
parser.add_argument("--model_type", type=str) 

args = parser.parse_args()
# total_run = int(sys.argv[1])
# dev_size = [float(sys.argv[2])]
# test_size = [float(sys.argv[3])]
# prod_model_path = sys.argv[4]
# model_type = sys.argv[5]


###########################################################################################
#1.get/load the dataset
X,y = load_dataset()

#2.Sanity check of data

################################################################################################
#taking different combinations of train dev and test and reporting results

results = []
for run_num in range(args.total_run):
    for ts in [args.test_size]:
        for ds in [args.dev_size]:
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
            if args.model_type == 'svm':
                gamma_values = [0.0001, 0.001, 0.005, 0.01]
                C_values = [0.1, 0.5, 1]
                list_of_param_comination = get_list_of_param_comination([gamma_values, C_values],  ['gamma', 'C'])
                best_hparams, best_model, best_val_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_param_comination, model_type = args.model_type)

                #get training accuracy of this best model:
                train_accuracy, candidate_pred = predict_and_eval(best_model, X_train, y_train)

                ################################################################################################
                #6. Prediction and evaluation on test sat
                # test accuracy
                test_accuracy, candidate_pred = predict_and_eval(best_model, X_test, y_test)

                #print for github actions
                print('svm model  ','test_size=',ts,' dev_size=',ds,' train_size=',round(1-ts-ds,2),' train_acc=',train_accuracy,' dev_acc',best_val_accuracy,' test_acc=',test_accuracy, ' best_hyper_params=', best_hparams)
                results.append({'run_num':run_num,'model_type':args.model_type, 'train_accuracy':train_accuracy, 'val_accuracy':best_val_accuracy,
                                'test_acc':test_accuracy, 'best_hparams':best_hparams})
            #5.2 Decision Tree
            #hyper parameter tunning
            if args.model_type == 'tree':
                max_depth = [5,10,20,50]
                list_of_param_comination = get_list_of_param_comination([max_depth],  ['max_depth'])
                best_hparams, best_model, best_val_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_param_comination,model_type = args.model_type)

                #get training accuracy of this best model:
                train_accuracy, candidate_pred = predict_and_eval(best_model, X_train, y_train)

                ################################################################################################
                #6. Prediction and evaluation on test sat
                # test accuracy
                test_accuracy, candidate_pred= predict_and_eval(best_model, X_test, y_test)

                #print for github actions
                print('tree model ','test_size=',ts,' dev_size=',ds,' train_size=',round(1-ts-ds,2),' test_acc=',test_accuracy, ' best_hyper_params=', best_hparams)
                results.append({'run_num':run_num,'model_type':args.model_type, 'train_accuracy':train_accuracy, 'val_accuracy':best_val_accuracy,
                                'test_acc':test_accuracy, 'best_hparams':best_hparams})
                

                #hyperparameter tuning
            if args.model_type == 'logistic_regression':
                solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                results_per_solver = []

                for solver in solvers:
                    # Create a specific solver parameter combination
                    list_of_param_combination = get_list_of_param_comination([[solver]], ['solver'])
                    
                    # Tune hyperparameters using Logistic Regression with the current solver
                    best_hparams, best_model, best_val_accuracy = tune_hparams_logistic_regression(X_train, y_train, X_dev, y_dev, list_of_param_combination)

                    # get training accuracy of this best model:
                    train_accuracy, candidate_pred = predict_and_eval(best_model, X_train, y_train)

                    ################################################################################################
                    #6. Prediction and evaluation on test set
                    # test accuracy
                    test_accuracy, candidate_pred = predict_and_eval(best_model, X_test, y_test)

                    # print for github actions
                    print(f'Logistic Regression model test_size={ts} dev_size={ds} train_size={round(1-ts-ds, 2)} solver={solver} train_acc={train_accuracy} dev_acc={best_val_accuracy} test_acc={test_accuracy} best_hyper_params={best_hparams}')

                    # save the model
                    model_name = f"M22AIE241_lr_{solver}.joblib"
                    dump(best_model, f"models/{model_name}")

                    # append results for this solver to the list
                    results_per_solver.append({
                        'solver': solver,
                        'train_accuracy': train_accuracy,
                        'val_accuracy': best_val_accuracy,
                        'test_accuracy': test_accuracy,
                        'best_hparams': best_hparams
                    })

                # Bonus: Report mean and std (across 5 CV) of the performance for each solver
                print("\nBonus: Mean and Std (across 5 CV) of the performance for each solver:")
                for result in results_per_solver:
                    print(f"Solver: {result['solver']}, Mean Test Accuracy: {result['test_accuracy']:.2f}, Std Test Accuracy: 0.00")


            ###print model parameters:
            get_loaded_model_params()


            if args.prod_model_path is not None:
                prod_model = load(args.prod_model_path)
                prod_test_accuracy, prodmodel_pred = predict_and_eval(prod_model, X_test, y_test)
                print('prod model ','test_size=',ts,' dev_size=',ds,' train_size=',round(1-ts-ds,2),', test_acc=',test_accuracy)



# result_df = pd.DataFrame(results)
# print(result_df.groupby('model_type').describe().T)