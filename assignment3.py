from utils import *
from itertools import product


test_split=[0.1, 0.2, 0.3]
dev_split=[0.1, 0.2, 0.3]
list_of_test_dev_split = [(a,b) for a, b in product(test_split, dev_split)]

gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
C_ranges = [0.1, 1, 2, 5, 10]
list_of_all_param_combination = [{"gamma":a,"C":b} for a, b in product(gamma_ranges, C_ranges)]
    

for index, split in enumerate(list_of_test_dev_split):

    x,y=read_digits()
    x=preprocess_data(x)
    X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(x, y, test_size=split[0], dev_size=split[1])
    model=train_model(X_train,y_train,{'gamma':0.001},model_type="svm")
    train_acc = predict_and_eval(model,X_train, y_train)
    dev_acc = predict_and_eval(model,X_dev, y_dev)
    test_acc = predict_and_eval(model,X_test, y_test)
    print(f"test_size={split[0]} dev_size={split[1]} train_size={1-split[0]-split[1]} train_acc={train_acc} dev_acc={dev_acc} test_acc={test_acc} ")
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]

    best_model,optimal_gamma,optimal_C,test_acc=hyper_param_tuning(X_train, X_test, X_dev, y_train, y_test, y_dev, list_of_all_param_combination)
    print(f"best hyper parameters for this run, gamma:{optimal_gamma}, C:{optimal_C}\n\n")
