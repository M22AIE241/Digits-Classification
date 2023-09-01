Creating a branch and push to check.(dummy)

1. Create an environment:
	-conda env list
2. Create an environment and activate:
	-if you see base at first means you have successfully installed conda
	-conda create -n <env name> python=<version>
	-conda activate env_name
3. Deactivate environment
	-conda deactivate

4. Create a directory for work
	-mkdir digit-classification
	-code . --> open visual code 
	-save your work to exp.py file
5. Validate the files
6. Run python file and validate.
7. So we should create a requirement.txt to pass the required installment params
	-package_name==version_number

	-pip install -r requirement.txt
8. See the version of any lib installed
	-pip list | grep matplotlib
9. Running the py file again
	-python exp.py (if all packages installed correctly , it should run)

10. Change params , model 
11. GIT :
    a. avoid git add .
    b. git add file_names
    c. git commit 
    d. git push


----------------------------------------------------------------------
system requirements: OS h/w -- may be skipped -- general commodity h/w is required

how to setup: 
	install conda

	conda create -n digits python=3.9 conda activate digits pip install -r requirements.txt

how to run

	python exp.py

Meaning of failure:

	poor performance metrics

	coding runtime/compile error

	the model gave bad predictions on the new test samples during demo.

feature:

	vary model hyper parameters



Overview of train test split :
	100 samples
	2 class classification/binary classification : image of carrot or turnip
	50 samples : carrots   |
	50 samples : turnips   |
	This kind of data distribution : balanced/uniform


	x amount data for training
	n-x amount of data for testing

	70 samples for training : 35 carrots , 35 turnips
	30 samples for testing  : 12 carrots , 15 turnips

	hence,
	calculate some eval metric (train model (70 samples for training : 35 carrots , 35 turnips),(30 samples for testing  : 12 carrots , 15 turnips))==performance

In practice :
    train , development/validation , test

	train = training the model(model type , model hyperparameters , model iterations)
	dev = selecting the model
	test = reportig the performance

