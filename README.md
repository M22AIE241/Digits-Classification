1. Create an environment:
	conda env list
2. Create an environment and activate:
	if you see base at first means you have successfully installed conda
	conda create -n <env name> python=<version>
	conda activate env_name
3. Deactivate environment
	conda deactivate

4. Create a directory for work
	mkdir digit-classification
	code . --> open visual code 
	save your work to exp.py file
5. Do a ls to see the file
6. Now if you try to run it it will not work as the required libraries are not installed.
7. So we should create a requirement.txt to pass the required installment params
	package_name==version_number
	pip install -r requirement.txt
8. See the version of any lib installed
	pip list | grep matplotlib
9. Running the py file again
	python exp.py (if all packages installed correctly , it should run)

10. Change params , model 
11. GIT :
    a. avoid git add .
    b.