docker build -t mlops_assignment_4:v1 -f docker/Dockerfile .
echo "=======================================BEFORE EXECUTING CONTAINERS RUN================================================"
ls -lh models
echo "======================================================================================================================"

docker run -v /mnt/c/Users/soura/Desktop/ML-OPS/Digits-Classification/models:/digits/models  mlops_assignment_4:v1 --total_run 1 --dev_size 0.3 --test_size 0.2 --model_type 'svm' 
docker run -v /mnt/c/Users/soura/Desktop/ML-OPS/Digits-Classification/models:/digits/models  mlops_assignment_4:v1 --total_run 1 --dev_size 0.3 --test_size 0.2 --model_type 'tree'

echo "=======================================AFTER EXECUTING CONTAINERS RUN================================================"
ls -lh models
echo "======================================================================================================================"

