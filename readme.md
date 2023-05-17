you can use run.bat file to run the code with the following configuration

run (pip install -r requirements.txt) to install all required packages

if you want to train new model
==> run.bat 1 --path_of_dataset --model_name_to_save --debug_flag

if you want to test on given model
==> run.bat 2 --path_to_testset --model_name_to_run --debug_flag