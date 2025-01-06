session9_mlops_revisited
==============================

Practice project for dvc pipeline cookiecutter code quality git tracking

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


1)Createed cookiecutter template 
2)modularize code data_ingestion, data_preprocessing, feature_engineering 
3)data tracking using dvc and location provided is s3 
--->pip install dvc[s3]
--->dvc remote add -d myremote s3://campusxmlopsdvcbucket
Now whenever you do dvc push the tracked files will be stored in aws s3
4)Track the model training part with mlflow
-->mlflow tracking server is set on aws EC2 
following are steps to set mlflow tracking server on aws ec2 machine.

a)First craete the user on aws which will have acces to s3 bucket and ec2 serevr.
  While creating server you will have to attche policies.
  IAM user -->Create user --->set permissions --->create users
  for polices following json is attaced

  {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": "arn:aws:s3:::campusxmlopsdvcbucket"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject"
            ],
            "Resource": "arn:aws:s3:::campusxmlopsdvcbucket/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:StartInstances",
                "ec2:StopInstances",
                "ec2:RebootInstances",
                "ec2:DescribeInstances"
            ],
            "Resource": "arn:aws:ec2:eu-north-1:730335412759:instance/i-07df27fb25f319de4"
        }
    ]
}

b)Now open ec2 serevr and execute following setps (check in campusx session16 (part2))
pip freeze
    2  history
    3  sudo apt update
    4  sudo apt install python3-pip
    5  pi freeze
    6  pip freeze
    7  sudo apt install pipx
    8  sudo pipx ensurepath
    9  pipx install pipenv
   10  export PATH=$PATH:/home/ubuntu/.local/bin
   11  echo 'export PATH=$PATH:/home/ubuntu/.local/bin'>> ~/.bashrc
   12  source ~/.bashrc
   13  ll
   14  mkdir mlflow
   15  ll
   16  cd mlflow/
   17  pipenv shell
   18  echo $VIRTUAL_ENV
   19  pipenv install setpools
   20  ps aux | grep pipenv
   21  ls -l
   22  chmod 644 Pipfile Pipfile.lock
   23  pwd
   24  ll
   25  cd mlflow/
   26  ps aux | grep pipenv
   27  ll
   28  chmod 644 Pipfile Pipfile.lock
   29  ll
   30  chmod 777 Pipfile Pipfile.lock
   31  ll
   32  pipenv install setpools
   33  ll
   34  cd mlflow/
   35  ll
   36  pipenv install setpools
   37  pipenv shell
   38  vi Pipfile
   39  vi Pipfile.lock 
   40  python --version
   41  vi Pipfile.lock 
   42  vi Pipfile
   43  ll
   44  pipenv --clear
   45  ll
   46  pipenv install setpools
   47  pipenv --clear
   48  ll
   49  rm Pipfile.lock 
   50  ll
   51  vi Pipfile 
   52  python --version
   53  vi Pipfile 
   54  pipenv lock
   55  ll
   56  chmod 777 Pipfile.lock
   57  ll
   58  vi Pipfile.lock 
   59  pipenv --clear
   60  pipenv install --skip-lock
   61  vi Pipfile.lock 
   62  pipenv install --verbose
   63  pipenv install setpools
   64  pipenv check
   65  pipenv install --verbose
   66  pip install pipdeptree
   67  pipdeptree
   68  pip install pip-tools
   69  pip-compile --verbose
   70  pipenv install setpools
   71  pipenv install mlflow
   72  pipenv install awscli
   73  pipenv install boto3
   74  aws configure
   75  mlflow server -h 0.0.0.0 --default-artifact-root-root s3://campusxmlopsdvcbucket
   76  mlflow server -h 0.0.0.0 --default-artifact-root s3://campusxmlopsdvcbucket
   77  history

c)Note that before excecuting the last command mentioned above i.e ( mlflow server -h 0.0.0.0 --default-artifact-root s3://campusxmlopsdvcbucket)
you will hav to go the 
ec2 machine => security => security group => edit inbound rules =>add rule =>custom tcp, port: 5000, 0.0.0/0 (this indicates any computer can acess this server. However you can restrict it)
d)now add the mlflow.set_tracking_uri("http://ec2-13-51-172-108.eu-north-1.compute.amazonaws.com:5000/")