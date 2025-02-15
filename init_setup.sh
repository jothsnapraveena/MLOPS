echo [$(date)]: "START"
echo [$(date)]: "Creating conda env with python 3.8"
conda create --prefix ./mlops python=3.8 -y
echo [$(date)]: "activate env"
source activate ./mlops
echo [$(date)]: "intalling requirements"
pip install -r requirements.txt
echo [$(date)]: "END"