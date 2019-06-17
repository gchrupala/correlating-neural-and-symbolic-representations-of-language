# correlating-neural-and-symbolic-representations-of-language

Code repository for the paper:

Grzegorz Chrupała and Afra Alishahi. 2019.  Correlating neural and symbolic representations of language. In Proceedings of ACL. https://arxiv.org/abs/1905.06401

## Installation

Clone repo and set up and activate a virtual environment with python3

```
cd correlating-neural-and-symbolic-representations-of-language
virtualenv -p python3 .
```

Install pre-requisites annd the Python code (in development mode if you will be modifying something).

```
pip install -r requirements.txt
python setup.py develop
```

Download InferSent moodels and data and unpack them:

```
wget http://grzegorz.chrupala.me/data/infersent.tgz
tar zxvf infersent.tgz
```



## Usage

- Train the LSTM models in [experiments](experiments) by running the `run.py` files.
- Run the `main` function in module [rsa.report](rsa/report.py) to generate the [results](report/results.tex):
```
python -c 'import rsa.report as R; R.main(open("report/results.tex", "w"))'
```
