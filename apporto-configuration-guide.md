## Package Installation

```
uv venv
source .venv/bin/activate

uv pip install jupyterlab
uv pip install \
  matplotlib \
  seaborn \
  numpy \
  pandas \
  scikit-learn \
  tensorflow \
  gdown \
  scipy \
  spacy \
  nltk \
  torch \
  xgboost \
  graphviz \
  ydf \
  pydot \
  ucimlrepo \
  tensorflow-text \
  tensorflow-datasets
```

## Github Configuration


```
ssh-keygen -t rsa -b 4096
cat ~/.ssh/id_rsa.pub

git config --global user.name "Your Name"
git config --global user.email "you@jcu.edu"
```