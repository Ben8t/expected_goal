FROM python:3.6

RUN pip install --upgrade pip

RUN pip install numpy pandas scikit-learn xgboost sklearn-pandas joblib

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

CMD ["/bin/bash"]