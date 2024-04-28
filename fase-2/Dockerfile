FROM python:3.12

ADD model.pkl .
ADD predict.py .
ADD 02_run_scripts.ipynb .
ADD test_data_input.csv .
ADD test_data_target.csv .
ADD test.csv .
ADD train.csv .
ADD train.py .

RUN pip install lightgbm==3.2.1
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install numpy
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install loguru
RUN pip install argparse