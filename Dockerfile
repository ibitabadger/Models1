FROM python:3.12
ADD main.py .
RUN pip install scikit-learn
CMD ["python", "./main.py"]