# FROM tiangolo/uvicorn-gunicorn:python3.1

# COPY requirements.txt .
# WORKDIR /Auto_ml/
# RUN python -m pip install -r requirements.txt

# # RUN pip3 install -r requirements.txt
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM tiangolo/uvicorn-gunicorn:python3.7

WORKDIR /Auto_ml

# COPY ./Auto_ml/requirment.txt .
RUN pip install -r .ex/Auto_ml/requirment.txt
RUN /Auto_ml
CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]