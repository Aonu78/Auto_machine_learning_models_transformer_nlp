# Auto_machine_learning_models_transformer_nlp
<h1 align="center">Dirrerent models</h1>


<p align="center">
<img alt="Auto ML" src="fastapi.png">
</p>


## Introduction

Project Insight is designed to create NLP as a service with code base for both front end GUI (**`streamlit`**)  and backend server (**`FastApi`**) the usage of transformers models on various downstream NLP task.

The downstream NLP tasks covered:

* Entity Recognition

* Sentiment Analysis

* Fill Mask

* Summarization

* Question Answering 

* `To Do`

The user can select different models from the drop down to run the inference.

The users can also directly use the backend fastapi server to have a command line inference. 
using :
        <b>  uvicorn main:app --reload  </b>

<a id='section01a'></a>

### Features of the solution

* **Python Code Base**: Built using `Fastapi` make the complete code base in Python.
* **Expandable**: The backend is desinged in a way that it can be expanded with more Transformer based models and it will be available in the front end app automatically. 




