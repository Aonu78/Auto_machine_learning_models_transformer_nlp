from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from typing import List

class Item(BaseModel):
    entity: str
    score: float
    index: int
    word: str
    start: int
    end : int
    
class Items(BaseModel):
    items: List[Item]

model_name = "dslim/bert-base-NER"
def get_ner(statement):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    ner_results = nlp(statement)
    return Items(items=ner_results)
