from fastapi import FastAPI
from summerization import get_sumeri
from ner import get_ner, Items
from sentiment import sentiment_analysis
from Qna._exportPairs import exportToJSON
from Qna._getentitypair import GetEntity
from Qna._qna import QuestionAnswer
from fill_mask import masking

class OurModel:
    def __init__(self):
        self.getent = GetEntity()
        self.qa = QuestionAnswer()
        self.export = exportToJSON()

    def getAnswer(self, paragraph, question):

        refined_text = self.getent.preprocess_text([paragraph])
        dataEntities, numberOfPairs = self.getent.get_entity(refined_text)

        if dataEntities:
            # data_in_dict = dataEntities[0].to_dict()
            self.export.dumpdata(dataEntities[0])
            outputAnswer = self.qa.findanswer(str(question+"?"), numberOfPairs)
            if outputAnswer == []:
                return None
            return outputAnswer
        return None


app = FastAPI(docs_url="/api/v1/docs")
@app.post("/api/v1/ner",response_model=Items,response_model_exclude_unset=True)
def name_enity_reg(statement : str):
    return get_ner(statement)

@app.post("/api/v1/sentiment")
def sentiment(senti_str : str):
    return sentiment_analysis(senti_str)

@app.post("/api/v1/summerization")
def summerization(paragraph :str,length_ : int):
    return get_sumeri(paragraph,length_)

@app.post("/api/v1/qna")
def question_answering(input_paragraph : str, input_question : str):
    model = OurModel()
    my_answer = model.getAnswer(input_paragraph, input_question)
    return my_answer
    
@app.post("/api/v1/fill_mask")
def fill_mask(statement : str):
    return masking(statement)