from Qna._exportPairs import exportToJSON
from Qna._getentitypair import GetEntity
from Qna._qna import QuestionAnswer

# app = FastAPI()


class CheckAndSave:
    """docstring for CheckAndSave."""

    def __init__(self):
        super(CheckAndSave, self).__init__()

    def createdataset(self, para, que, ent, ans1, ans2):

        wholedata = {"para":[str(para)],"que":[[str(que)]], "entities":[ent], "ans1": [ans1], "ans2":[ans2]}
