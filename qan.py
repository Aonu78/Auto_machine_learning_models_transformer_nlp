
from transformers import GPT2Tokenizer, GPT2LMHeadModel
token = "gpt2"
modeliz = "danyaljj/gpt2_question_answering_squad2"
def ans_the_ques(query,ques):
    tokenizer = GPT2Tokenizer.from_pretrained(token)
    model = GPT2LMHeadModel.from_pretrained(modeliz)
    input_ids = tokenizer.encode(query +". Q: "+ ques + " A: ", return_tensors="pt")
    outputs = model.generate(input_ids,pad_token_id=tokenizer.eos_token_id)
    a = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(a)
    res = a.split()
    tem_str = ""
    for i in range(len(res)):
        if str(res[i])== "A:":
            return(str(res[i+1]))
