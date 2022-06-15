from transformers import pipeline
purpose_of_model = "sentiment-analysis"
used_model = "siebert/sentiment-roberta-large-english"
def sentiment_analysis(senti_str):
    senti_anlys = pipeline(purpose_of_model,model=used_model)
    return senti_anlys(senti_str)
