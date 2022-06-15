from transformers import pipeline
def masking(statement):
    classifier = pipeline("fill-mask")
    list = classifier(statement)
    return list