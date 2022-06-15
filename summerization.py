from transformers import T5Tokenizer, T5ForConditionalGeneration
model_name = "cahya/t5-base-indonesian-summarization-cased"
# ARTICLE_TO_SUMMARIZE = "Mohammad Ali Jinnah the founder of Pakistan, was born on December 25, 1876, in a house known as Wazir Mansion located in Karachi. His father’s name was Jinnah Poonja, and Mother was Mithibai, he belongs to a merchant family. He was a great politician and a well-known lawyer of his time. He struggled a lot for the freedom of the Muslims of sub-continent and on the behalf of his extraordinary efforts, he was rewarded with the title of “Quaid-e-Azam” (the father of the nation) by maulana Mazharuddin.Quaid e Azam Mohammad Ali Jinnah received his early education from Sindh Madarsat ul Islam and Christian missionary school. He went to England for higher education and got admission at Lincoln’s Inn law school in London. At the age of 20 he enrolled in Bombay high court when he came back to British India, he was the youngest one to enter the bar, where he started to take interest in political affairs of the nation and became famous in the next three years. The advocate general of Bombay invited him to work for his bar and after six months offered a salary of 1500 rupees per month, which was the huge amount that time but he gently refuses the offer and stated that he planned to earn 1500 daily and proved it possible in future by his flawless efforts. But as a Governor-General of newly state Pakistan, he fixed 1 rupee as his monthly salary. He was the man of the judiciary and sensible personality."
def get_sumeri(ARTICLE_TO_SUMMARIZE,GET_LENGTH):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    input_ids = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors='pt')
    summary_ids = model.generate(input_ids,
                min_length=20,
                max_length=GET_LENGTH,
                num_beams=10,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=2,
                use_cache=True,
                do_sample = True,
                temperature = 0.8,
                top_k = 50,
                top_p = 0.95)

    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text
# a = get_sumeri(ARTICLE_TO_SUMMARIZE)
# print(a)