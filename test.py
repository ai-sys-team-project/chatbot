import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain_openai import ChatOpenAI
from datasets import load_dataset, load_metric
import pandas as pd
from flask import Flask
from nltk import word_tokenize
from rouge import Rouge
import sacrebleu
import nltk
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings

nltk.download('punkt')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# Load necessary data
test = pd.read_csv('C:/Users/tjxod/PycharmProjects/AI_System/testcase(~100).csv', encoding='utf-8-sig')
test_data = [{'input': test['Question'][i], 'expected_output': test['Answer'][i]} for i in range(99)]


params = {
    "temperature": 0.7,
    "max_tokens": 100,
}

kwargs = {
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "stop": ["\n"]
}

llm = ChatOpenAI(
    api_key="sk-proj-eIHTi7cUfaONgyffJ2E8T3BlbkFJizj14PZd0boMuMunsGCe",
    model_name='gpt-4o',
    **params,
    model_kwargs=kwargs
)

loader = CSVLoader(file_path='test_AI.csv', encoding='unicode_escape')
data = loader.load()

template = """
성균관대학교의 새로운 학생으로서, 학교 생활에 대한 궁금증을 해결해 드리겠습니다. 아래에 질문을 입력하면, 당신의 질문에 가장 잘 맞는 정보를 제공해드릴 수 있도록 노력하겠습니다.
질문: {input}
응답: 응답의 내용은 최대 50글자 내외로 답을 해줄래?
"""
embeddings = OpenAIEmbeddings(openai_api_key="sk-proj-eIHTi7cUfaONgyffJ2E8T3BlbkFJizj14PZd0boMuMunsGCe")
vectorstore = FAISS.from_documents(data, embeddings)
retriever = vectorstore.as_retriever()
qa_chain = load_qa_chain(llm, chain_type='refine')
retrieval_qa = RetrievalQA(
    combine_documents_chain=qa_chain,
    retriever=retriever,
    return_source_documents=True
)
prompt = PromptTemplate.from_template(template=template)

# Function to generate responses using the refined model
def generate_responses(test_data):
    generated_responses = []
    for example in test_data:
        input_text = example["input"]
        generated_response = retrieval_qa({"query": input_text})["result"]
        generated_responses.append(generated_response)
    return generated_responses

# Function to calculate BLEU score using sacrebleu
def calculate_bleu_score(generated_responses, expected_responses):
    bleu = sacrebleu.corpus_bleu(generated_responses, [expected_responses])
    return bleu.score

# Function to calculate ROUGE scores
def calculate_rouge_scores(generated_responses, expected_responses):
    rouge = Rouge()
    scores = rouge.get_scores(generated_responses, expected_responses, avg=True)
    return scores

# Function to calculate perplexity
def calculate_perplexity(generated_responses):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    perplexities = []
    for response in generated_responses:
        input_ids = tokenizer(response, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        perplexities.append(perplexity.item())
    return sum(perplexities) / len(perplexities)

# Function to calculate METEOR score
def calculate_meteor(generated_responses, expected_responses):
    meteor_metric = load_metric('meteor')
    results = meteor_metric.compute(predictions=generated_responses, references=expected_responses)
    return results['meteor']

# Generate responses and calculate evaluation metrics
generated_responses = generate_responses(test_data)
expected_responses = [example["expected_output"] for example in test_data]

bleu_score = calculate_bleu_score(generated_responses, expected_responses)
perplexity = calculate_perplexity(generated_responses)
rouge = calculate_rouge_scores(generated_responses, expected_responses)
meteor = calculate_meteor(generated_responses, expected_responses)

print("BLEU score:", bleu_score)
print("Perplexity:", perplexity)
print("ROUGE:", rouge)
print("METEOR:", meteor)

