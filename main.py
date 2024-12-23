from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from flask import Flask, request, render_template, session, jsonify

"""
추후에 LLM 평가를 위해서 사용할 함수들
import sacrebleu
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from langchain_openai import ChatOpenAI
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from nltk import word_tokenize
import nltk
from datasets import load_dataset
"""
app = Flask(__name__)

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
    api_key="",
    model_name='gpt-4o', **params, model_kwargs = kwargs)

loader = CSVLoader(file_path='test_AI.csv',encoding='unicode_escape')
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

"""
text = input('말을 입력해 주세요')

def socratic_method(text):
    formatted_template = template.format(input=text)
    result = retrieval_qa({"query": formatted_template})
    response = result["result"]
    return response

while True:
    text = input('말을 입력해 주세요 : ')
    if text.lower()== 'exit':
        print('지금까지 대화하느라 즐거운 시간이였어 성균관대에서 즐거운 시간이 되길 바래 ')
        break
    else:
        response = socratic_method(text)
        print(response)
    """
@app.route('/ask', methods=['POST'])
def ask():
    try:
        content = request.json
        question = content.get('question')
        if not question:
            return jsonify("Error: No question provided."), 400
        if question.lower() == "exit":
            return jsonify("지금까지 너와 대화한 AI였어 좋은 학교 생활이 되길 바래"), 200
        formatted_template = template.format(input=question)
        result = retrieval_qa({"query": formatted_template})
        response = result["result"]
        return jsonify(response)
    except Exception as e:
        return jsonify(f"An error occurred: {str(e)}"), 500

@app.route('/')
def home():
    return render_template('design.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)

"""
이 부분은 추후에 수정해서 Test Case를 만든다면 수정할 예정.
nltk.download('punkt')
dataset = load_dataset("nbertagnolli/counsel-chat")

test_data = [
    {
        'input': dataset['train'][i]['questionText'],
        'expected_output': dataset['train'][i]['answerText']
    }
    for i in range(10)
]

llm1 = ChatOpenAI(api_key="your_openai_api_key")
gen = []

for example in test_data:
    response_GPT = llm1.invoke(example["input"])
    gen.append(response_GPT)
gen_str = [str(response) for response in gen]

def generate_responses(test_data):
    generated_responses = []
    for example in test_data:
        input_text = example["input"]
        generated_response = retrieval_qa({"query": input_text})["result"]
        generated_responses.append(generated_response)
    return generated_responses

def calculate_rouge_scores(generated_responses, expected_responses):
    rouge = Rouge()
    scores = rouge.get_scores(generated_responses, expected_responses, avg=True)
    return scores

def calculate_bleu_score(generated_responses, expected_responses):
    refs = [[word_tokenize(ref)] for ref in expected_responses]
    sys = [word_tokenize(gen_resp) for gen_resp in generated_responses]
    bleu = corpus_bleu(refs, sys)
    return bleu

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

bleu_without = calculate_bleu_score(gen_str, expected_responses)
perplexity_without = calculate_perplexity(gen_str)
rogue_without = calculate_rouge_scores(gen_str, expected_responses)
meteor_without = calculate_meteor(gen_str, expected_responses)

print("Generated responses:")
print(generated_responses)
print("Expected responses:")
print(expected_responses)
print("BLEU score:", bleu_score)
print("Perplexity:", perplexity)
print("ROUGE:", rouge)
print("METEOR:", meteor)
print("Without method:")
print("BLEU score:", bleu_without)
print("Perplexity:", perplexity_without)
print("ROUGE:", rogue_without)
print("METEOR:", meteor_without)
"""
