import streamlit as st
import requests, re, json

SYSTEM_CONTENT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

You must reply in Japanese.
"""
    
def input_question(question):
    st.session_state.question = st.text_input("入力",question)

def answer(input):
    # Stream
    url = 'https://bam-api.res.ibm.com/v2/text/chat_stream?version=2024-05-10'
    # No-Stream
    # url = 'https://bam-api.res.ibm.com/v2/text/chat?version=2024-05-10'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer pak-JcUU6jMIwSl9xJtO-bV-IvXbuU3sUZufLfMvWo5ThXc'
    }
    data = {
        "model_id": "meta-llama/llama-3-70b-instruct",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_CONTENT
            },
            {
                "role": "user",
                "content": input
            }
        ],
        "parameters": {
            "decoding_method": "greedy",
            "min_new_tokens": 1,
            "max_new_tokens": 1000
        },
        "moderations": {}
    }

    print("Query: " + input)
    # Stream
    response_text = ''
    data_pattern = re.compile(rb'^data:\s*(.*)')
    response = requests.post(url, headers=headers, json=data, stream=True)
    for line in response.iter_lines():
        match = data_pattern.match(line)
        if match:
            json_data = match.group(1).decode('utf-8')
            json_loaded = json.loads(json_data)
            content = json_loaded["results"][0]["generated_text"]
            response_text += content
            yield content
    print("Output: " + response_text)

    # No-Stream
    # response = requests.post(url, headers=headers, json=data, stream=True)
    # try:
    #     response = response.json()["results"][0]["generated_text"]
    # except:
    #     print(response)
    #     print(response.json())
    # output_memory = response
    # return response

st.set_page_config(layout="wide")
st.title("Chatbot Sample")

st.write("こちらはチャットボットのサンプルです。1問1答形式で、会話の履歴は覚えていません。")
input_question("JUnit5の使い方を、Junit4との違いに触れつつ教えて")

if st.button('実行'):
    st.write("---")
    with st.spinner("実行中…"):
        response = answer(st.session_state.question)
        st.write(response)
