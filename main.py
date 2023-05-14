from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, ServiceContext, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import openai
import json
import os

"""
sources: 
- Liam Ottley - YouTube: https://www.youtube.com/@LiamOttley
- OpenAI GPT4
"""

class GPTSimpleVectorIndexWithPrompt(GPTSimpleVectorIndex):
    def query(self, query: str, prompt: str = '') -> "GPTSimpleVectorIndex.Response":
        query = prompt + query  # Concatenate the prompt with the query
        return super().query(query)

class Chatbot:
    def __init__(self, api_key, index):
        self.index = index
        openai.api_key = api_key
        self.chat_history = []

    def generate_response(self, user_input):
        prompt = "\n".join([f"{message['role']}: {message['content']}" for message in self.chat_history[-5:]])
        prompt += f"\nUser: {user_input}"
        response = self.index.query(user_input, prompt=prompt)

        message = {"role": "assistant", "content": response.response}
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append(message)
        return message

    def load_chat_history(self, filename):
        try:
            with open(filename, 'r') as f:
                self.chat_history = json.load(f)
        except FileNotFoundError:
            pass

    def save_chat_history(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.chat_history, f)

documents = SimpleDirectoryReader('./pdf').load_data()
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo"))
max_input_size = 4096
num_output = 256
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
custom_LLM_index = GPTSimpleVectorIndexWithPrompt.from_documents(documents, service_context=service_context)

bot = Chatbot(os.environ['OPENAI_API_KEY'], custom_LLM_index)
bot.load_chat_history("chat_history.json")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "goodbye"]:
        print("Bot: Goodbye!")
        bot.save_chat_history("chat_history.json")
        break
    response = bot.generate_response(user_input)
    print(f"Bot: {response['content']}")
