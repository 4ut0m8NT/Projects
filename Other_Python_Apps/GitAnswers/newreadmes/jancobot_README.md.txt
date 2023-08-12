<img src="https://i.imgur.com/3hgDsT0.png" width="200" height="200" align="center"/>

# JancoBot
A chatbot that is leveraging GPT-4, Langchain Framework and StreamLit to simulate a convo with Jean-Marc Jancovici

![alt text](https://i.imgur.com/rbV0BPM.png)
## How to run
1. Install the requirements with `pip install -r requirements.txt`
2. Add your LLM global prompt in a file called `prompt.py` such as :
```python
janco_prompt = """..."""
```
3. Add your OpenAI API key to your environment variables OR add it to a file called `.env` such as :
```
OPENAI_API_KEY=...
```
4. Run the app with `streamlit run app.py`
5. Enjoy!

## Papar Information
- Title:  `JancoBot`
- Authors:  `Joris Carol`

## Install & Dependence
langchain==0.0.177
openai
streamlit==1.22.0
streamlit_chat==0.0.2.2
easyocr==1.7.0
backports.zoneinfo==0.2.1;python_version<"3.8"
faiss-cpu==1.7.4
tiktoken==0.4.0
typing-inspect==0.8.0
typing_extensions==4.5.0
python-dotenv
youtube-transcript-api

## Directory Hierarchy
```
|—— .gitignore
|—— App.py
|—— __pycache__
|    |—— prompt.cpython-38.pyc
|—— jancobot
|    |—— .gitignore
|    |—— App.py
|    |—— README.md
|—— prompt.py
|—— requirements.txt
```
