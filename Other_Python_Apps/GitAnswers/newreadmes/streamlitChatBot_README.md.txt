# Streamlit ChatGPT with Azure Speech Services

This Streamlit app uses ChatGPT along with Azure Speech Services to provide an interactive and engaging chatting experience for users. Users can chat with the AI both using text input and voice input, making your experience more personalized and convenient.
***
## How to Use

1. Clone the repository:

```bash
git clone https://github.com/USERNAME/chatApp.git
```

2. Navigate to the project folder:

```bash
cd chatApp
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```
***
## Requirements

Before you begin, make sure you run 
```bash
pip install -r requirements.txt
```
***
## Text

To chat with ChatGPT using text inputs, follow these steps:

1. Navigate to the sidebar and select "Text" from the dropdown menu.
2. Enter your [OpenAI API Key](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/#:~:text=Go%20to%20OpenAI%27s%20Platform%20website,generate%20a%20new%20API%20key.) and hit Enter.
3. The chat window should appear in the main page.
4. Type your chats in the box provided and hit Enter to send your messages.
***
## Voice

To chat with ChatGPT using voice inputs through Azure Speech Services, follow these steps:

1. Navigate to the sidebar and select "Voice" from the dropdown menu.
2. Enter your [OpenAI API Key](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/#:~:text=Go%20to%20OpenAI%27s%20Platform%20website,generate%20a%20new%20API%20key.) and your [Azure Speech Subscription Key](https://carldesouza.com/get-a-microsoft-cognitive-services-subscription-key/).
3. The bot should greet you right away.
4. When you see "Speak Now" you may speak, and when you see "Speech received" along with a transcript of what it received, the bot is preparing to respond. When you see "Speech received" become grayed out, you may speak again.
5. Listen to the AI-generated response through your speakers or headphones.

*NOTE: You need to have an Azure Speech Services subscription key to use the voice input feature. Also the conversation will be just like the chat bot, so you must wait until the bot finishes speaking before talking.

Enjoy your chatting experience with Streamlit ChatGPT with Azure Speech Services!
