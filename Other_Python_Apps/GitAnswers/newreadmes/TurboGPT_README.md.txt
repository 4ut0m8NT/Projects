# TurboGPT
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://turbogpt.streamlit.app/)


TurboGPT is a chatbot similar to ChatGPT, powered by the GPT-3.5 language model, developed by [Tushar Aggarwal](https://tushar-aggarwal.com). This chatbot is built using Streamlit, a Python library for creating web applications. With TurboGPT, you can have interactive conversations with an AI-powered language model, making it ideal for various applications such as virtual assistants, customer support, and more.
### Note: Due to obvious reasons, an OpenAI API key is needed, I promise, TurboGPT won't store your API key, but will only store conversation history. After you exit, that too will be dumped.

![Untitled design](https://github.com/tushar2704/TurboGPT/assets/66141195/2c4e3a06-c043-4be9-ac9d-d4ee685c0e55)


## Features
- **Natural Language Processing**: TurboGPT utilizes the advanced natural language processing capabilities of the GPT-3.5 model, allowing it to understand and generate human-like responses.
- **Interactive Conversations**: Engage in interactive and dynamic conversations with TurboGPT. You can chat with the chatbot and receive real-time responses.
- **Easy-to-Use Interface**: TurboGPT is built using Streamlit, which provides a user-friendly and intuitive interface for interacting with the chatbot. You can easily input your queries and view the responses in a conversational format.
- **Customizable Responses**: TurboGPT can be customized to generate responses according to specific prompts or scenarios. You can tailor the chatbot's behavior to suit your application requirements.
- **Application Integration**: TurboGPT can be easily integrated into existing web applications or services. You can leverage the chatbot's capabilities to enhance user experiences and provide automated assistance.

## Getting Started

To run TurboGPT locally, follow these steps:

1. Clone the TurboGPT repository: `git clone https://github.com/tushar2704/TurboGPT.git`
2. Navigate to the project directory: `cd TurboGPT`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the application: `streamlit run app.py`
5. Access the chatbot in your browser at `http://localhost:8501`

## Configuration

The configuration for TurboGPT can be found in the `TurboGPT.py` file. Here, you can modify parameters such as the GPT model version, maximum response length, and more, to customize the behavior of the chatbot.



## Limitations

While TurboGPT provides advanced natural language processing capabilities, it also has certain limitations:

- **Contextual Understanding**: The chatbot does not maintain long-term memory of previous interactions, resulting in a lack of context in ongoing conversations.
- **Subjective Responses**: TurboGPT generates responses based on patterns and examples from its training data, which means it may not always provide objective or accurate information.
- **Safety Concerns**: The chatbot may produce biased or inappropriate responses. It is recommended to implement content filtering and moderation mechanisms to ensure responsible usage.

## License

TurboGPT is released under the [MIT License](https://github.com/tushar2704/TurboGPT/blob/main/LICENSE).

## Acknowledgments

TurboGPT is based on the GPT-3.5 language model developed by OpenAI. We would like to acknowledge their contributions to the field of natural language processing and their efforts in making the GPT models accessible for developers.
