import torch
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate
import gradio as gr
import time

custom_prompt_template = """
You are an AI Chatbot named Sunny, created by 'Sic Team', and your task is to provide information to users and chat with them based on given user's query. Below is the user's query.
Query: {query}

You just return the helpful message in English and always try to provide relevant answers to the user's query.
"""


def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=['query'])
    return prompt


def load_model():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the locally downloaded model here, specifying the device
    llm = CTransformers(
        model="TheBloke/zephyr-7B-beta-GGUF",
        model_type="mistral",
        max_new_tokens=4384,
        temperature=0.2,
        repetition_penalty=1.13,
        device=device  # Set the device explicitly during model initialization
    )

    return llm


def chain_pipeline():
    llm = load_model()
    main_prompt = set_custom_prompt()
    main_chain = LLMChain(prompt=main_prompt, llm=llm)
    return main_chain


llmchain = chain_pipeline()


def bot(query):
    llm_response = llmchain.run({"query": query})
    return llm_response


with gr.Blocks(title='Sunny', css="footer {visibility: hidden}; background-color: #000; overflow: hidden") as main:
    gr.HTML("""
        <style>
            ::-webkit-scrollbar {
                width: 10px; /* Set the width of the scrollbar */
            }

            ::-webkit-scrollbar-track {
                background: #000; /* Set the track background color to black */
            }

            ::-webkit-scrollbar-thumb {
                background-color: #fff; /* Set the thumb color to black */
                border-radius: var(--scroll-radius);
                height: 10px;

            }
        </style>
    """)

    gr.Markdown("# Sunny Chatbot")
    chatbot = gr.Chatbot([], elem_id="chatbot", height=660)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    css="footer {visibility: hidden}"

    def respond(message, chat_history):
        bot_message = bot(message)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

main.launch(share=False)
