import streamlit as st
import os

# create sidebar and ask for openai api key if not set in secrets
if "OPENAI_API_KEY" in st.secrets:
  os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
  os.environ["OPENAI_API_KEY"] = st.sidebar.text_input("OpenAI API Key",
                                                       type="password")

# create the app
st.title("Welcome to NimaGPT")

# check if openai api key is set
if "OPENAI_API_KEY" not in os.environ or not os.environ[
    "OPENAI_API_KEY"].startswith("sk-"):
  st.warning("Please enter your OpenAI API key!", icon="⚠")
  st.stop()

# load the agent
from llm_helper import convert_message, get_rag_chain

custom_chain = get_rag_chain()

# create the message history state
if "messages" not in st.session_state:
  st.session_state.messages = []

# render older messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# render the chat input
prompt = st.chat_input("Enter your message...")
if prompt:
  st.session_state.messages.append({"role": "user", "content": prompt})

  # render the user's new message
  with st.chat_message("user"):
    st.markdown(prompt)

  # render the assistant's response
  with st.chat_message("assistant"):
    message_placeholder = st.empty()

    if "messages" in st.session_state:
      chat_history = [
          convert_message(m) for m in st.session_state.messages[:-1]
      ]
    else:
      chat_history = []

    full_response = ""
    for response in custom_chain.stream({
        "input": prompt,
        "chat_history": chat_history
    }):
      if "output" in response:
        full_response += response["output"]
      else:
        full_response += response.content

      message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)

  # add the full response to the message history
  st.session_state.messages.append({
      "role": "assistant",
      "content": full_response
  })
