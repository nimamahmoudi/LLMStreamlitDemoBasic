import streamlit as st
import os
import embed_pdf

# create sidebar and ask for openai api key if not set in secrets
secrets_file_path = os.path.join(".streamlit", "secrets.toml")
if os.path.exists(secrets_file_path):
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        else:
            print("OpenAI API Key found in environment variables")
    except FileNotFoundError:
        print('Secrets file not found')
else:
    print('Secrets file not found')

if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    os.environ["OPENAI_API_KEY"] = st.sidebar.text_input(
        "OpenAI API Key", type="password"
    )
else:
    if st.sidebar.button("Embed Documents"):
        st.sidebar.info("Embedding documents...")
        try:
            embed_pdf.embed_all_pdf_docs()
            st.sidebar.info("Done!")
        except Exception as e:
            st.sidebar.error(e)
            st.sidebar.error("Failed to embed documents.")

chosen_file = st.radio(
    "Choose a file to search", embed_pdf.get_all_index_files(), index=0
)

# create the app
st.title("Welcome to NimaGPT")

# check if openai api key is set
if not os.getenv('OPENAI_API_KEY', '').startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
    st.stop()

# load the agent
from llm_helper import convert_message, get_rag_chain

custom_chain = get_rag_chain(chosen_file)

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
            chat_history = [convert_message(m) for m in st.session_state.messages[:-1]]
        else:
            chat_history = []

        full_response = ""
        for response in custom_chain.stream(
            {"input": prompt, "chat_history": chat_history}
        ):
            if "output" in response:
                full_response += response["output"]
            else:
                full_response += response.content

            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # add the full response to the message history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
