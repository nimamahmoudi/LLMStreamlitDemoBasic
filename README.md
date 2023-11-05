# LLM Streamlit Demo (Basic)

## Installation Process

### With Poetry

To install the required packages, you can install them using poetry:

```sh
poetry install
```

In case you changed requirements.txt, you'll need to run the following command to update the poetry definitions:

```sh
cat requirements.txt | xargs poetry add
```

### Without Poetry

You can use conda to install the required packages:

```sh
conda create -n p311-llm python=3.11
conda activate p311-llm
pip install -r requirements.txt
```

## Running the App

After installing all requirements, you'll need to add your OpenAI API key to the secrets,
or let the user input it in the sidebar every time they visit the page.
You can add your secrets to `.streamlit/secrets.toml` in the following format:

```toml
OPENAI_API_KEY = "sk-..."
```

For more detail about this code, you can follow [my blog posts](https://medium.com/@nima.mahmoudi).
