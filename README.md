# LLM Streamlit Demo (Basic)

- [Live Demo](https://llmappdemobasic.streamlit.app/)

## Installation Process

### Without Poetry

You can use conda to install the required packages:

```sh
conda create -n p311-llm python=3.11
conda activate p311-llm
pip install -r requirements.txt
```

### With Poetry (optional)

To install the required packages, you can install them using poetry:

```sh
# (optional) to set poetry to use the project folder
# poetry config virtualenvs.in-project true
# install all dependencies
poetry install
```

In case you changed requirements.txt, you'll need to run the following command to update the poetry definitions:

```sh
cat requirements.txt | xargs poetry add
```

You can then enable poetry shell:

```sh
poetry shell
```

## Running the App

After installing all requirements, you'll need to add your OpenAI API key to the secrets,
or let the user input it in the sidebar every time they visit the page.
You can add your secrets to `.streamlit/secrets.toml` in the following format:

```toml
OPENAI_API_KEY = "sk-..."
```

Then, you can run the code using the following command:

```sh
streamlit run app.py
```

For more detail about this code, you can follow [my blog posts](https://medium.com/@nima.mahmoudi).
