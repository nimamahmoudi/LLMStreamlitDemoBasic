# langchain imports
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableMap
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage


def format_docs(docs):
  res = ""
  # res = str(docs)
  for doc in docs:
    escaped_page_content = doc.page_content.replace('\n', '\\n')
    res += "<doc>\n"
    res += f"  <content>{escaped_page_content}</content>\n"
    for m in doc.metadata:
      res += f"  <{m}>{doc.metadata[m]}</{m}>\n"
    res += "</doc>\n"

  return res


def get_search_index():
  # load embeddings
  from langchain.vectorstores import FAISS
  from langchain.embeddings.openai import OpenAIEmbeddings

  file_folder = 'index'
  file_name = 'Mahmoudi_Nima_202202_PhD.pdf'
  search_index = FAISS.load_local(folder_path=file_folder,
                                  index_name=file_name + '.index',
                                  embeddings=OpenAIEmbeddings())
  return search_index


def convert_message(m):
  if m['role'] == 'user':
    return HumanMessage(content=m['content'])
  elif m['role'] == 'assistant':
    return AIMessage(content=m['content'])
  elif m['role'] == 'system':
    return SystemMessage(content=m['content'])
  else:
    raise ValueError(f"Unknown role {m['role']}")


_condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {input}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_condense_template)

_rag_template = """Answer the question based only on the following context, citing the page number(s) of the document(s) you used to answer the question:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(_rag_template)


def _format_chat_history(chat_history):

  def format_single_chat_message(m):
    print(m)
    if type(m) is HumanMessage:
      return "Human: " + m.content
    elif type(m) is AIMessage:
      return "Assistant: " + m.content
    elif type(m) is SystemMessage:
      return "System: " + m.content
    else:
      raise ValueError(f"Unknown role {m['role']}")

  return "\n".join([format_single_chat_message(m) for m in chat_history])


def get_rag_chain():
  vectorstore = get_search_index()
  retriever = vectorstore.as_retriever()

  _inputs = RunnableMap(standalone_question=RunnablePassthrough.assign(
      chat_history=lambda x: _format_chat_history(x["chat_history"]))
                        | CONDENSE_QUESTION_PROMPT
                        | ChatOpenAI(temperature=0)
                        | StrOutputParser(), )
  _context = {
      "context": itemgetter("standalone_question") | retriever | format_docs,
      "question": lambda x: x["standalone_question"],
  }
  conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()
  return conversational_qa_chain
