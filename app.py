# Important libraries
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from typing import List, Literal, Optional
import uuid
from transformers import AutoModel, AutoTokenizer
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
# from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# Initialize global variables with proper error handling

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="deepseek-r1-distill-llama-70b"
)

model_name = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
recall_vector_store = InMemoryVectorStore(embedding_model)

class State(MessagesState):
    # add memories that will be retrieved based on the conversation context
    recall_memories: List[str]

def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id

@tool 
def web_agent(web_link:str,config: RunnableConfig) -> None:
    """
    Process a webpage by loading its content, splitting it into chunks, and storing the embeddings in FAISS.
    
    Args:
        web_link (str): The URL of the webpage to process. Must start with 'http://' or 'https://'.
        config (RunnableConfig): Configuration for the runnable.
        
    Returns:
        None: This function returns None and prints status messages.
    """
    print("called web_agent")
    if web_link.startswith("http://") or web_link.startswith("https://"):
        # Load and process webpage
        loader = WebBaseLoader(web_link)
        page_data = loader.load()
        
        # Split content into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        documents = text_splitter.split_documents(page_data)
        if not documents:
            print(" No content extracted from the webpage. Try another URL.")
        else:
            # Store embeddings in FAISS
            vectorstore = FAISS.from_documents(documents, embedding_model)
            # Store vectorstore in session for further querying
            cl.user_session.set("vectorstore", vectorstore)
            
            print("✅ Webpage processed successfully! You can now ask questions.")
    else:
        print("No Web_link")
    return "Webpage processed successfully! You can now ask questions."
@tool
def Pdf_agent(pdf_path: str, config: RunnableConfig) -> None:
    """
    Process a pdf by loading its content, splitting it into chunks, and storing the embeddings in FAISS and deleting the pdf file.
    
    Args:
        pdf_path (str): The path of the pdf to process. Must end with '.pdf'.
        config (RunnableConfig): Configuration for the runnable.
        
    Returns:
        None: This function returns None and prints status messages.
    """
    # if element.mime and "pdf" in element.mime.lower():
    print("called Pdf_agent and pdf path-",pdf_path)
    if pdf_path.endswith(".pdf"):
        loader=PyPDFLoader(pdf_path)
        docs=loader.load()
        # await cl.Message(content=f"{str(docs[0].page_content[:1000])}").send()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        documents=text_splitter.split_documents(docs)
        if not documents:
            print("⚠️ No content extracted from the pdf. Try another URL.")
        else:
            # Store embeddings in FAISS
            vectorstore = FAISS.from_documents(documents, embedding_model)
            # Store vectorstore in session for further querying
            cl.user_session.set("vectorstore", vectorstore)
            
            print("✅ pdf processed successfully! You can now ask questions.")
        try:
            os.remove(pdf_path)
            print(f"Deleted file: {pdf_path}")  # Debugging
        except Exception as e:
            print(f"Error deleting file {pdf_path}: {e}")

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    print("called save_recall_memory")
    user_id = get_user_id(config)
    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
    )
    recall_vector_store.add_documents([document])
    return memory

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    print("called search_recall_memories")
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = recall_vector_store.similarity_search(
        query, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]

# Define the prompt template for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant with advanced long-term memory and agents tool"
            " capabilities. Powered by a stateless LLM, you must rely on"
            " external memory to store information between conversations."
            " Utilize the available memory and agents tools to store and retrieve"
            " important details that will help you better attend to the user's"
            " needs and understand their context.\n\n"
            ## 
            ## Agent Prompt
            " You have three tools at your disposal: `memory`, `recall`, `web_agent`, `pdf_agent`."
            " to help you store and retrieve information. You can use these tools.\n"
            " When user ask question that vary over time then please web agent for getting latest information."
            " if possible use some good websites like Wikipedia, Google, etc. for getting latest information.\n\n"
            ## Rag Prompt
            # " You are designed to help user in finding the best possible answer from given information."
            # " You can ask user for clarification if you are unsure about"
            " Use the following context while answering the user's question:\n\n"
            " Context :\n {context}\n\n"
            ##########
            "Memory Usage Guidelines:\n"
            "1. Actively use memory tools (save_core_memory, save_recall_memory)"
            " to build a comprehensive understanding of the user.\n"
            "2. Make informed suppositions and extrapolations based on stored"
            " memories.\n"
            "3. Regularly reflect on past interactions to identify patterns and"
            " preferences.\n"
            "4. Update your mental model of the user with each new piece of"
            " information.\n"
            "5. Cross-reference new information with existing memories for"
            " consistency.\n"
            "6. Prioritize storing emotional context and personal values"
            " alongside facts.\n"
            "7. Use memory to anticipate needs and tailor responses to the"
            " user's style.\n"
            "8. Recognize and acknowledge changes in the user's situation or"
            " perspectives over time.\n"
            "9. Leverage memories to provide personalized examples and"
            " analogies.\n"
            "10. Recall past challenges or successes to inform current"
            " problem-solving.\n\n"
            "## Recall Memories\n"
            "Recall memories are contextually retrieved based on the current"
            " conversation:\n{recall_memories}\n\n"
            "## Instructions\n"
            "Engage with the user naturally, as a trusted colleague or friend."
            " There's no need to explicitly mention your memory capabilities."
            " Instead, seamlessly incorporate your understanding of the user"
            " into your responses. Be attentive to subtle cues and underlying"
            " emotions. Adapt your communication style to match the user's"
            " preferences and current emotional state. Use tools to persist"
            " information you want to retain in the next conversation. If you"
            " do call tools, all text preceding the tool call is an internal"
            " message. Respond AFTER calling the tool, once you have"
            " confirmation that the tool completed successfully.\n\n",
        ),
        ("placeholder", "{messages}"),
    ]
)
tools = [save_recall_memory, search_recall_memories, web_agent, Pdf_agent]
model_with_tools = llm.bind_tools(tools)
def agent(state: State) -> State:
    """Process the current state and generate a response using the LLM.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        schemas.State: The updated state with the agent's response.
    """
    bound = prompt | model_with_tools
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    # Load the vector store
    print("content in agent-",state["messages"])
    # vectorstore = Chroma(persist_directory="chroma-BAAI", embedding_function=embedding_model)
    vectorstore = cl.user_session.get("vectorstore",None)
    if vectorstore is not None:
        res = vectorstore.similarity_search(state["messages"][-1].content, k=3) 
        context = " ".join([doc.page_content for doc in res])
    else:
        context = "User have not specified any context yet please go ahead with knowledege and give answer"
    # context = "I am dheeraj"

    prediction = bound.invoke(
        {
            "messages": state["messages"],
            "context": context,
            "recall_memories": recall_str
            
        }
    )
    return {
        "messages": [prediction],
    }

def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation.

    Args:
        state (schemas.State): The current state of the conversation.
        config (RunnableConfig): The runtime configuration for the agent.

    Returns:
        State: The updated state with loaded memories.
    """
    convo_str = get_buffer_string(state["messages"])
    convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
    recall_memories = search_recall_memories.invoke(convo_str, config)
    return {
        "recall_memories": recall_memories,
    }

def route_tools(state: State):
    """Determine whether to use tools or end the conversation based on the last message.

    Args:
        state (schemas.State): The current state of the conversation.

    Returns:
        Literal["tools", "__end__"]: The next step in the graph.
    """
    msg = state["messages"][-1]
    # print("message-------",state["messages"][-1])
    print("called route_tools-",msg.tool_calls)
    if msg.tool_calls:
        return "tools"

    return END

# Create the graph and add nodes
builder = StateGraph(State)
builder.add_node(load_memories)
builder.add_node(agent)

builder.add_node("tools", ToolNode(tools))

# Add edges to the graph
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "agent")
builder.add_conditional_edges("agent", route_tools, ["tools", END])
builder.add_edge("tools", "agent")

# Compile the graph
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"user_id": "1", "thread_id": "1"}}



@cl.on_message
async def main(msg: cl.Message):
    try: 
        # await cl.Message(content=f"Unsupported file type: {msg.elements}").send()
        # file_path = ""
        if msg.elements:
            file_path = msg.elements[0].path
            msg.content = msg.content + "pdf_file-" + file_path
            # for element in msg.elements:
            #     print(f"Received file: {element.name}, Type: {element.mime}, Path: {element.path}")  # Debugging
            #     # if element.mime and "pdf" in element.mime.lower():
            #     loader=PyPDFLoader(element.path)
            #     docs=loader.load()
            #     # await cl.Message(content=f"{str(docs[0].page_content[:1000])}").send()
            #     text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
            #     documents=text_splitter.split_documents(docs)
            #     if not documents:
            #         await cl.Message(content="⚠️ No content extracted from the pdf. Try another URL.").send()
            #     else:
            #         # Store embeddings in FAISS
            #         vectorstore = FAISS.from_documents(documents, embedding_model)
            #         # Store vectorstore in session for further querying
            #         cl.user_session.set("vectorstore", vectorstore)
                    
            #         await cl.Message(content="✅ pdf processed successfully! You can now ask questions.").send()
            #     try:
            #         # os.remove(element.path)
            #         print(f"Deleted file: {element.path}")  # Debugging
            #     except Exception as e:
            #         print(f"Error deleting file {element.path}: {e}")
        # print(msg.elements[0].path)
        response = list(graph.stream({"messages": [HumanMessage(content=msg.content)]}, config=config))
        
        answer = cl.Message(content=response[-1]['agent']['messages'][0].content)
        await answer.send()
        
    except Exception as e:
        # Handle errors that might occur during processing
        error_msg = cl.Message(content=f"An error occurred: {str(e)}")
        await error_msg.send()