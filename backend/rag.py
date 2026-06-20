"""
rag.py — RAG pipeline for BizBot
Loads business data (products, orders, customers) from CSV files,
embeds them using Gemini, stores/retrieves via ChromaDB.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables (GEMINI_API_KEY)
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
ORDERS_CSV = os.path.join(DATA_DIR, "orders.csv")
CUSTOMERS_CSV = os.path.join(DATA_DIR, "customers.csv")

COLLECTION_NAME = "bizbot_data"


def build_documents_from_products(df: pd.DataFrame) -> list[Document]:
    """Convert each product row into a Document."""
    docs = []
    for _, row in df.iterrows():
        text = (
            f"Product: {row['product_name']} | "
            f"Category: {row['category']} | "
            f"Price: ₹{row['price']} per {row['unit']} | "
            f"Stock: {row['stock_quantity']} {row['unit']} available"
        )
        metadata = {
            "source": "products",
            "product_id": str(row["product_id"]),
            "product_name": str(row["product_name"]),
            "category": str(row["category"]),
            "price": float(row["price"]),
            "stock_quantity": int(row["stock_quantity"]),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def build_documents_from_orders(df: pd.DataFrame) -> list[Document]:
    """Convert each order row into a Document."""
    docs = []
    for _, row in df.iterrows():
        text = (
            f"Order #{row['order_id']}: {row['customer_name']} "
            f"({row['customer_phone']}) ordered {row['quantity']} x "
            f"{row['product_name']} for ₹{row['total_price']}, "
            f"status: {row['status']}, date: {row['order_date']}"
        )
        metadata = {
            "source": "orders",
            "order_id": str(row["order_id"]),
            "customer_name": str(row["customer_name"]),
            "customer_phone": str(row["customer_phone"]),
            "product_name": str(row["product_name"]),
            "status": str(row["status"]),
            "order_date": str(row["order_date"]),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def build_documents_from_customers(df: pd.DataFrame) -> list[Document]:
    """Convert each customer row into a Document."""
    docs = []
    for _, row in df.iterrows():
        text = (
            f"Customer: {row['customer_name']} ({row['customer_phone']}), "
            f"balance due: ₹{row['balance_due']}, "
            f"last purchase: {row['last_purchase_date']}"
        )
        metadata = {
            "source": "customers",
            "customer_id": str(row["customer_id"]),
            "customer_name": str(row["customer_name"]),
            "customer_phone": str(row["customer_phone"]),
            "balance_due": float(row["balance_due"]),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def load_all_documents() -> list[Document]:
    """Load all three CSVs and convert to a single list of Documents."""
    products_df = pd.read_csv(PRODUCTS_CSV)
    orders_df = pd.read_csv(ORDERS_CSV)
    customers_df = pd.read_csv(CUSTOMERS_CSV)

    docs = []
    docs.extend(build_documents_from_products(products_df))
    docs.extend(build_documents_from_orders(orders_df))
    docs.extend(build_documents_from_customers(customers_df))
    return docs


def get_embeddings():
    """Return the Gemini embeddings model."""
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_vectorstore() -> Chroma:
    """
    Load existing ChromaDB if present, otherwise build it fresh
    from the CSVs and persist it to disk.
    """
    embeddings = get_embeddings()

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        # Reuse existing persisted vectorstore
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    else:
        # Build fresh
        documents = load_all_documents()
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
        )
    return vectorstore


def get_retriever(k: int = 4):
    """Return a retriever that fetches top-k relevant documents."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are BizBot, a helpful assistant for a local Indian kirana store.
Answer the customer's question using ONLY the context below.
If the answer isn't in the context, say you don't have that information
and suggest they contact the store directly.
Be concise and friendly. Use ₹ for prices.

Context:
{context}

Customer question: {question}

Answer:"""
)


def get_llm():
    """Return the Gemini chat model."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)


def answer_query(question: str, k: int = 4) -> str:
    """
    Full RAG pipeline: retrieve relevant docs, then ask Gemini
    to answer the question using that context.
    """
    retriever = get_retriever(k=k)
    docs = retriever.invoke(question)

    context = "\n".join(doc.page_content for doc in docs)

    llm = get_llm()
    chain = RAG_PROMPT | llm

    response = chain.invoke({"context": context, "question": question})
    return response.content


def rebuild_vectorstore():
    """Force a rebuild of the vectorstore from CSVs (call this when data changes)."""
    import shutil
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    return get_vectorstore()


if __name__ == "__main__":
    # Quick manual test: run `python rag.py` from inside backend/
    test_query = "Do you have Parle-G biscuits?"
    answer = answer_query(test_query)
    print(f"\nQuery: {test_query}\n")
    print(f"Answer: {answer}\n")