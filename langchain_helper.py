from langchain_google_genai import GoogleGenerativeAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX
from langchain.prompts.prompt import PromptTemplate

from few_shots import few_shots

import os
from dotenv import load_dotenv
import re

load_dotenv()  # Load environment variables


def get_few_shot_db_chain():
    db_user = "root"
    db_password = "root"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    # Ensure the database connection is correct
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
        sample_rows_in_table_info=3
    )

    llm = GoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.1
    )

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    
    mysql_prompt = """
    You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, 
    then look at the results of the query and return the answer to the input question.
    Unless the user specifies a specific number of examples, query for at most {top_k} results using LIMIT.
    Never query for all columns from a table—only query necessary columns. 
    Use CURDATE() for questions involving "today".
    Ensure the SQL query does not contain markdown-style formatting (e.g., ```sql ... ```).
    Use this format:

    Question: Question here
    SQLQuery: Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer"],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],
    )
    
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    
    return chain


# ✅ Execute the SQL Query and Extract the Actual Result
def get_tshirt_stock():
    chain = get_few_shot_db_chain()
    
    # Generate and execute the query
    response = chain.run("how many T-shirts are left?")  # Execute the query

    # Extract the numeric result using regex
    if isinstance(response, str):
        match = re.search(r"SQLResult:\s*(\d+)", response)
        if match:
            return f"Total T-shirts left: {match.group(1)}"
    
    return "Could not fetch the stock count."


# ✅ Run the function
print(get_tshirt_stock())
