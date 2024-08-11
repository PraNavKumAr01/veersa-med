from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0.5, model_name="llama3-8b-8192")

query_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant specializing in medical information. 
    Your task is to provide accurate and helpful responses based on the patient's query.
    Always maintain a professional and empathetic tone. If you're unsure about any information, 
    state that clearly and suggest consulting a healthcare professional.

    Here is the patient's medical history. Use this information when needed to give better answers:
    {context}

    Here is the user query:
    Human: {question}

    Your response should only consist of the answer to the user query and nothing else.
    AI Assistant:
    """
)

query_chain = query_prompt | llm

class ConsultationRequest(BaseModel):
    diagnosis: list[str]
    symptoms: str

app = FastAPI()

@app.post("/generate-medical-response/")
async def generate_medical_response(consultation_request: ConsultationRequest):
    try:
        context = "\n".join(consultation_request.diagnosis)
        question = consultation_request.symptoms
        response = query_chain.invoke({
            "context": context,
            "question": question
        })

        return {"response": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))