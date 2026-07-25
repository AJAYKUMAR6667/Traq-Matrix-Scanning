import os
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from llama_cloud import LlamaCloud

app = FastAPI(title="Vehicle, Expense & Document Extraction Service Fast")

# Initialize the official client SDK
LLAMA_CLOUD_API_KEY = os.getenv(
    "LLAMA_CLOUD_API_KEY", 
    "llx-KGq9lskPn7hyyxd7UZcbePLsAbjMZ2WcS3HsSjlc93LKinp3"
)
client = LlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

# Optimized polling configuration for fast execution
MAX_POLL_SECONDS = 30
POLL_INTERVAL_SECONDS = 0.5  # Check every 0.5s instead of 2.0s for speed

# ==============================================================================
# SECTION 1: UPDATED DATA SCHEMAS
# ==============================================================================

class VehicleDocumentSchema(BaseModel):
    """Schema for processing core vehicle logs and certificates."""
    vehicle_number: str = Field(description="Vehicle registration number like TN01AB1234")
    document_type: str = Field(description="Document type. Return only one of these values: RC, Insurance, Permit, Fitness Certificate, Pollution Certificate, Road Tax")
    issue_date: str = Field(description="Issue date printed on the document")
    expiry_date: str = Field(description="Expiry date or Valid Upto date printed on the document")
    
    model_config = ConfigDict(populate_by_name=True)

class ExpensesSchema(BaseModel):
    """Schema for processing parts, fuel, maintenance, and structural card totals."""
    vehicle_number: str = Field(description="Vehicle registration number like TN01AB1234 or AP08AS5506")
    expense_date: str = Field(description="Expense statement or claim date from the document")
    expense_category: str = Field(description="Look at the 'Particulars' table grid. Extract all individual listed item names across all rows (e.g., Tyre, Mirrors, Steering Wheel, Oil, Diesel, Parking) and combine them together into a single text string separated by commas.")
    cost: float = Field(description="The grand Total Amount value listed at the bottom of the statement card only.")
    
    model_config = ConfigDict(populate_by_name=True)

class IdentityDocumentSchema(BaseModel):
    """Schema for processing personal verification documents."""
    Aadhar_No1: str = Field(description="12-digit unique identification string parsed from the primary asset layout framework")
    Name: str = Field(description="Full Name as printed on the identity card")
    DOB: str = Field(description="Date of Birth in DD/MM/YYYY format")
    Gender: str = Field(description="Gender (Male, Female, or Transgender)")
    
    model_config = ConfigDict(populate_by_name=True)

SCHEMA_MAP = {
    "vehicle_document": VehicleDocumentSchema,
    "expenses": ExpensesSchema,
    "aadhar": IdentityDocumentSchema,
}

# ==============================================================================
# SECTION 2: UTILITIES & ROUTING
# ==============================================================================

def _resolve_media_type(filename: str) -> str:
    filename_lower = filename.lower()
    if filename_lower.endswith(".pdf"):
        return "application/pdf"
    elif filename_lower.endswith(".png"):
        return "image/png"
    elif filename_lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    raise HTTPException(
        status_code=400,
        detail="Unsupported format type. Send PDF, PNG, or JPEG image assets.",
    )

# ==============================================================================
# SECTION 3: ENDPOINTS
# ==============================================================================

@app.post("/extract")
async def extract_document(
    file: UploadFile = File(...),
    doc_type: str = Query(..., description="Schema selection: vehicle_document, expenses, aadhar")
):
    """
    Submits extraction to LlamaIndex production and runs a sub-second optimized polling engine.
    """
    schema_cls = SCHEMA_MAP.get(doc_type)
    if schema_cls is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid doc_type routing identifier. Choose from: {list(SCHEMA_MAP.keys())}",
        )

    media_type = _resolve_media_type(file.filename)
    file_bytes = await file.read()

    try:
        # 1. Safely upload the binary content via the native SDK
        uploaded_file = await run_in_threadpool(
            client.files.create,
            file=(file.filename, file_bytes, media_type),
            purpose="extract",
        )

        # 2. Spawn the job using the v2 agentic framework configuration options
        job = await run_in_threadpool(
            client.extract.create,
            file_input=uploaded_file.id,
            configuration={
                "data_schema": schema_cls.model_json_schema(),
                "tier": "agentic",
            },
        )

        # 3. Fast high-frequency polling loop targeting the correct API infrastructure
        elapsed = 0.0
        while job.status not in ("COMPLETED", "FAILED", "CANCELLED"):
            if elapsed >= MAX_POLL_SECONDS:
                raise HTTPException(
                    status_code=504,
                    detail=f"Extraction job {job.id} timed out after {MAX_POLL_SECONDS}s."
                )
            
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            job = await run_in_threadpool(client.extract.get, job.id)
            elapsed += POLL_INTERVAL_SECONDS

        if job.status == "COMPLETED":
            return job.extract_result
        else:
            raise HTTPException(status_code=500, detail=f"LlamaCloud extraction pipeline failed with status: {job.status}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction runtime exception: {str(e)}")