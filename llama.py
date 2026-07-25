import time
from pydantic import BaseModel, Field
from typing import List
from llama_cloud import LlamaCloud

# 1. Initialize the client directly
client = LlamaCloud(api_key="llx-knaUlGzQqxYtuAe9FnOO2YrMrjP2GXvmVycN5dQOtTA49XMX")

# 2. Define the schema
class LineItem(BaseModel):
    description: str = Field(description="The name or description of the product or service")
    quantity: int = Field(description="The number of items purchased")
    unit_price: float = Field(description="The price per single unit")
    amount: float = Field(description="The total amount for this specific line item")

class InvoiceSchema(BaseModel):
    vendor_name: str = Field(description="The name of the company issuing the invoice")
    invoice_number: str = Field(description="The unique invoice ID number or reference")
    invoice_date: str = Field(description="The date the invoice was issued")
    line_items: List[LineItem] = Field(description="A list of all individual products or services listed")
    total_tax: float = Field(description="The total tax amount charged")
    grand_total: float = Field(description="The final total amount due on the invoice")

# 3. Upload the file
print("Uploading document...")
uploaded_file = client.files.create(
    file="test2.jpeg",  # Ensure this file exists in your working directory
    purpose="extract"
)

# 4. Trigger the extraction job with the correct parameters
print("Processing extraction job...")
job = client.extract.create(
    file_input=uploaded_file.id,
    configuration={
        "data_schema": InvoiceSchema.model_json_schema(),
        "tier": "agentic", 
    },
)

# 5. Poll the API until the server finishes parsing the document
print("Waiting for extraction to complete...")
while job.status not in ("COMPLETED", "FAILED", "CANCELLED"):
    time.sleep(2)
    job = client.extract.get(job.id)

# 6. Output the successful result
if job.status == "COMPLETED":
    print("\n--- Extracted JSON Data ---")
    print(job.extract_result)
else:
    print(f"\nExtraction failed with status: {job.status}")