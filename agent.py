import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Fake Data
LEASES = {
    "123_main": {
        "address": "123 Main St, Amherst MA",
        "tenant": "CVS Pharmacy",
        "noi": 280000,
        "base_rent": 265000,
        "escalations": "3% annually",
        "lease_term_remaining_years": 8,
        "ti_allowance": 0,
    },
    "456_broadway": {
        "address": "456 Broadway, Northampton MA",
        "tenant": "Local restaurant group",
        "noi": 145000,
        "base_rent": 140000,
        "escalations": "2% annually",
        "lease_term_remaining_years": 4,
        "ti_allowance": 50000,
    },
}

COMPS = [
    {"address": "110 Main St", "sale_price": 3900000, "cap_rate": 7.1, "sqft": 8500},
    {"address": "145 Main St", "sale_price": 4500000, "cap_rate": 7.3, "sqft": 9200},
    {"address": "201 Pleasant St", "sale_price": 3750000, "cap_rate": 7.2, "sqft": 8100},
    {"address": "88 North Pleasant", "sale_price": 4100000, "cap_rate": 7.0, "sqft": 8800},
    {"address": "22 University Dr", "sale_price": 4250000, "cap_rate": 7.4, "sqft": 9000},
]

# Tools

@tool
def get_lease_details(property_id: str) -> dict:
    """Look up lease terms for a property by ID. Returns NOI, rent, escalations, etc."""
    return LEASES.get(property_id, {"error": f"Property {property_id} not found"})

@tool
def get_comps(address: str, radius_miles: float = 1.0) -> list:
    """Get comparable sales within a radius of the given address."""
    return COMPS

@tool
def calculate_metrics(noi: float, asking_price: float) -> dict:
    """Compute deal metrics: cap rate, price per unit of NOI."""
    cap_rate = (noi / asking_price) * 100
    return {
        "cap_rate_percent": round(cap_rate, 2),
        "price_per_dollar_noi": round(asking_price / noi, 2),
        "asking_price": asking_price,
        "noi": noi,
    }

tools = [get_lease_details, get_comps, calculate_metrics]


llm = ChatOpenAI(
    model="deepseek-chat",  # or "deepseek-reasoner"
    temperature=0,
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1",
)