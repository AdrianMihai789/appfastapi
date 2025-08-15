# main.py
from __future__ import annotations

import datetime as dt
from typing import List, Optional
import re

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator

from sqlalchemy import Column, Date, Integer, String, Text, Index, select, case
from sqlalchemy.ext.asyncio import AsyncAttrs, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# -------------------------------------------------------------------------
# App & Security
# -------------------------------------------------------------------------

app = FastAPI(title="Contracts API (Async + Indexed)")
API_KEY = "secret-key"
bearer_scheme = HTTPBearer(auto_error=False)


async def authenticate(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    if not creds or creds.scheme.lower() != "bearer" or creds.credentials != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return creds.credentials

# -------------------------------------------------------------------------
# Database
# -------------------------------------------------------------------------
# DB_URL = "sqlite+aiosqlite:///C:/Users/Predoi/Documents/myprojectpython/contracts.db"
DB_URL = "sqlite+aiosqlite:///./contracts.db"
engine = create_async_engine(DB_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(AsyncAttrs, DeclarativeBase):
    pass


class Contract(Base):
    __tablename__ = "contracts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vendor: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    renewal_date: Mapped[Optional[dt.date]] = mapped_column(Date, index=True)
    details: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(100))

    __table_args__ = (
        Index("ix_contracts_value", "value"),
        Index("ix_contracts_renewal_date", "renewal_date"),
    )


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()


async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

# -------------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------------

class ContractIn(BaseModel):
    vendor: str = Field(..., min_length=1, max_length=255)
    value: int = Field(..., ge=0)
    renewal_date: Optional[dt.date] = None
    details: Optional[str] = None
    category: Optional[str] = None

    @validator("renewal_date", pre=True)
    def parse_date(cls, v):
        if v in (None, "", "null"):
            return None
        if isinstance(v, dt.date):
            return v
        return dt.date.fromisoformat(v)


class ContractOut(BaseModel):
    id: int
    vendor: str
    value: int
    renewal_date: Optional[dt.date] = None
    details: Optional[str] = None
    category: Optional[str] = None
    insight: Optional[str] = None


class SaveResponse(BaseModel):
    message: str
    saved: int

# -------------------------------------------------------------------------
# Insight logic
# -------------------------------------------------------------------------

def generate_insight(value: int, details: Optional[str]) -> Optional[str]:
    if not details:
        details_l = ""
    else:
        details_l = details.lower()

    if value > 1000 and "high" in details_l:
        return "High cost—suggest review and negotiation"
    if "privacy" in details_l:
        return "Privacy terms noted—check compliance"
    if "negotiation" in details_l:
        return "Negotiation mentioned—consider re-pricing"
    if "renewal" in details_l:
        return "Upcoming renewal—plan discussion"
    return None

# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------

@app.post("/contracts", response_model=SaveResponse)
async def save_contracts(
    contracts: List[ContractIn],
    _apikey: str = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
):
    models = [
        Contract(
            vendor=c.vendor,
            value=c.value,
            renewal_date=c.renewal_date,
            details=c.details,
            category=c.category,
        )
        for c in contracts
    ]
    session.add_all(models)
    await session.commit()
    return SaveResponse(message="Data received and saved", saved=len(models))


@app.get("/contracts/high-value", response_model=List[ContractOut])
async def get_high_value_contracts(
    threshold: int = 1000,
    _apikey: str = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
):
    """
    Returns contracts where value > threshold, sorted by renewal_date ascending,
    with NULLs last. Generates basic insight.
    """
    # corectare func.case → sqlalchemy.case
    order_expr = (
        case((Contract.renewal_date.is_(None), 1), else_=0),
        Contract.renewal_date.asc()
    )

    stmt = select(Contract).where(Contract.value > threshold).order_by(*order_expr)
    result = await session.execute(stmt)
    contracts: List[Contract] = result.scalars().all()

    return [
        ContractOut(
            id=c.id,
            vendor=c.vendor,
            value=c.value,
            renewal_date=c.renewal_date,
            details=c.details,
            category=c.category,
            insight=generate_insight(c.value, c.details),
        )
        for c in contracts
    ]


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/contracts/all", response_model=List[ContractOut])
async def get_all_contracts(
    _apikey: str = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
):
    """
    Returns all contracts from the database with insights.
    """
    stmt = select(Contract).order_by(Contract.id.asc())
    result = await session.execute(stmt)
    contracts: List[Contract] = result.scalars().all()

    return [
        ContractOut(
            id=c.id,
            vendor=c.vendor,
            value=c.value,
            renewal_date=c.renewal_date,
            details=c.details,
            category=c.category,
            insight=generate_insight(c.value, c.details),
        )
        for c in contracts
    ]

# -------------------------------------------------------------------------
# Developer seed
# -------------------------------------------------------------------------

SEED_DATA = [
    {"vendor": "ExampleCo", "value": 1200, "renewal_date": "2026-01-01",
     "details": "Standard software license with high renewal cost and privacy clauses.", "category": "Software"},
    {"vendor": "TechCorp", "value": 750, "renewal_date": "2025-12-15",
     "details": "Cloud service bundle, potential for negotiation on terms.", "category": "Cloud"},
    {"vendor": "ServiceLtd", "value": 2500, "renewal_date": "2026-03-20",
     "details": "Enterprise plan with data privacy terms and high scalability needs.", "category": "Enterprise"},
    {"vendor": "GlobalInc", "value": 900, "renewal_date": "2025-11-05",
     "details": "Hardware supply agreement requiring urgent renewal discussion.", "category": "Hardware"},
    {"vendor": "InnoTech", "value": 1800, "renewal_date": "2026-05-10",
     "details": "AI tool subscription with high performance but negotiation possible on pricing.", "category": "AI"},
    {"vendor": "SecureNet", "value": 400, "renewal_date": "2025-10-30",
     "details": "Network security service with standard terms, no major issues.", "category": "Security"},
]


@app.post("/dev/seed", response_model=SaveResponse)
async def dev_seed(
    _apikey: str = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
):
    models = [
        Contract(
            vendor=item["vendor"],
            value=item["value"],
            renewal_date=dt.date.fromisoformat(item["renewal_date"]),
            details=item["details"],
            category=item["category"]
        )
        for item in SEED_DATA
    ]
    session.add_all(models)
    await session.commit()
    return SaveResponse(message="Seeded", saved=len(models))


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
db_path = re.sub(r'^sqlite\+aiosqlite:///', '', DB_URL)
print(db_path)  # va afișa exact calea folosită de engine