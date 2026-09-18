from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class CustomerBase(BaseModel):
    customer_id: str = Field(..., max_length=50)
    gender: str = Field(..., max_length=10)
    senior_citizen: bool
    partner: str = Field(..., max_length=5)
    dependents: str = Field(..., max_length=5)
    tenure: int = Field(..., ge=0, le=72)
    phone_service: str = Field(..., max_length=5)
    multiple_lines: Optional[str] = Field(None, max_length=20)
    internet_service: str = Field(..., max_length=20)
    online_security: Optional[str] = Field(None, max_length=20)
    online_backup: Optional[str] = Field(None, max_length=20)
    device_protection: Optional[str] = Field(None, max_length=20)
    tech_support: Optional[str] = Field(None, max_length=20)
    streaming_tv: Optional[str] = Field(None, max_length=20)
    streaming_movies: Optional[str] = Field(None, max_length=20)
    contract: str = Field(..., max_length=20)
    paperless_billing: str = Field(..., max_length=5)
    payment_method: str = Field(..., max_length=30)
    monthly_charges: float = Field(..., ge=0)
    total_charges: Optional[float] = Field(None, ge=0)


class CustomerCreate(CustomerBase):
    pass


class CustomerUpdate(BaseModel):
    gender: Optional[str] = Field(None, max_length=10)
    senior_citizen: Optional[bool] = None
    partner: Optional[str] = Field(None, max_length=5)
    dependents: Optional[str] = Field(None, max_length=5)
    tenure: Optional[int] = Field(None, ge=0, le=72)
    phone_service: Optional[str] = Field(None, max_length=5)
    multiple_lines: Optional[str] = Field(None, max_length=20)
    internet_service: Optional[str] = Field(None, max_length=20)
    online_security: Optional[str] = Field(None, max_length=20)
    online_backup: Optional[str] = Field(None, max_length=20)
    device_protection: Optional[str] = Field(None, max_length=20)
    tech_support: Optional[str] = Field(None, max_length=20)
    streaming_tv: Optional[str] = Field(None, max_length=20)
    streaming_movies: Optional[str] = Field(None, max_length=20)
    contract: Optional[str] = Field(None, max_length=20)
    paperless_billing: Optional[str] = Field(None, max_length=5)
    payment_method: Optional[str] = Field(None, max_length=30)
    monthly_charges: Optional[float] = Field(None, ge=0)
    total_charges: Optional[float] = Field(None, ge=0)


class CustomerResponse(CustomerBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CustomerListResponse(BaseModel):
    customers: list[CustomerResponse]
    total: int
    page: int
    page_size: int
