from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.customer import (
    CustomerCreate,
    CustomerUpdate,
    CustomerResponse,
    CustomerListResponse,
)
from app.services.customer_service import (
    get_customers,
    get_customer_by_customer_id,
    get_customer_by_id,
    create_customer,
    update_customer,
    create_bulk_customers,
)

router = APIRouter(prefix="/api/v1/customers", tags=["customers"])


@router.get("", response_model=CustomerListResponse)
def list_customers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    customers, total = get_customers(db, page, page_size)
    return CustomerListResponse(
        customers=customers, total=total, page=page, page_size=page_size
    )


@router.get("/{customer_id}", response_model=CustomerResponse)
def get_customer(customer_id: str, db: Session = Depends(get_db)):
    customer = get_customer_by_customer_id(db, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@router.post("", response_model=CustomerResponse, status_code=201)
def create_customer_endpoint(customer: CustomerCreate, db: Session = Depends(get_db)):
    existing = get_customer_by_customer_id(db, customer.customer_id)
    if existing:
        raise HTTPException(
            status_code=409, detail="Customer with this ID already exists"
        )
    return create_customer(db, customer)


@router.put("/{customer_id}", response_model=CustomerResponse)
def update_customer_endpoint(
    customer_id: int, customer: CustomerUpdate, db: Session = Depends(get_db)
):
    updated = update_customer(db, customer_id, customer)
    if not updated:
        raise HTTPException(status_code=404, detail="Customer not found")
    return updated


@router.post("/bulk", response_model=list[CustomerResponse], status_code=201)
def bulk_create_customers(
    customers: list[CustomerCreate], db: Session = Depends(get_db)
):
    return create_bulk_customers(db, customers)
