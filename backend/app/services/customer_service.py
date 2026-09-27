from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.customer import Customer
from app.schemas.customer import CustomerCreate, CustomerUpdate


def get_customers(db: Session, page: int = 1, page_size: int = 20):
    total = db.query(func.count(Customer.id)).scalar()
    customers = (
        db.query(Customer)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return customers, total


def get_customer_by_customer_id(db: Session, customer_id: str):
    return db.query(Customer).filter(Customer.customer_id == customer_id).first()


def get_customer_by_id(db: Session, customer_id: int):
    return db.query(Customer).filter(Customer.id == customer_id).first()


def create_customer(db: Session, customer: CustomerCreate):
    db_customer = Customer(**customer.model_dump())
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    return db_customer


def update_customer(db: Session, customer_id: int, customer: CustomerUpdate):
    db_customer = get_customer_by_id(db, customer_id)
    if not db_customer:
        return None

    update_data = customer.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_customer, field, value)

    db.commit()
    db.refresh(db_customer)
    return db_customer


def create_bulk_customers(db: Session, customers: list[CustomerCreate]):
    created = []
    for customer in customers:
        existing = get_customer_by_customer_id(db, customer.customer_id)
        if existing:
            update_data = customer.model_dump()
            for field, value in update_data.items():
                setattr(existing, field, value)
            db.flush()
            created.append(existing)
        else:
            db_customer = Customer(**customer.model_dump())
            db.add(db_customer)
            db.flush()
            created.append(db_customer)

    db.commit()
    for c in created:
        db.refresh(c)

    return created
