from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, Numeric, DateTime, ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship
from app.database import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(50), unique=True, nullable=False, index=True)

    # Datos demográficos
    gender = Column(String(10), nullable=False)
    senior_citizen = Column(Boolean, nullable=False)
    partner = Column(String(5), nullable=False)
    dependents = Column(String(5), nullable=False)

    # Servicios contratados
    tenure = Column(Integer, nullable=False)
    phone_service = Column(String(5), nullable=False)
    multiple_lines = Column(String(20))
    internet_service = Column(String(20), nullable=False)
    online_security = Column(String(20))
    online_backup = Column(String(20))
    device_protection = Column(String(20))
    tech_support = Column(String(20))
    streaming_tv = Column(String(20))
    streaming_movies = Column(String(20))

    # Contrato y facturación
    contract = Column(String(20), nullable=False)
    paperless_billing = Column(String(5), nullable=False)
    payment_method = Column(String(30), nullable=False)
    monthly_charges = Column(Numeric(10, 2), nullable=False)
    total_charges = Column(Numeric(10, 2))

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    predictions = relationship("Prediction", back_populates="customer")
