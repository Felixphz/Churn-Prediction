from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from app.database import Base


class ApiLog(Base):
    __tablename__ = "api_logs"

    id = Column(Integer, primary_key=True, index=True)
    endpoint = Column(String(100), nullable=False)
    method = Column(String(10), nullable=False)
    customer_id = Column(Integer)
    response_status = Column(Integer)
    response_time_ms = Column(Integer)
    requested_at = Column(DateTime, default=datetime.utcnow)
