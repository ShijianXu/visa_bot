"""Data models shared across the application."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class VisaDocument:
    """A retrieved and stored visa-related document."""

    content: str
    source_url: str
    origin_country: str
    destination_country: str
    page_title: str = ""
    visa_type: str = ""
    retrieval_time: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.source_url.encode()).hexdigest()

    def to_metadata(self) -> dict:
        return {
            "source_url": self.source_url,
            "origin_country": self.origin_country,
            "destination_country": self.destination_country,
            "page_title": self.page_title,
            "visa_type": self.visa_type,
            "retrieval_time": self.retrieval_time,
        }


@dataclass
class VisaQuery:
    """Collected travel information from the user."""

    nationality: str
    residence: str
    destination: str
    purpose: str
    departure_date: str
    duration_of_stay: str
    # Optional fields
    residence_permit: str = ""
    entry_type: str = "single"
    companions: str = ""

    def summary(self) -> str:
        return (
            f"{self.nationality} → {self.destination} | "
            f"{self.purpose} | {self.departure_date}"
        )
