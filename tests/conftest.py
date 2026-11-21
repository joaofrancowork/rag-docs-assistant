from __future__ import annotations

import pytest
from fpdf import FPDF


def make_pdf(pages: list[str]) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


@pytest.fixture
def two_page_pdf() -> bytes:
    return make_pdf(
        [
            "Refund policy. Customers can request a full refund within 30 days of purchase.\n\n"
            "Refunds are processed to the original payment method within 5 business days.",
            "Support hours. The support team is available Monday to Friday, 9:00 to 18:00 CET.\n\n"
            "Enterprise customers have a dedicated account manager.",
        ]
    )
