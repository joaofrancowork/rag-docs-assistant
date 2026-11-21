"""Generate the sample corpus used by the demo and the evaluation.

The documents describe a fictional company, Northwind Labs, so the repository
ships no third-party or personal data. Run: uv run python scripts/make_sample_docs.py
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"

DOCS: dict[str, list[tuple[str, str]]] = {
    "northwind-employee-handbook.pdf": [
        (
            "Working at Northwind Labs",
            "Northwind Labs builds scheduling and messaging software for small clinics and "
            "service businesses. The company was founded in 2019 and has 64 employees across "
            "offices in Lisbon and Porto, plus remote staff in six countries.\n\n"
            "Core hours are 10:00 to 16:00 Lisbon time. Outside core hours, employees choose "
            "when to work, as long as their team agreements are respected. Every team publishes "
            "its agreements in the internal wiki and reviews them each quarter.",
        ),
        (
            "Time off and leave",
            "Full-time employees receive 25 days of paid annual leave per calendar year, plus "
            "public holidays of the country where they are based. Up to 5 unused days can be "
            "carried over to the following year and must be used by 31 March.\n\n"
            "Parental leave is 20 weeks at full pay for every parent, regardless of gender, and "
            "can be split into up to three blocks within the child's first two years. "
            "Sick leave does not require a medical certificate for absences of up to 3 days.",
        ),
        (
            "Remote work and equipment",
            "Employees may work remotely from any country where Northwind has a legal entity "
            "for up to 90 days per year without prior approval. Longer periods need approval "
            "from People Operations, because of tax and social security rules.\n\n"
            "Every new employee receives a laptop and a one-off home office budget of 900 euros "
            "for a desk, chair and monitor. The budget is claimed through the expenses tool with "
            "the original invoices, within the first six months.",
        ),
        (
            "Learning and growth",
            "Each employee has an annual learning budget of 1,200 euros for courses, books and "
            "conferences, plus 5 learning days per year. Requests up to 300 euros are approved "
            "by the employee's manager; larger requests also need approval from the head of "
            "department.\n\n"
            "Performance conversations happen twice a year, in April and October. Salary "
            "reviews happen once a year, in January, and are based on the published salary bands.",
        ),
        (
            "Expenses and travel",
            "Business travel must be booked through the travel desk at least 14 days in advance "
            "when possible. Economy class is the default for flights under 6 hours.\n\n"
            "Meal expenses during business travel are reimbursed up to 45 euros per day. "
            "Expenses must be submitted within 30 days of the purchase, with a receipt. "
            "Reimbursements are paid with the next monthly salary.",
        ),
    ],
    "northwind-product-guide.pdf": [
        (
            "Northwind Booking: overview",
            "Northwind Booking lets customers book appointments online, 24 hours a day, from a "
            "booking page, a website widget or a messaging assistant. Availability is calculated "
            "from every connected calendar, so double bookings are prevented.\n\n"
            "Supported calendar integrations are Google Calendar and Microsoft 365. Practice "
            "management systems are connected through the Northwind Connect API.",
        ),
        (
            "Plans and pricing",
            "The Starter plan costs 29 euros per month and includes 1 location, 3 staff "
            "calendars and email reminders. The Growth plan costs 79 euros per month and adds "
            "SMS reminders, up to 15 staff calendars and payments at booking.\n\n"
            "The Enterprise plan, plan code ENT-42, is priced per location and adds single "
            "sign-on, audit logs, a 99.95% uptime SLA and a dedicated account manager. Annual "
            "billing gives a discount of 2 months on every plan.",
        ),
        (
            "Reminders and no-shows",
            "Reminders are sent 24 hours before the appointment by default; the interval can be "
            "changed between 1 and 72 hours. Customers can confirm, reschedule or cancel from "
            "the reminder without logging in.\n\n"
            "Businesses that turned on SMS reminders reduced no-shows by 38% on average in the "
            "first three months. A cancellation window can be set so that late cancellations "
            "are flagged in the dashboard.",
        ),
        (
            "Refunds, billing and cancellation",
            "Subscriptions are billed monthly in advance. Customers can cancel at any time from "
            "the billing page; the plan stays active until the end of the paid period.\n\n"
            "New customers can request a full refund within 30 days of their first payment. "
            "After that, payments are non-refundable, except when Northwind misses the uptime "
            "SLA, in which case service credits apply. Refunds are issued to the original "
            "payment method within 10 business days.",
        ),
        (
            "Support",
            "Support is available by chat and email from Monday to Friday, 08:00 to 20:00 "
            "Lisbon time. Enterprise customers also have phone support and a 1-hour first "
            "response target for critical incidents.\n\n"
            "Status updates for incidents are published on status.northwind.example and "
            "sent to subscribed administrators.",
        ),
    ],
    "northwind-security-and-privacy.pdf": [
        (
            "Data protection",
            "Northwind processes personal data as a data processor on behalf of its customers. "
            "All customer data is hosted in the European Union, in data centres in Frankfurt "
            "and Dublin. Data is encrypted at rest with AES-256 and in transit with TLS 1.3.\n\n"
            "Customers can export all of their data at any time from the admin console in CSV "
            "and JSON formats.",
        ),
        (
            "Retention and deletion",
            "Appointment records are kept for as long as the customer's subscription is active. "
            "When a subscription ends, data is kept for 30 days to allow reactivation and is "
            "then permanently deleted within a further 60 days, including backups.\n\n"
            "Audit logs are retained for 13 months. Individual end users can ask the business "
            "to delete their data; the business can do this from the client record.",
        ),
        (
            "Access control and incidents",
            "Employees access production systems only through single sign-on with hardware "
            "security keys, and access is reviewed every quarter. Production access requires an "
            "approved ticket and is logged.\n\n"
            "Security incidents that affect personal data are notified to affected customers "
            "without undue delay and at the latest within 48 hours of confirmation. Northwind "
            "runs an external penetration test every year and holds ISO 27001 certification.",
        ),
    ],
}


def render(path: Path, pages: list[tuple[str, str]]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    for title, body in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 15)
        pdf.multi_cell(0, 9, title)
        pdf.ln(2)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, body)
    pdf.output(str(path))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCS.items():
        render(OUT / name, pages)
        print(f"wrote {OUT / name} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
