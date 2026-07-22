"""Generate realistic synthetic medical-bill and salvage-claim documents.

All content is fictional and patterned after the document surfaces an auto
insurer (e.g. American Family–style intake) would collect for casualty and
total-loss salvage workflows. No proprietary insurer data is used.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from src.storage.types import ClaimRecord, DocumentRecord, FieldRecord

# Carrier branding used only for synthetic AmFam-style intake simulation.
CARRIER_VARIANTS = (
    "American Family Insurance",
    "American Family",
    "AmFam",
    "American Family Mutual",
)

WI_CITIES = (
    ("Madison", "WI", "53703"),
    ("Madison", "WI", "53704"),
    ("Green Bay", "WI", "54301"),
    ("Eau Claire", "WI", "54701"),
    ("La Crosse", "WI", "54601"),
    ("Wausau", "WI", "54401"),
    ("Kenosha", "WI", "53140"),
    ("Milwaukee", "WI", "53202"),
    ("Appleton", "WI", "54911"),
    ("Racine", "WI", "53403"),
)

FIRST_NAMES = (
    "Jane", "John", "Alex", "Maria", "Robert", "Priya", "Chris", "Hannah",
    "Derek", "Alicia", "Sam", "Kelly", "Taylor", "Jordan", "Morgan", "Casey",
)
MIDDLE_INITIALS = ("A", "B", "J", "L", "M", "Q", "R", "S", "T", "")
LAST_NAMES = (
    "Public", "Smith", "Santos", "Ellis", "Nguyen", "Rivera", "Desai",
    "Brooks", "Olson", "Moore", "Ortiz", "White", "Keller", "Hansen",
    "Patel", "Wagner",
)

STREETS = (
    "Oak Avenue", "Lake Street", "Willow Rd", "Pine Court", "Capitol Dr",
    "River Bend", "Farm Road", "Hilltop Lane", "Maple Ave", "State St",
    "Washington Blvd", "Park Place",
)

PROVIDERS = (
    ("Lakeside Clinic", "physician_office", "1234567890"),
    ("University Hospital", "hospital", "1098765432"),
    ("Capitol Urgent Care", "urgent_care", "1122334455"),
    ("Midwest Orthopedics", "physician_office", "1567890123"),
    ("Community Medical Center", "hospital", "1987654321"),
    ("Northside Family Practice", "physician_office", "1456789012"),
)

LIENHOLDERS = (
    ("First National Bank Lienholder Services", "100 Finance Way, Chicago IL 60601"),
    ("Midwest Credit Union Lienholder Desk", "450 Credit Union Dr, Milwaukee WI 53202"),
    ("Capital Auto Finance", "88 Auto Finance Blvd, Minneapolis MN 55401"),
    ("Heartland Bank Title Department", "12 Commerce Plaza, Madison WI 53703"),
    ("Premier Lending Lien Payoff Unit", "900 Premier Pkwy, Des Moines IA 50309"),
)

SALVAGE_BUYERS = (
    ("Midwest Auto Salvage LLC", "2200 Salvage Rd, Rockford IL 61101"),
    ("Great Lakes Auction Yard", "15 Auction Lane, Kenosha WI 53140"),
    ("Northstar Auto Recyclers", "77 Recycle Way, Duluth MN 55802"),
    ("Badger Salvage Partners", "401 Industrial Dr, Janesville WI 53545"),
)

TOW_COMPANIES = (
    "Capitol Towing",
    "Badger Roadside Assist",
    "Lakeshore Recovery",
    "Interstate Tow & Recovery",
)

VEHICLES = (
    ("1HGCM82633A004352", "2018", "Honda", "Accord"),
    ("2T1BURHE0JC123456", "2017", "Toyota", "Corolla"),
    ("5YJSA1E26HF000111", "2019", "Tesla", "Model 3"),
    ("1FADP3F20EL123456", "2015", "Ford", "Focus"),
    ("3VWDP7AJ5DM123789", "2013", "Volkswagen", "Jetta"),
    ("1G1ZD5ST1JF012345", "2018", "Chevrolet", "Malibu"),
    ("5NPE24AF5FH123456", "2015", "Hyundai", "Sonata"),
    ("KM8J3CA46KU123456", "2019", "Hyundai", "Tucson"),
    ("1C4RJFBG5EC123456", "2014", "Jeep", "Grand Cherokee"),
    ("WBA8E9G50JNU12345", "2018", "BMW", "330i"),
)

DX_CODES = ("S13.4XXA", "M54.5", "S06.0X0A", "S82.201A", "V43.52XA", "R51.9")
CPT_CODES = ("99213", "99214", "99284", "70450", "73030", "97110", "12001")
REV_CODES = (
    ("0450", "Emergency Room"),
    ("0300", "Laboratory"),
    ("0250", "Pharmacy"),
    ("0320", "Radiology Diagnostic"),
    ("0420", "Physical Therapy"),
)


@dataclass
class GeneratedCorpus:
    claims: list[ClaimRecord]
    documents: list[DocumentRecord]


def _money(rng: random.Random, low: float, high: float) -> float:
    return round(rng.uniform(low, high), 2)


def _claim_id(rng: random.Random, year: int | None = None) -> str:
    y = year or rng.randint(2022, 2026)
    return f"CLM-{y}-{rng.randint(100000, 999999)}"


def _person(rng: random.Random) -> tuple[str, str]:
    first = rng.choice(FIRST_NAMES)
    mid = rng.choice(MIDDLE_INITIALS)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {mid} {last}".replace("  ", " ").strip()
    dob = (
        f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/"
        f"{rng.randint(1960, 2005)}"
    )
    return name, dob


def _address(rng: random.Random) -> tuple[str, str]:
    city, state, zipc = rng.choice(WI_CITIES)
    street_no = rng.randint(10, 999)
    street = rng.choice(STREETS)
    return f"{street_no} {street}, {city} {state} {zipc}", state


def _patient_id(rng: random.Random, prefix: str | None = None) -> str:
    p = prefix or rng.choice(("PID", "MRN", "ACC"))
    return f"{p}-{rng.randint(100000, 999999)}"


def _policy_number(rng: random.Random) -> str:
    return f"AF-{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"


def _date(rng: random.Random, year: int | None = None) -> str:
    y = year or rng.randint(2023, 2026)
    return f"{y}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def _split_for_index(i: int) -> str:
    # Stable ~80/10/10 without requiring sklearn.
    mod = i % 10
    if mod == 0:
        return "test"
    if mod == 1:
        return "val"
    return "train"


def _fields_from_mapping(mapping: dict[str, Any]) -> list[FieldRecord]:
    return [
        FieldRecord(
            field_name=k,
            field_value=None if v is None else str(v),
            field_role="ground_truth",
        )
        for k, v in mapping.items()
    ]


# ---------------------------------------------------------------------------
# Medical document renderers
# ---------------------------------------------------------------------------

def _render_hcfa(sk: dict[str, Any]) -> str:
    patient = sk["patient"]
    provider = sk["provider"]
    carrier = sk["carrier"]
    services = sk["services"]
    financials = sk["financials"]
    dx = ", ".join(services.get("diagnosis_codes") or [])
    cpt = ", ".join(services.get("procedure_codes") or [])
    lines = [
        "HCFA CMS-1500 HEALTH INSURANCE CLAIM FORM",
        "Physician or Supplier Information",
        f"Carrier Name: {carrier['name']}",
        f"Claim Number: {sk['claim_id']}",
        f"Policy Number: {carrier.get('policy_number') or ''}",
        f"Patient Name: {patient['name']}",
        f"Date of Birth: {patient['dob']}",
        f"Patient ID: {patient['patient_id']}",
        f"Address: {patient['address']}",
        f"Physician or Supplier: {provider['name']}",
        f"NPI: {provider.get('npi') or ''}",
        f"Place of Service: {services.get('place_of_service') or '11'}",
        f"Service Dates: {services['service_date_from']} to {services['service_date_to']}",
        f"Diagnosis Codes: {dx}",
        f"Procedure Codes: {cpt}",
        f"Total Charges: ${financials['total_charges']:,.2f}",
        "Diagnosis and procedure codes follow.",
    ]
    return "\n".join(lines)


def _render_ub04(sk: dict[str, Any]) -> str:
    patient = sk["patient"]
    provider = sk["provider"]
    carrier = sk["carrier"]
    services = sk["services"]
    financials = sk["financials"]
    rev = services.get("revenue_codes") or []
    rev_line = rev[0] if rev else "0450 Emergency Room"
    lines = [
        "UB-04 UNIFORM BILLING FORM CMS-1450",
        f"Type of Bill: {services.get('type_of_bill') or '111'}",
        f"Carrier Name: {carrier['name']}",
        f"Claim Number: {sk['claim_id']}",
        f"Patient Name: {patient['name']}",
        f"DOB: {patient['dob']}",
        f"MRN: {patient['patient_id']}",
        f"Patient Address: {patient['address']}",
        f"Facility: {provider['name']}",
        f"NPI: {provider.get('npi') or ''}",
        f"Statement Covers Period: {services['service_date_from']} - {services['service_date_to']}",
        f"Revenue Code: {rev_line}",
        f"Total Charges: ${financials['total_charges']:,.2f}",
    ]
    return "\n".join(lines)


def _render_medical_other(sk: dict[str, Any]) -> str:
    patient = sk["patient"]
    provider = sk["provider"]
    financials = sk["financials"]
    note = sk.get("narrative_notes") or "Non-standard medical bill layout."
    balance = financials.get("balance_due")
    if balance is None:
        balance = financials["total_charges"]
    lines = [
        f"{provider['name'].upper()} STATEMENT",
        "Thank you for your recent visit.",
        f"Patient: {patient['name']}",
        f"Account: {patient['patient_id']}",
        f"Amount Due: ${float(balance):,.2f}",
        "Please remit payment within 30 days.",
        note,
    ]
    return "\n".join(lines)


def _medical_ground_truth(sk: dict[str, Any]) -> dict[str, str | None]:
    patient = sk["patient"]
    doc_type = sk["document_type"]
    if doc_type == "other":
        # Match existing eval convention: sparse fields on non-standard bills.
        return {
            "claim_id": None,
            "name": patient["name"],
            "dob": None,
            "patient_id": None,
            "address": None,
        }
    return {
        "claim_id": sk["claim_id"],
        "name": patient["name"],
        "dob": patient["dob"],
        "patient_id": patient["patient_id"],
        "address": patient["address"],
    }


def generate_medical_skeleton(
    rng: random.Random,
    *,
    document_type: str,
    claim_id: str | None = None,
) -> dict[str, Any]:
    name, dob = _person(rng)
    address, state = _address(rng)
    provider_name, facility, npi = rng.choice(PROVIDERS)
    cid = claim_id or _claim_id(rng)
    svc_from = _date(rng)
    svc_to = svc_from
    total = _money(rng, 85.0, 18500.0)
    paid = round(total * rng.uniform(0.0, 0.4), 2)
    sk: dict[str, Any] = {
        "claim_id": cid,
        "document_type": document_type,
        "carrier": {
            "name": rng.choice(CARRIER_VARIANTS),
            "claim_office": f"{state} Claims Center",
            "policy_number": _policy_number(rng),
        },
        "patient": {
            "name": name,
            "dob": dob,
            "patient_id": _patient_id(rng, "MRN" if document_type == "ub04" else "PID"),
            "address": address,
            "sex": rng.choice(["M", "F", "U"]),
        },
        "provider": {
            "name": provider_name,
            "npi": npi,
            "tax_id": f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}",
            "address": _address(rng)[0],
            "facility_type": facility,
        },
        "services": {
            "service_date_from": svc_from,
            "service_date_to": svc_to,
            "place_of_service": rng.choice(["11", "21", "22", "23"]),
            "type_of_bill": rng.choice(["111", "131", "121"]) if document_type == "ub04" else None,
            "diagnosis_codes": rng.sample(DX_CODES, k=rng.randint(1, 3)),
            "procedure_codes": rng.sample(CPT_CODES, k=rng.randint(1, 3)),
            "revenue_codes": [
                f"{code} {label}" for code, label in rng.sample(REV_CODES, k=1)
            ]
            if document_type == "ub04"
            else [],
            "line_items": [
                {
                    "description": "Professional / facility charge",
                    "code": rng.choice(CPT_CODES),
                    "units": 1,
                    "charge": total,
                }
            ],
        },
        "financials": {
            "total_charges": total,
            "amount_paid": paid,
            "balance_due": round(total - paid, 2),
            "currency": "USD",
        },
        "narrative_notes": None
        if document_type != "other"
        else rng.choice(
            [
                "Non-standard medical bill layout.",
                "Walk-in urgent care receipt — arbitrary layout.",
                "Physical therapy billing statement.",
            ]
        ),
    }
    return sk


def generate_medical_document(
    rng: random.Random,
    *,
    document_type: str,
    index: int = 0,
    claim_id: str | None = None,
) -> tuple[ClaimRecord, DocumentRecord]:
    sk = generate_medical_skeleton(rng, document_type=document_type, claim_id=claim_id)
    if document_type == "hcfa":
        text = _render_hcfa(sk)
        title = "HCFA / CMS-1500 Medical Claim"
        prefix = "med-hcfa"
    elif document_type == "ub04":
        text = _render_ub04(sk)
        title = "UB-04 Uniform Billing Form"
        prefix = "med-ub04"
    else:
        text = _render_medical_other(sk)
        title = "Non-standard Medical Statement"
        prefix = "med-other"

    gt = _medical_ground_truth(sk)
    claim = ClaimRecord(
        claim_id=sk["claim_id"],
        application="medical_bills",
        carrier_name=sk["carrier"]["name"],
        state=sk["patient"]["address"].split()[-2]
        if " " in sk["patient"]["address"]
        else "WI",
        date_of_loss=sk["services"]["service_date_from"],
        loss_type="liability_third_party",
        policy_number=sk["carrier"].get("policy_number"),
        insured_name=sk["patient"]["name"],
        metadata={"domain": "medical_bills", "synthetic": True},
    )
    # Prefer a readable WI state code when address parse is fragile.
    if len(claim.state or "") != 2:
        claim.state = "WI"

    doc = DocumentRecord(
        document_id=f"{prefix}-{index:03d}",
        claim_id=sk["claim_id"],
        application="medical_bills",
        document_type=document_type,
        title=title,
        text=text,
        source_kind="synthetic_seed",
        is_synthetic=True,
        split=_split_for_index(index),
        skeleton=sk,
        metadata={
            "carrier_style": "american_family_simulation",
            "schema": "medical_bill_skeleton",
        },
        fields=_fields_from_mapping(gt),
    )
    return claim, doc


# ---------------------------------------------------------------------------
# Salvage document renderers
# ---------------------------------------------------------------------------

def _render_log(sk: dict[str, Any]) -> str:
    vehicle = sk["vehicle"]
    parties = sk["parties"]
    carrier = sk["carrier"]
    financials = sk["financials"]
    dates = sk.get("dates") or {}
    payoff = financials.get("payoff_amount") or 0.0
    lines = [
        "LETTER OF GUARANTEE",
        parties.get("lienholder") or "Lienholder Services",
        parties.get("lienholder_address") or "",
        "",
        f"Date: {dates.get('letter_date') or ''}",
        f"Carrier: {carrier['name']}",
        f"Claim Number: {sk['claim_id']}",
        f"Policy Number: {carrier.get('policy_number') or ''}",
        f"Insured: {parties['insured']}",
        f"Adjuster: {carrier.get('adjuster_name') or ''}",
        "",
        "This letter guarantees that the insurer reimbursement will pay the bank first "
        "for the outstanding loan/lease balance on the total-loss vehicle described below.",
        "",
        f"VIN: {vehicle['vin']}",
        f"Year: {vehicle['year']}",
        f"Make: {vehicle['make']}",
        f"Model: {vehicle['model']}",
        f"Payoff Amount: ${float(payoff):,.2f}",
        "",
        "Please remit payoff funds payable to the lienholder of record. "
        "Upon receipt, the lienholder will release title interest.",
    ]
    return "\n".join(line for line in lines if line is not None)


def _render_sales(sk: dict[str, Any]) -> str:
    vehicle = sk["vehicle"]
    parties = sk["parties"]
    carrier = sk["carrier"]
    financials = sk["financials"]
    dates = sk.get("dates") or {}
    price = financials.get("purchase_price") or 0.0
    tax = financials.get("sales_tax") or 0.0
    lines = [
        "SALVAGE SALES RECEIPT",
        f"Carrier: {carrier['name']}",
        f"Claim Number: {sk['claim_id']}",
        f"Bill of Sale Date: {dates.get('bill_of_sale_date') or ''}",
        f"Sold To: {parties.get('buyer') or ''}",
        f"Buyer: {parties.get('buyer') or ''}",
        f"Buyer Address: {parties.get('buyer_address') or ''}",
        f"Auction / Yard: {parties.get('auction_yard') or ''}",
        f"Vehicle: {vehicle['year']} {vehicle['make']} {vehicle['model']}",
        f"VIN: {vehicle['vin']}",
        f"Year: {vehicle['year']}",
        f"Make: {vehicle['make']}",
        f"Model: {vehicle['model']}",
        f"Purchase Price: ${float(price):,.2f}",
        f"Sales Tax: ${float(tax):,.2f}",
        f"Total: ${float(price) + float(tax):,.2f}",
        "Title branded SALVAGE. Sold as-is, where-is.",
    ]
    return "\n".join(lines)


def _render_salvage_other(sk: dict[str, Any]) -> str:
    vehicle = sk["vehicle"]
    parties = sk["parties"]
    financials = sk["financials"]
    dates = sk.get("dates") or {}
    note = sk.get("narrative_notes") or "Other salvage-claim attachment."
    tow = financials.get("towing_fees") or 0.0
    storage = financials.get("storage_fees") or 0.0
    lines = [
        "TOWING / STORAGE INVOICE",
        f"Tow Company: {parties.get('tow_company') or 'Recovery Services'}",
        f"Claim Number: {sk['claim_id']}",
        f"Vehicle: {vehicle['year']} {vehicle['make']} {vehicle['model']}",
        f"VIN: {vehicle['vin']}",
        f"Tow Date: {dates.get('tow_date') or ''}",
        f"Storage Days: {dates.get('storage_days') or 0}",
        f"Towing Fees: ${float(tow):,.2f}",
        f"Storage Fees: ${float(storage):,.2f}",
        f"Total: ${float(tow) + float(storage):,.2f}",
        note,
    ]
    return "\n".join(lines)


def _salvage_ground_truth(sk: dict[str, Any]) -> dict[str, str | None]:
    vehicle = sk["vehicle"]
    doc_type = sk["document_type"]
    if doc_type == "other":
        # Keep core vehicle fields when present; claim_id may appear on invoices.
        return {
            "claim_id": sk["claim_id"],
            "vin": vehicle["vin"],
            "year": vehicle["year"],
            "make": vehicle["make"],
            "model": vehicle["model"],
        }
    return {
        "claim_id": sk["claim_id"],
        "vin": vehicle["vin"],
        "year": vehicle["year"],
        "make": vehicle["make"],
        "model": vehicle["model"],
    }


def generate_salvage_skeleton(
    rng: random.Random,
    *,
    document_type: str,
    claim_id: str | None = None,
) -> dict[str, Any]:
    vin, year, make, model = rng.choice(VEHICLES)
    insured, _ = _person(rng)
    lienholder, lien_addr = rng.choice(LIENHOLDERS)
    buyer, buyer_addr = rng.choice(SALVAGE_BUYERS)
    cid = claim_id or _claim_id(rng)
    loss_date = _date(rng)
    payoff = _money(rng, 800.0, 18500.0)
    purchase = _money(rng, 150.0, 3500.0)
    tax = round(purchase * 0.05, 2)
    sk: dict[str, Any] = {
        "claim_id": cid,
        "document_type": document_type,
        "carrier": {
            "name": rng.choice(CARRIER_VARIANTS),
            "claim_office": "WI Salvage Unit",
            "adjuster_name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
            "policy_number": _policy_number(rng),
        },
        "vehicle": {
            "vin": vin,
            "year": year,
            "make": make,
            "model": model,
            "body_style": rng.choice(["4D Sedan", "SUV", "Coupe", "Pickup"]),
            "color": rng.choice(["Silver", "Black", "White", "Blue", "Red"]),
            "odometer": rng.randint(18000, 185000),
            "license_plate": f"WI{rng.randint(1000, 9999)}",
            "state": "WI",
        },
        "parties": {
            "insured": insured,
            "lienholder": lienholder if document_type == "log" else None,
            "lienholder_address": lien_addr if document_type == "log" else None,
            "buyer": buyer if document_type == "sales" else None,
            "buyer_address": buyer_addr if document_type == "sales" else None,
            "auction_yard": buyer if document_type == "sales" else None,
            "tow_company": rng.choice(TOW_COMPANIES) if document_type == "other" else None,
        },
        "loss_event": {
            "date_of_loss": loss_date,
            "loss_type": rng.choice(["collision", "theft", "fire", "flood"]),
            "location": _address(rng)[0],
            "total_loss_declared": True,
            "acv": _money(rng, 2500.0, 32000.0),
            "salvage_value": purchase,
        },
        "financials": {
            "payoff_amount": payoff if document_type == "log" else None,
            "purchase_price": purchase if document_type == "sales" else None,
            "sales_tax": tax if document_type == "sales" else None,
            "storage_fees": _money(rng, 75.0, 600.0) if document_type == "other" else None,
            "towing_fees": _money(rng, 125.0, 450.0) if document_type == "other" else None,
            "currency": "USD",
        },
        "dates": {
            "letter_date": loss_date if document_type == "log" else None,
            "bill_of_sale_date": _date(rng) if document_type == "sales" else None,
            "tow_date": loss_date if document_type == "other" else None,
            "storage_days": rng.randint(1, 14) if document_type == "other" else None,
        },
        "narrative_notes": None
        if document_type != "other"
        else "Vehicle relocated from accident scene to storage lot.",
    }
    return sk


def generate_salvage_document(
    rng: random.Random,
    *,
    document_type: str,
    index: int = 0,
    claim_id: str | None = None,
) -> tuple[ClaimRecord, DocumentRecord]:
    sk = generate_salvage_skeleton(rng, document_type=document_type, claim_id=claim_id)
    if document_type == "log":
        text = _render_log(sk)
        title = "Letter of Guarantee"
        prefix = "sal-log"
    elif document_type == "sales":
        text = _render_sales(sk)
        title = "Salvage Sales Receipt"
        prefix = "sal-sales"
    else:
        text = _render_salvage_other(sk)
        title = "Towing / Storage Invoice"
        prefix = "sal-other"

    gt = _salvage_ground_truth(sk)
    claim = ClaimRecord(
        claim_id=sk["claim_id"],
        application="salvage_claims",
        carrier_name=sk["carrier"]["name"],
        state=sk["vehicle"].get("state") or "WI",
        date_of_loss=(sk.get("loss_event") or {}).get("date_of_loss"),
        loss_type=(sk.get("loss_event") or {}).get("loss_type"),
        policy_number=sk["carrier"].get("policy_number"),
        insured_name=sk["parties"]["insured"],
        metadata={"domain": "salvage_claims", "synthetic": True},
    )
    doc = DocumentRecord(
        document_id=f"{prefix}-{index:03d}",
        claim_id=sk["claim_id"],
        application="salvage_claims",
        document_type=document_type,
        title=title,
        text=text,
        source_kind="synthetic_seed",
        is_synthetic=True,
        split=_split_for_index(index),
        skeleton=sk,
        metadata={
            "carrier_style": "american_family_simulation",
            "schema": "salvage_document_skeleton",
        },
        fields=_fields_from_mapping(gt),
    )
    return claim, doc


def generate_claim_bundle(
    rng: random.Random,
    *,
    application: str,
    bundle_index: int,
) -> tuple[ClaimRecord, list[DocumentRecord]]:
    """Generate a claim with multiple related documents (bundle)."""
    cid = _claim_id(rng)
    docs: list[DocumentRecord] = []
    if application == "medical_bills":
        # Typical casualty file: HCFA + optional UB-04 + clinic statement.
        types = ["hcfa", "ub04", "other"]
        claim: ClaimRecord | None = None
        for i, dtype in enumerate(types):
            c, d = generate_medical_document(
                rng, document_type=dtype, index=bundle_index * 10 + i, claim_id=cid
            )
            # Keep a single claim record; reuse first.
            if claim is None:
                claim = c
            d.document_id = f"med-bundle{bundle_index:02d}-{dtype}"
            docs.append(d)
        assert claim is not None
        return claim, docs

    if application == "salvage_claims":
        types = ["log", "sales", "other"]
        claim = None
        for i, dtype in enumerate(types):
            c, d = generate_salvage_document(
                rng, document_type=dtype, index=bundle_index * 10 + i, claim_id=cid
            )
            if claim is None:
                claim = c
            d.document_id = f"sal-bundle{bundle_index:02d}-{dtype}"
            docs.append(d)
        assert claim is not None
        return claim, docs

    raise ValueError(f"Unsupported application for bundles: {application}")


def generate_corpus(
    *,
    seed: int = 42,
    medical_per_type: int = 8,
    salvage_per_type: int = 8,
    bundles_per_app: int = 2,
    include_canonical_fixtures: bool = True,
) -> GeneratedCorpus:
    """Build a diverse synthetic corpus of medical + salvage documents."""
    rng = random.Random(seed)
    claims: list[ClaimRecord] = []
    documents: list[DocumentRecord] = []
    seen_claims: set[str] = set()

    def _add(claim: ClaimRecord, docs: Iterable[DocumentRecord]) -> None:
        if claim.claim_id not in seen_claims:
            claims.append(claim)
            seen_claims.add(claim.claim_id)
        documents.extend(docs)

    if include_canonical_fixtures:
        for claim, doc in _canonical_fixtures():
            _add(claim, [doc])

    for i in range(medical_per_type):
        for dtype in ("hcfa", "ub04", "other"):
            claim, doc = generate_medical_document(
                rng, document_type=dtype, index=100 + i
            )
            # Avoid ID collisions across types by encoding type in id (already).
            doc.document_id = f"med-{dtype}-{100 + i:03d}"
            _add(claim, [doc])

    for i in range(salvage_per_type):
        for dtype in ("log", "sales", "other"):
            claim, doc = generate_salvage_document(
                rng, document_type=dtype, index=100 + i
            )
            doc.document_id = f"sal-{dtype}-{100 + i:03d}"
            _add(claim, [doc])

    for b in range(bundles_per_app):
        claim, docs = generate_claim_bundle(
            rng, application="medical_bills", bundle_index=b
        )
        _add(claim, docs)
        claim, docs = generate_claim_bundle(
            rng, application="salvage_claims", bundle_index=b
        )
        _add(claim, docs)

    return GeneratedCorpus(claims=claims, documents=documents)


def _canonical_fixtures() -> list[tuple[ClaimRecord, DocumentRecord]]:
    """Preserve the well-known CI fixture IDs used across DICIE tests/eval."""
    fixtures: list[tuple[ClaimRecord, DocumentRecord]] = []

    med_rows = [
        (
            "med-hcfa-001",
            "hcfa",
            "CLM-2024-551122",
            {
                "claim_id": "CLM-2024-551122",
                "name": "Jane Q Public",
                "dob": "03/14/1988",
                "patient_id": "PID-778812",
                "address": "100 Oak Avenue, Madison WI 53703",
            },
            (
                "HCFA CMS-1500 HEALTH INSURANCE CLAIM FORM\n"
                "Physician or Supplier Information\n"
                "Patient Name: Jane Q Public\n"
                "Date of Birth: 03/14/1988\n"
                "Patient ID: PID-778812\n"
                "Claim Number: CLM-2024-551122\n"
                "Address: 100 Oak Avenue, Madison WI 53703\n"
                "Carrier Name: American Family\n"
                "Diagnosis and procedure codes follow."
            ),
        ),
        (
            "med-ub04-002",
            "ub04",
            "CLM-2024-660033",
            {
                "claim_id": "CLM-2024-660033",
                "name": "John A Smith",
                "dob": "1990-07-22",
                "patient_id": "MRN-990011",
                "address": "55 Lake Street, Madison WI 53704",
            },
            (
                "UB-04 UNIFORM BILLING FORM CMS-1450\n"
                "Type of Bill: 111\n"
                "Patient Name: John A Smith\n"
                "DOB: 1990-07-22\n"
                "MRN: MRN-990011\n"
                "Claim Number: CLM-2024-660033\n"
                "Patient Address: 55 Lake Street, Madison WI 53704\n"
                "Revenue Code: 0450 Emergency Room"
            ),
        ),
        (
            "med-other-003",
            "other",
            "CLM-SYN-MED-OTHER-003",
            {
                "claim_id": None,
                "name": "Alex Rivera",
                "dob": None,
                "patient_id": None,
                "address": None,
            },
            (
                "COMMUNITY CLINIC STATEMENT\n"
                "Thank you for your recent visit.\n"
                "Patient: Alex Rivera\n"
                "Account: ACC-4411\n"
                "Amount Due: $240.00\n"
                "Please remit payment within 30 days."
            ),
        ),
    ]
    for doc_id, dtype, claim_id, gt, text in med_rows:
        claim = ClaimRecord(
            claim_id=claim_id,
            application="medical_bills",
            carrier_name="American Family",
            state="WI",
            insured_name=gt.get("name"),
            metadata={"canonical_fixture": True},
        )
        doc = DocumentRecord(
            document_id=doc_id,
            claim_id=claim_id,
            application="medical_bills",
            document_type=dtype,
            title=f"Canonical {dtype}",
            text=text,
            source_kind="canonical_fixture",
            is_synthetic=True,
            split="test",
            metadata={"carrier_style": "american_family_simulation"},
            fields=_fields_from_mapping(gt),
        )
        fixtures.append((claim, doc))

    sal_rows = [
        (
            "sal-log-001",
            "log",
            "CLM-2024-100200",
            {
                "claim_id": "CLM-2024-100200",
                "vin": "1HGCM82633A004352",
                "year": "2018",
                "make": "Honda",
                "model": "Accord",
            },
            (
                "LETTER OF GUARANTEE\n"
                "First National Bank Lienholder Services\n"
                "This letter guarantees that the insurer reimbursement will pay the bank first.\n"
                "Claim Number: CLM-2024-100200\n"
                "VIN: 1HGCM82633A004352\n"
                "Year: 2018\n"
                "Make: Honda\n"
                "Model: Accord\n"
                "Payoff Amount: $4,250.00"
            ),
        ),
        (
            "sal-sales-002",
            "sales",
            "CLM-2024-100201",
            {
                "claim_id": "CLM-2024-100201",
                "vin": "1FADP3F20EL123456",
                "year": "2015",
                "make": "Ford",
                "model": "Focus",
            },
            (
                "SALVAGE SALES RECEIPT\n"
                "Sold To: Midwest Auto Salvage\n"
                "Buyer: Midwest Auto Salvage LLC\n"
                "Purchase Price: $850.00\n"
                "Sales Tax: $42.50\n"
                "Bill of Sale Date: 2024-05-12\n"
                "Vehicle: 2015 Ford Focus\n"
                "VIN: 1FADP3F20EL123456\n"
                "Claim Number: CLM-2024-100201"
            ),
        ),
        (
            "sal-other-003",
            "other",
            "CLM-2024-100202",
            {
                "claim_id": "CLM-2024-100202",
                "vin": None,
                "year": None,
                "make": None,
                "model": None,
            },
            (
                "TOWING INVOICE\n"
                "Vehicle relocated from accident scene to storage lot.\n"
                "Claim Number: CLM-2024-100202\n"
                "Tow Date: 2024-05-10\n"
                "Storage Days: 3\n"
                "Total: $375.00"
            ),
        ),
    ]
    for doc_id, dtype, claim_id, gt, text in sal_rows:
        claim = ClaimRecord(
            claim_id=claim_id,
            application="salvage_claims",
            carrier_name="American Family",
            state="WI",
            metadata={"canonical_fixture": True},
        )
        doc = DocumentRecord(
            document_id=doc_id,
            claim_id=claim_id,
            application="salvage_claims",
            document_type=dtype,
            title=f"Canonical {dtype}",
            text=text,
            source_kind="canonical_fixture",
            is_synthetic=True,
            split="test",
            metadata={"carrier_style": "american_family_simulation"},
            fields=_fields_from_mapping(gt),
        )
        fixtures.append((claim, doc))

    return fixtures
