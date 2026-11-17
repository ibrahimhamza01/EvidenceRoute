import pytest

from evidenceroute.ingestion.parser import _item_heading


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("ITEM 1. B USINESS", "1"),
        ("ITEM 1. BUSINESS |  |  |  | ", "1"),
        ("ITEM 1A. | RISK FACTORS.", "1A"),
        ("ITEM 1C. | CYBERSECURITY.", "1C"),
        (
            "ITEM 7. | MANAGEMENT’S DISCUSSION AND ANALYSIS "
            "OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS.",
            "7",
        ),
        ("ITEM 8. | FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA.", "8"),
        (
            "ITEM 9. CHANGES IN AND DISAGREEMENTS WITH "
            "ACCOU NTANTS ON ACCOUNTING AND FINANCIAL DISCLOSURE",
            "9",
        ),
        ("Item 1", None),
        ("ITEM 1. BUSINESS STRATEGY FOR 2024", None),
        ("ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA | 34", None),
    ],
)
def test_expanded_heading_formats(heading, expected):
    assert _item_heading(heading) == expected


def test_combined_business_and_properties_heading():
    assert (
        _item_heading("Items 1 and 2. Business and Properties")
        == "1"
    )


def test_decorated_financial_item_headings():
    assert _item_heading(
        "Item 7. Bank of America Corporation and Subsidiaries "
        "Management's Discussion and Analysis of Financial Condition "
        "and Results of Operations Table of Contents"
    ) == "7"

    assert _item_heading(
        "Item 8. Financial Statements and Supplementary Data "
        "Table of Contents"
    ) == "8"


def test_decorated_headings_do_not_accept_page_numbers():
    assert _item_heading(
        "Item 8. Financial Statements and Supplementary Data "
        "Table of Contents 86"
    ) is None


def test_bac_actual_multiline_item_7_heading():
    heading = (
        "Item 7. Bank of America Corporation and Subsidiaries\n"
        "Management's Discussion and Analysis of Financial "
        "Condition and Results of Operations\n"
        "Table of Contents"
    )
    assert _item_heading(heading) == "7"


def test_bac_actual_multiline_item_8_heading():
    heading = (
        "Item 8. Financial Statements and Supplementary Data\n"
        "Table of Contents"
    )
    assert _item_heading(heading) == "8"


def test_unrelated_multiline_text_still_rejected():
    heading = (
        "Item 8. Financial Statements and Supplementary Data\n"
        "See the discussion in the next section."
    )
    assert _item_heading(heading) is None
