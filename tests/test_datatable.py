from app.datatable import header_label, parse_series

TABLE = """
<table class="wikitable">
<tr><th>Election</th><th>Turnout</th><th>Turnout as % of VEP</th></tr>
<tr><td>2000</td><td>105,000,000</td><td>50.3%</td></tr>
<tr><td>2004</td><td>122,000,000</td><td>55.7%</td></tr>
<tr><td>2008</td><td>131,000,000</td><td>58.2%</td></tr>
</table>
"""


def test_series_uses_the_percentage_column():
    chart = parse_series(TABLE, "US voter turnout by election")
    assert chart["labels"] == ["2000", "2004", "2008"]
    assert chart["values"] == [50.3, 55.7, 58.2]
    assert chart["unit"] == "%"


def test_series_needs_three_rows():
    one = TABLE.replace("<tr><td>2004</td><td>122,000,000</td><td>55.7%</td></tr>", "")
    one = one.replace("<tr><td>2008</td><td>131,000,000</td><td>58.2%</td></tr>", "")
    assert parse_series(one, "turnout") is None


def test_series_ignores_tables_without_a_percentage():
    html = TABLE.replace("Turnout as % of VEP", "Turnout").replace("50.3%", "50").replace("55.7%", "55").replace("58.2%", "58")
    assert parse_series(html, "turnout") is None


def test_header_label_drops_citations():
    assert header_label("Turnout as % of VEP[11][12]") == "Turnout as % of VEP"


def test_us_question_rejects_a_foreign_page():
    assert parse_series(TABLE, "US voter turnout by year", "Voter turnout in Australia") is None


def test_us_question_accepts_a_us_page():
    chart = parse_series(TABLE, "US voter turnout by year", "Voter turnout in United States presidential elections")
    assert chart["labels"] == ["2000", "2004", "2008"]


def test_us_page_url_with_underscores_matches():
    chart = parse_series(TABLE, "USA turnout by year", "https://en.wikipedia.org/wiki/Voter_turnout_in_United_States_presidential_elections")
    assert chart["values"] == [50.3, 55.7, 58.2]
