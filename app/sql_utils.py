"""
Shared SQL extraction logic, used by both the CLI test harness and the
Streamlit UI, so there is a single source of truth.
"""

import re


def extract_sql(raw_output: str) -> str:
    text = raw_output.strip()

    text = re.sub(r"```sql", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    no_query_match = re.match(r"^\s*NO_QUERY\s*:", text, re.IGNORECASE)
    if no_query_match:
        return text.split("\n")[0].strip()

    select_match = re.search(r"SELECT\b.*", text, re.IGNORECASE | re.DOTALL)
    if select_match:
        text = select_match.group(0)
    else:
        return text.strip()

    semicolon_index = text.find(";")
    if semicolon_index != -1:
        text = text[: semicolon_index + 1]

    return text.strip()