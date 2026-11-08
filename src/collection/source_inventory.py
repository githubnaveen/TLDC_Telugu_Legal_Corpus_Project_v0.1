from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "source_id",
    "state",
    "court_or_portal",
    "document_category",
    "language_availability",
    "access_url",
    "terms_reviewed",
    "permission_status",
    "notes",
]

ALLOWED_STATES = {"Telangana", "Andhra Pradesh"}


def validate_source_inventory(csv_path):
    """Validate a source inventory CSV and return the parsed dataframe."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    invalid_states = set(df["state"].dropna()) - ALLOWED_STATES
    if invalid_states:
        raise ValueError(f"Invalid state values: {invalid_states}")

    if df["source_id"].duplicated().any():
        raise ValueError("Duplicate source IDs found")

    print("Source inventory validation passed.")
    print(f"Total sources: {len(df)}")
    print("\nSources by state:")
    print(df["state"].value_counts(dropna=False))
    print("\nPermission status:")
    print(df["permission_status"].value_counts(dropna=False))
    return df


def select_approved_sources(csv_path, output_path=None):
    """Return and optionally save the approved source list."""
    df = validate_source_inventory(csv_path)
    approved = df[df["permission_status"].astype(str).str.lower().eq("approved")].copy()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        approved.to_csv(output_path, index=False)
    return approved


if __name__ == "__main__":
    validate_source_inventory("metadata/source_inventory.csv")
