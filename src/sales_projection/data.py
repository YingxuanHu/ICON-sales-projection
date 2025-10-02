from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

REQUIRED_COLUMNS: Sequence[str] = ("ItemSku", "Quantity", "InvoiceTxnDate", "ItemType")


def load_invoice_data(invoice_path: Path, item_types: Optional[Iterable[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(invoice_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Invoice data missing required columns: {sorted(missing)}")

    if item_types is None:
        item_types = ("Inventory",)

    filtered = df[df["ItemType"].isin(item_types)].copy()
    if filtered.empty:
        raise ValueError("No rows left after filtering ItemType. Check the source data.")

    filtered["InvoiceTxnDate"] = filtered["InvoiceTxnDate"].astype(str).str[:10]
    filtered["InvoiceTxnDate"] = pd.to_datetime(filtered["InvoiceTxnDate"], errors="coerce")
    filtered = filtered.dropna(subset=["InvoiceTxnDate", "Quantity", "ItemSku"])

    filtered["Quantity"] = pd.to_numeric(filtered["Quantity"], errors="coerce")
    filtered = filtered.dropna(subset=["Quantity"])

    filtered = filtered[["ItemSku", "InvoiceTxnDate", "Quantity"]]
    filtered = filtered.rename(columns={"InvoiceTxnDate": "ds", "Quantity": "y"})
    filtered["ItemSku"] = filtered["ItemSku"].astype(str).str.strip()

    return filtered


def ensure_monthly_frequency(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    monthly_frames = []
    for sku, group in df.groupby("ItemSku"):
        prepared = group.sort_values("ds").copy()
        prepared = prepared.set_index("ds").asfreq("MS", fill_value=fill_value)
        prepared = prepared.reset_index()
        prepared["ItemSku"] = sku
        monthly_frames.append(prepared)

    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly = monthly.rename(columns={"index": "ds"})
    monthly = monthly.sort_values(["ItemSku", "ds"]).reset_index(drop=True)
    return monthly
