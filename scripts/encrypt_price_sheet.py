"""
Encrypt a brand's price workbook so it can live in a public repo.

    python scripts/encrypt_price_sheet.py "Copy of HCC Signature Series Quote.xlsm" price_lists/HCC.xlsm.enc

Uses PRICE_SHEET_KEY from the environment / .env, or creates a new key and prints it.
The app decrypts the file in memory at runtime when the original workbook isn't present
(e.g. on Streamlit Cloud) - put the same PRICE_SHEET_KEY in the app's secrets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from cryptography.fernet import Fernet
from dotenv import load_dotenv


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    key = os.getenv("PRICE_SHEET_KEY")
    created = False
    if not key:
        key, created = Fernet.generate_key().decode(), True
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(Fernet(key.encode()).encrypt(src.read_bytes()))
    print(f"Encrypted {src} -> {dst} ({dst.stat().st_size:,} bytes)")
    if created:
        print("New key (add to .env and to the app's secrets as PRICE_SHEET_KEY):")
        print(key)


if __name__ == "__main__":
    main()
