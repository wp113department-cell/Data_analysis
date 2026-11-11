HCC - Heritage Craft Cabinetry, Signature Series (VERIFIED guide; checked against the workbook on 2026-11-11.
If the workbook's layout changes - new columns, moved rows, new lines - update this file.)

- Product tab: "Worksheet". Header row: 27 (Item # = column B, Description = column C). Products from row 29;
  section titles (Wall Cabinets, Base Cabinets, ...) sit in column A.
- ALL 13 HCC color lines (customer-facing name -> price column on the product row):
  Colonial White = L | Colonial Grey = M | Colonial Cherry = N | Colonial Espresso = O | Regal Ivory = P |
  Regal Truffle = Q | Summit White = R | Sierra Desert = S | Oxford White = T | Colonial Navy = U |
  Oxford Oak = V | Colonial Timber = W | A+Plus = X
  (Row 27 headers in L-X are the manufacturer's internal names - never show them to customers.)
- MSRP for an item = the value in that line's column on the item's row.
- Multiplier: Worksheet cell H7 - always read the current value from the sheet. Unit price = MSRP x multiplier.
- Assembly fee: 10% of the cabinet subtotal, only when assembled (Order tab row 63).
- Shipping: never priced - the office sends a shipping quote.
- Search tip: search_price_sheet(brand="HCC", tab="Worksheet", header_row=27, texts=[...all codes...]).
