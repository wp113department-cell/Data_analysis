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
- Multiplier: Worksheet cell H7 - its live value is shown in PRICE SHEETS LOADED. Unit price = MSRP x multiplier.
- Assembly fee: 10% of the cabinet subtotal, only when assembled (Order tab row 63).
- Shipping: never priced - the office sends a shipping quote.
- Search tip: search_price_sheet(brand="HCC", tab="Worksheet", header_row=27, texts=[...all codes...],
  columns=["B", "C", "<the line's price column>"]) - e.g. Colonial White -> columns=["B","C","L"].
- Naming patterns (to turn sizes into codes; sizes are inches W x H x D, base height 34.5, base depth 24,
  wall depth 12):
  wall cabinet W<width><height> (30x18 -> W3018, 12x30 -> W1230, 36x24 -> W3624, 36x12 -> W3612);
  base B<width> (B24); 3-drawer base 3DB<width>; 2-drawer base 2DB<width>; sink base SB<width> (any base described as for/under a sink is a sink base, e.g. 33 under sink -> SB33);
  trash pull-out base B18-TR / B21-TR (only 18 and 21 exist); lazy susan LS33 / LS36; pantry UP<width><height>.
  Trim is sold in 8 ft pieces: outside corner molding OC8, toe kick TK8, scribe molding SM8, quarter round QR8,
  inside corner ICM8, crown CM8 / BCM8; skins/panels: BSK (base skin), WSK<width><height> (wall skin),
  BEP2430 / WEP12xx (end panels). Sheet descriptions look like 24"W 34 1/2"T 24"D.
- Build the likely code for every sized item, check them all in ONE search, then ask only about items with
  no exact match (offer the real closest rows).
