"""Format recipe ingredient quantity + unit for display / embedding."""


def format_ingredient_amount(quantity: str | None, unit: str | None) -> str:
    q = (quantity or "").strip()
    u = (unit or "").strip()
    if q and u:
        return f"{q} {u}"
    return q or u or ""


def ingredient_label(ing: dict) -> str:
    name = ing.get("name", "")
    if ing.get("amount"):
        amount = str(ing["amount"]).strip()
        return f"{name} ({amount})" if amount else name
    amount = format_ingredient_amount(ing.get("quantity"), ing.get("unit"))
    return f"{name} ({amount})" if amount else name
