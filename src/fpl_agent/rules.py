"""FPL rules and constants."""

SQUAD_SIZE = 15
STARTING_XI = 11
BUDGET_TENTHS = 1000  # £100.0m — FPL stores prices in tenths of millions
MAX_PER_CLUB = 3

# element_type id -> (name, squad count, starting min, starting max)
POSITIONS: dict[int, tuple[str, int, int, int]] = {
    1: ("GK",  2, 1, 1),
    2: ("DEF", 5, 3, 5),
    3: ("MID", 5, 2, 5),
    4: ("FWD", 3, 1, 3),
}
