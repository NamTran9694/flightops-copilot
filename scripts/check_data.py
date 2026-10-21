"""Quick checks of the generated database."""
import sqlite3
from pathlib import Path
DB = Path(__file__).resolve().parents[1] / "data" / "flightops.db"
con = sqlite3.connect(DB)

queries = {
    "Avg flights per day (first 3 months)": """
        SELECT substr(flight_date, 1, 7) AS month,
                COUNT(*) / COUNT(DISTINCT flight_date) AS flights_per_day
        FROM flights GROUP BY month ORDER BY month LIMIT 3""",
    "Aircraft mix": """
        SELECT aircraft_type, COUNT(*) AS flights, ROUND(AVG(meals_loaded)) AS avg_meals
        FROM flights GROUP BY aircraft_type ORDER BY flights DESC""",
    "Catering delay rate (%)": """
        SELECT ROUND(100.0 * AVG(catering_delay_min > 0), 2) FROM flights""",
}

for title, q in queries.items():
    print(f"\n== {title} ==")
    for row in con.execute(q):
        print(row)