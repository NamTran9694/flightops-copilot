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
    "daily_ops view (first 3 days)": """
        SELECT * FROM daily_ops LIMIT 3""",
    "Hours per flight by month (KPI trend)": """
        SELECT substr(ops_date, 1, 7) AS month,
               ROUND(SUM(labor_hours) / SUM(flights), 2) AS hours_per_flight
        FROM daily_ops GROUP BY month ORDER BY month""",
    "Overtime ratio by department (%)": """
        SELECT department,
               ROUND(100.0 * SUM(overtime_hours) / SUM(regular_hours + overtime_hours), 1)
        FROM labor_daily GROUP BY department""",
    "Active employees by role": """
        SELECT role, COUNT(*) FROM employees
        WHERE term_date IS NULL GROUP BY role ORDER BY role""",
    "Driver monthly attrition (%)": """
        WITH months AS (SELECT DISTINCT substr(flight_date, 1, 7) || '-01' AS m FROM flights)
        SELECT substr(m, 1, 7) AS month,
               ROUND(100.0 *
                 (SELECT COUNT(*) FROM employees WHERE role = 'Driver'
                    AND term_date >= m AND term_date < date(m, '+1 month'))
                 / (SELECT COUNT(*) FROM employees WHERE role = 'Driver'
                    AND hire_date < m AND (term_date IS NULL OR term_date >= m)), 1) AS attrition_pct
        FROM months ORDER BY m""",       
}

for title, q in queries.items():
    print(f"\n== {title} ==")
    for row in con.execute(q):
        print(row)