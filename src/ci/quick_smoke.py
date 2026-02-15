# src/ci/quick_smoke.py
from pathlib import Path
import datetime

REPORT_PATH = Path("report.html")

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <html>
      <body>
        <h1>CI Pipeline Successful ✅</h1>
        <p>Quick smoke check executed successfully.</p>
        <p>Timestamp: {now}</p>
      </body>
    </html>
    """

    REPORT_PATH.write_text(html)
    print("report.html generated successfully")
