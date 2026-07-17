"""
AI-Powered Supermarket Sales Dashboard
utils/config.py

Central configuration for the application.
"""

from pathlib import Path

# -----------------------------
# App Information
# -----------------------------

APP_NAME = "AI-Powered Supermarket Sales Dashboard"
APP_ICON = "🛒"
APP_LAYOUT = "wide"

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

DEFAULT_DATASET = DATA_DIR / "supermarket_sales.csv"
LOGO_PATH = ASSETS_DIR / "logo.png"
STYLE_PATH = ASSETS_DIR / "styles.css"

# -----------------------------
# Theme
# -----------------------------

PRIMARY = "#2563EB"
SECONDARY = "#10B981"
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER = "#EF4444"
INFO = "#06B6D4"
BACKGROUND = "#F8FAFC"
CARD = "#FFFFFF"
TEXT = "#111827"

PLOTLY_TEMPLATE = "plotly_white"

# -----------------------------
# Dashboard Settings
# -----------------------------

DEFAULT_PAGE_SIZE = 20
MAX_UPLOAD_MB = 100

SUPPORTED_FILES = [
    "csv",
    "xlsx",
    "xls",
]

# -----------------------------
# KPI Icons
# -----------------------------

KPI_ICONS = {
    "revenue": "💰",
    "orders": "🧾",
    "customers": "👥",
    "products": "📦",
    "profit": "📈",
    "rating": "⭐",
    "branch": "🏬",
    "city": "🌍",
}

# -----------------------------
# Required Dataset Columns
# -----------------------------

REQUIRED_COLUMNS = [
    "Date",
    "Total",
    "Quantity",
]

# -----------------------------
# ML Configuration
# -----------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 200
N_CLUSTERS = 3

# -----------------------------
# Export Settings
# -----------------------------

PDF_NAME = "dashboard_summary.pdf"
CSV_NAME = "sales.csv"
EXCEL_NAME = "sales.xlsx"

# -----------------------------
# Chart Defaults
# -----------------------------

CHART_HEIGHT = 420
USE_CONTAINER_WIDTH = True

# -----------------------------
# Date Formats
# -----------------------------

DATE_FORMAT = "%d-%m-%Y"
DATETIME_FORMAT = "%d-%m-%Y %H:%M"

# -----------------------------
# Sidebar Menu
# -----------------------------

MENU_ITEMS = [
    "Dashboard",
    "Sales",
    "Branches",
    "Products",
    "Customers",
    "Finance",
    "Ratings",
    "Prediction",
    "Reports",
]
