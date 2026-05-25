"""
25 diverse function definitions across 5 domains.
Used for generating function calling preference data.
"""

FUNCTION_POOL = [
    # ── Domain 1: Information Retrieval ──
    {
        "name": "get_weather",
        "description": "Get current weather information for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "Temperature unit, default celsius"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information on a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results to return, default 5"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_news",
        "description": "Get latest news articles by category or keyword",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["technology", "business", "sports", "science", "health", "entertainment"],
                             "description": "News category"},
                "keyword": {"type": "string", "description": "Optional keyword filter"},
                "count": {"type": "integer", "description": "Number of articles, default 5"},
            },
            "required": ["category"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get current stock price and basic metrics for a ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL'"},
                "include_history": {"type": "boolean", "description": "Include 30-day price history, default false"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_exchange_rate",
        "description": "Get currency exchange rate between two currencies",
        "parameters": {
            "type": "object",
            "properties": {
                "from_currency": {"type": "string", "description": "Source currency code, e.g. 'USD'"},
                "to_currency": {"type": "string", "description": "Target currency code, e.g. 'CNY'"},
                "amount": {"type": "number", "description": "Amount to convert, default 1"},
            },
            "required": ["from_currency", "to_currency"],
        },
    },

    # ── Domain 2: Productivity ──
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body content"},
                "cc": {"type": "string", "description": "CC email address (optional)"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new event on the calendar",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title"},
                "date": {"type": "string", "description": "Event date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Event start time in HH:MM format"},
                "duration_minutes": {"type": "integer", "description": "Duration in minutes, default 60"},
                "location": {"type": "string", "description": "Event location (optional)"},
            },
            "required": ["title", "date", "time"],
        },
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder for a specific date and time",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Reminder message"},
                "datetime": {"type": "string", "description": "When to remind, ISO 8601 format"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "description": "Priority level"},
            },
            "required": ["message", "datetime"],
        },
    },
    {
        "name": "create_note",
        "description": "Create a new note in the note-taking app",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Note title"},
                "content": {"type": "string", "description": "Note content"},
                "tags": {"type": "string", "description": "Comma-separated tags for categorization"},
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "search_contacts",
        "description": "Search contacts by name, email, or company",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term (name, email, or company)"},
                "limit": {"type": "integer", "description": "Max results to return, default 10"},
            },
            "required": ["query"],
        },
    },

    # ── Domain 3: Travel & Lifestyle ──
    {
        "name": "search_flights",
        "description": "Search for available flights between two cities",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Departure city or airport code"},
                "destination": {"type": "string", "description": "Arrival city or airport code"},
                "date": {"type": "string", "description": "Departure date in YYYY-MM-DD format"},
                "passengers": {"type": "integer", "description": "Number of passengers, default 1"},
                "cabin_class": {"type": "string", "enum": ["economy", "business", "first"],
                                "description": "Cabin class, default economy"},
            },
            "required": ["origin", "destination", "date"],
        },
    },
    {
        "name": "book_hotel",
        "description": "Search and book hotels in a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "check_in": {"type": "string", "description": "Check-in date YYYY-MM-DD"},
                "check_out": {"type": "string", "description": "Check-out date YYYY-MM-DD"},
                "guests": {"type": "integer", "description": "Number of guests, default 1"},
                "max_price": {"type": "number", "description": "Maximum price per night in USD"},
            },
            "required": ["city", "check_in", "check_out"],
        },
    },
    {
        "name": "search_restaurant",
        "description": "Find restaurants by location and cuisine",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City or area name"},
                "cuisine": {"type": "string", "description": "Cuisine type, e.g. 'italian', 'chinese'"},
                "price_range": {"type": "string", "enum": ["$", "$$", "$$$", "$$$$"],
                                "description": "Price range"},
                "min_rating": {"type": "number", "description": "Minimum rating (1-5)"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_directions",
        "description": "Get directions from one location to another",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Starting location"},
                "destination": {"type": "string", "description": "Ending location"},
                "mode": {"type": "string", "enum": ["driving", "walking", "transit", "cycling"],
                         "description": "Travel mode, default driving"},
            },
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "get_attractions",
        "description": "Get popular tourist attractions in a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "category": {"type": "string", "enum": ["museum", "park", "landmark", "shopping", "all"],
                             "description": "Attraction category, default all"},
                "count": {"type": "integer", "description": "Number of results, default 10"},
            },
            "required": ["city"],
        },
    },

    # ── Domain 4: Finance ──
    {
        "name": "transfer_money",
        "description": "Transfer money to another account",
        "parameters": {
            "type": "object",
            "properties": {
                "to_account": {"type": "string", "description": "Recipient account number or username"},
                "amount": {"type": "number", "description": "Amount to transfer"},
                "currency": {"type": "string", "description": "Currency code, default USD"},
                "note": {"type": "string", "description": "Transfer note or memo"},
            },
            "required": ["to_account", "amount"],
        },
    },
    {
        "name": "check_balance",
        "description": "Check account balance",
        "parameters": {
            "type": "object",
            "properties": {
                "account_type": {"type": "string", "enum": ["checking", "savings", "credit"],
                                 "description": "Account type to check"},
            },
            "required": ["account_type"],
        },
    },
    {
        "name": "get_transaction_history",
        "description": "Get transaction history for an account",
        "parameters": {
            "type": "object",
            "properties": {
                "account_type": {"type": "string", "enum": ["checking", "savings", "credit"],
                                 "description": "Account type"},
                "days": {"type": "integer", "description": "Number of days of history, default 30"},
                "category": {"type": "string", "description": "Filter by transaction category"},
            },
            "required": ["account_type"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount between two currencies at current rates",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code"},
                "to_currency": {"type": "string", "description": "Target currency code"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "calculate_loan",
        "description": "Calculate monthly payment for a loan",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Loan principal amount"},
                "annual_rate": {"type": "number", "description": "Annual interest rate as percentage, e.g. 5.5"},
                "term_months": {"type": "integer", "description": "Loan term in months"},
            },
            "required": ["principal", "annual_rate", "term_months"],
        },
    },

    # ── Domain 5: Technical ──
    {
        "name": "execute_code",
        "description": "Execute a Python code snippet and return the output",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "description": "Execution timeout in seconds, default 10"},
            },
            "required": ["code"],
        },
    },
    {
        "name": "query_database",
        "description": "Execute a read-only SQL query on the database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL SELECT query"},
                "database": {"type": "string", "description": "Database name"},
                "limit": {"type": "integer", "description": "Max rows to return, default 100"},
            },
            "required": ["query", "database"],
        },
    },
    {
        "name": "translate_text",
        "description": "Translate text from one language to another",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to translate"},
                "source_lang": {"type": "string", "description": "Source language code, e.g. 'en'"},
                "target_lang": {"type": "string", "description": "Target language code, e.g. 'zh'"},
            },
            "required": ["text", "target_lang"],
        },
    },
    {
        "name": "summarize_text",
        "description": "Summarize a long text into key points",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to summarize"},
                "max_sentences": {"type": "integer", "description": "Maximum sentences in summary, default 3"},
                "style": {"type": "string", "enum": ["bullet", "paragraph"],
                          "description": "Summary style, default bullet"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "analyze_sentiment",
        "description": "Analyze the sentiment of a given text",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"},
                "language": {"type": "string", "description": "Text language code, default 'en'"},
            },
            "required": ["text"],
        },
    },
]


DOMAIN_MAP = {
    "information": ["get_weather", "search_web", "get_news", "get_stock_price", "get_exchange_rate"],
    "productivity": ["send_email", "create_calendar_event", "set_reminder", "create_note", "search_contacts"],
    "travel": ["search_flights", "book_hotel", "search_restaurant", "get_directions", "get_attractions"],
    "finance": ["transfer_money", "check_balance", "get_transaction_history", "convert_currency", "calculate_loan"],
    "technical": ["execute_code", "query_database", "translate_text", "summarize_text", "analyze_sentiment"],
}


def get_functions_by_names(names):
    pool = {f["name"]: f for f in FUNCTION_POOL}
    return [pool[n] for n in names if n in pool]


def format_functions_for_prompt(functions):
    """Format function definitions into a system prompt string."""
    lines = []
    for f in functions:
        params = f["parameters"]["properties"]
        required = f["parameters"].get("required", [])
        param_parts = []
        for pname, pinfo in params.items():
            req = " [REQUIRED]" if pname in required else ""
            desc = pinfo.get("description", "")
            ptype = pinfo.get("type", "string")
            enum = pinfo.get("enum")
            enum_str = f", options: {enum}" if enum else ""
            param_parts.append(f"    - {pname} ({ptype}{enum_str}): {desc}{req}")
        lines.append(f"- {f['name']}: {f['description']}")
        lines.extend(param_parts)
    return "\n".join(lines)
