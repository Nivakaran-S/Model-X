"""
test_news_tools.py
Test for news and official source tools (no social media sessions required)
"""
import sys
import os
import io
import json

# Force UTF-8 output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("[TEST] NEWS & OFFICIAL SOURCE TOOLS")
print("="*60)

try:
    from src.utils.tool_factory import create_tool_set
    print("[OK] Tool factory imported")
    tools = create_tool_set(include_profile_scrapers=False)
    print("[OK] ToolSet created")
except Exception as e:
    print(f"[FAIL] Could not create ToolSet: {e}")
    sys.exit(1)

# Test 1: Local News
print("\n[1] Testing scrape_local_news...")
try:
    local_news = tools.get("scrape_local_news")
    if local_news:
        result = local_news.invoke({"keywords": ["sri lanka"], "max_articles": 3})
        parsed = json.loads(result) if isinstance(result, str) else result
        if "error" in parsed:
            print(f"[WARN] Local News returned error: {parsed['error']}")
        else:
            count = len(parsed.get('results', []))
            print(f"[OK] Local News: {count} articles fetched")
            if count > 0:
                print(f"   Sample: {str(parsed['results'][0])[:150]}...")
    else:
        print("[WARN] scrape_local_news not found in ToolSet")
except Exception as e:
    print(f"[FAIL] Local News: {e}")

# Test 2: CSE Stock Data
print("\n[2] Testing scrape_cse_stock_data...")
try:
    cse_tool = tools.get("scrape_cse_stock_data")
    if cse_tool:
        result = cse_tool.invoke({"symbol": "ASPI", "period": "1d"})
        parsed = json.loads(result) if isinstance(result, str) else result
        if "error" in parsed:
            print(f"[WARN] CSE Stock returned error: {parsed['error']}")
        else:
            print(f"[OK] CSE Stock: {str(parsed)[:200]}...")
    else:
        print("[WARN] scrape_cse_stock_data not found in ToolSet")
except Exception as e:
    print(f"[FAIL] CSE Stock: {e}")

# Test 3: Government Gazette
print("\n[3] Testing scrape_government_gazette...")
try:
    gazette_tool = tools.get("scrape_government_gazette")
    if gazette_tool:
        result = gazette_tool.invoke({"keywords": None, "max_items": 3})
        parsed = json.loads(result) if isinstance(result, str) else result
        if "error" in parsed:
            print(f"[WARN] Gazette returned error: {parsed['error']}")
        else:
            count = len(parsed.get('results', []))
            print(f"[OK] Gazette: {count} items fetched")
    else:
        print("[WARN] scrape_government_gazette not found in ToolSet")
except Exception as e:
    print(f"[FAIL] Gazette: {e}")

# Test 4: Parliament Minutes
print("\n[4] Testing scrape_parliament_minutes...")
try:
    parliament_tool = tools.get("scrape_parliament_minutes")
    if parliament_tool:
        result = parliament_tool.invoke({"keywords": None, "max_items": 3})
        parsed = json.loads(result) if isinstance(result, str) else result
        if "error" in parsed:
            print(f"[WARN] Parliament returned error: {parsed['error']}")
        else:
            count = len(parsed.get('results', []))
            print(f"[OK] Parliament: {count} items fetched")
    else:
        print("[WARN] scrape_parliament_minutes not found in ToolSet")
except Exception as e:
    print(f"[FAIL] Parliament: {e}")

# Test 5: Reddit (no auth needed)
print("\n[5] Testing scrape_reddit...")
try:
    reddit_tool = tools.get("scrape_reddit")
    if reddit_tool:
        result = reddit_tool.invoke({"keywords": ["sri lanka"], "limit": 3})
        parsed = json.loads(result) if isinstance(result, str) else result
        if "error" in parsed:
            print(f"[WARN] Reddit returned error: {parsed['error']}")
        else:
            count = len(parsed.get('results', parsed.get('posts', [])))
            print(f"[OK] Reddit: {count} posts fetched")
    else:
        print("[WARN] scrape_reddit not found in ToolSet")
except Exception as e:
    print(f"[FAIL] Reddit: {e}")

print("\n" + "="*60)
print("[DONE] News & official source tools test complete")
print("="*60)
