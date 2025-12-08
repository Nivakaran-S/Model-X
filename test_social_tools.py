"""
test_social_tools.py
Test for social media tools (session required)
Will identify which sessions need to be created/refreshed
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
print("[TEST] SOCIAL MEDIA TOOLS (Session Required)")
print("="*60)

# Check for existing sessions
print("\n[SESSIONS] Checking for existing session files...")
session_dirs = [
    "src/utils/.sessions",
    ".sessions",
]

session_files = {}
for dir in session_dirs:
    if os.path.exists(dir):
        for f in os.listdir(dir):
            if f.endswith('.json'):
                session_files[f] = os.path.join(dir, f)

if session_files:
    print(f"[OK] Found {len(session_files)} session files:")
    for name, path in session_files.items():
        print(f"   - {name}: {path}")
else:
    print("[WARN] No session files found!")
    print("   Please create sessions using the session manager")

# Try to create ToolSet with profile scrapers
print("\n[TEST] Creating ToolSet with profile scrapers...")
try:
    from src.utils.tool_factory import create_tool_set
    tools = create_tool_set(include_profile_scrapers=True)
    print("[OK] ToolSet created")
except Exception as e:
    print(f"[FAIL] Could not create ToolSet: {e}")
    sys.exit(1)

# Test Twitter
print("\n[1] Testing scrape_twitter (keyword search)...")
try:
    twitter_tool = tools.get("scrape_twitter")
    if twitter_tool:
        result = twitter_tool.invoke({"query": "sri lanka", "max_items": 2})
        parsed = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed, dict) and "error" in parsed:
            if "session" in str(parsed['error']).lower():
                print(f"[SESSION] Twitter: Session not found or expired")
                print(f"   Error: {parsed['error'][:100]}")
            else:
                print(f"[WARN] Twitter error: {parsed['error'][:100]}")
        elif isinstance(parsed, dict):
            count = len(parsed.get('results', []))
            print(f"[OK] Twitter: {count} tweets fetched")
        elif isinstance(parsed, list):
            print(f"[OK] Twitter: {len(parsed)} tweets fetched")
        else:
            print(f"[OK] Twitter returned: {type(parsed)}")
    else:
        print("[WARN] scrape_twitter not found in ToolSet")
except Exception as e:
    print(f"[FAIL] Twitter: {e}")

# Test Facebook
print("\n[2] Testing scrape_facebook (keyword search)...")
try:
    fb_tool = tools.get("scrape_facebook")
    if fb_tool:
        result = fb_tool.invoke({"keywords": ["sri lanka"], "max_items": 2})
        parsed = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed, dict) and "error" in parsed:
            if "session" in str(parsed['error']).lower():
                print(f"[SESSION] Facebook: Session not found or expired")
                print(f"   Error: {parsed['error'][:100]}")
            else:
                print(f"[WARN] Facebook error: {parsed['error'][:100]}")
        elif isinstance(parsed, dict):
            count = len(parsed.get('results', []))
            print(f"[OK] Facebook: {count} posts fetched")
        elif isinstance(parsed, list):
            print(f"[OK] Facebook: {len(parsed)} posts fetched")
        else:
            print(f"[OK] Facebook returned: {type(parsed)}")
    else:
        print("[WARN] scrape_facebook not found in ToolSet")
except Exception as e:
    print(f"[FAIL] Facebook: {e}")

# Test LinkedIn
print("\n[3] Testing scrape_linkedin (keyword search)...")
try:
    linkedin_tool = tools.get("scrape_linkedin")
    if linkedin_tool:
        result = linkedin_tool.invoke({"keywords": ["sri lanka"], "max_items": 2})
        parsed = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed, dict) and "error" in parsed:
            if "session" in str(parsed['error']).lower():
                print(f"[SESSION] LinkedIn: Session not found or expired")
                print(f"   Error: {parsed['error'][:100]}")
            else:
                print(f"[WARN] LinkedIn error: {parsed['error'][:100]}")
        elif isinstance(parsed, dict):
            count = len(parsed.get('results', []))
            print(f"[OK] LinkedIn: {count} posts fetched")
        elif isinstance(parsed, list):
            print(f"[OK] LinkedIn: {len(parsed)} posts fetched")
        else:
            print(f"[OK] LinkedIn returned: {type(parsed)}")
    else:
        print("[WARN] scrape_linkedin not found in ToolSet")
except Exception as e:
    print(f"[FAIL] LinkedIn: {e}")

# Test Instagram
print("\n[4] Testing scrape_instagram (hashtag search)...")
try:
    instagram_tool = tools.get("scrape_instagram")
    if instagram_tool:
        result = instagram_tool.invoke({"keywords": ["srilanka"], "max_items": 2})
        parsed = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed, dict) and "error" in parsed:
            if "session" in str(parsed['error']).lower():
                print(f"[SESSION] Instagram: Session not found or expired")
                print(f"   Error: {parsed['error'][:100]}")
            else:
                print(f"[WARN] Instagram error: {parsed['error'][:100]}")
        elif isinstance(parsed, dict):
            count = len(parsed.get('results', []))
            print(f"[OK] Instagram: {count} posts fetched")
        elif isinstance(parsed, list):
            print(f"[OK] Instagram: {len(parsed)} posts fetched")
        else:
            print(f"[OK] Instagram returned: {type(parsed)}")
    else:
        print("[WARN] scrape_instagram not found in ToolSet")
except Exception as e:
    print(f"[FAIL] Instagram: {e}")

# Test Profile Scrapers
print("\n[5] Testing scrape_twitter_profile (specific account)...")
try:
    from src.utils.profile_scrapers import scrape_twitter_profile
    result = scrape_twitter_profile.invoke({"username": "SLTMobitel", "max_items": 2})
    parsed = json.loads(result) if isinstance(result, str) else result
    if isinstance(parsed, dict) and "error" in parsed:
        if "session" in str(parsed['error']).lower():
            print(f"[SESSION] Twitter Profile: Session not found or expired")
            print(f"   Error: {parsed['error'][:100]}")
        elif "timeout" in str(parsed['error']).lower():
            print(f"[TIMEOUT] Twitter Profile: Navigation timed out (X blocks automation)")
            print(f"   Error: {parsed['error'][:100]}")
        else:
            print(f"[WARN] Twitter Profile error: {parsed['error'][:100]}")
    elif isinstance(parsed, dict):
        count = len(parsed.get('results', []))
        print(f"[OK] Twitter Profile: {count} tweets fetched from @SLTMobitel")
    else:
        print(f"[OK] Twitter Profile returned: {type(parsed)}")
except Exception as e:
    print(f"[FAIL] Twitter Profile: {e}")

print("\n" + "="*60)
print("[SUMMARY]")
print("="*60)
print("If you see [SESSION] errors, please create new sessions using:")
print("  - Twitter: Run session manager with Twitter login")
print("  - Facebook: Run session manager with Facebook login") 
print("  - LinkedIn: Run session manager with LinkedIn login")
print("  - Instagram: Run session manager with Instagram login")
print("\nSession manager: python src/utils/session_manager.py")
print("="*60)
