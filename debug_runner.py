import os
import sys
import json
from datetime import datetime

# Ensure we can find the 'src' module from the root
sys.path.append(os.getcwd())

try:
    from src.utils.utils import (
        scrape_facebook,
        scrape_twitter,
        scrape_local_news,
        scrape_reddit,
        scrape_government_gazette,
        scrape_cse_stock_data,
        tool_weather_nowcast,
        tool_dmc_alerts,
        scrape_linkedin,
        scrape_instagram,
    )
    print("✅ Libraries loaded successfully.\n")
except ImportError as e:
    print(f"❌ Error loading libraries: {e}")
    print("Make sure you are running this from the 'Roger-Final' folder.")
    sys.exit(1)

def print_separator(char="=", length=70):
    print(char * length)

def print_header(text):
    print_separator()
    print(f"  {text}")
    print_separator()

def run_test(name, func, description="", **kwargs):
    print(f"\n🔍 Testing: {name}")
    if description:
        print(f"   {description}")
    print("-" * 70)
    
    start_time = datetime.now()
    
    try:
        # Check if it's a LangChain tool (needs .invoke)
        if hasattr(func, "invoke"):
            res = func.invoke(kwargs)
        else:
            res = func(**kwargs)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Try to print pretty JSON
        try:
            parsed = json.loads(res)
            
            # Custom formatting for better readability
            if isinstance(parsed, dict):
                if "results" in parsed:
                    print(f"\n✅ Success! Found {len(parsed.get('results', []))} results in {elapsed:.2f}s")
                    print(f"\nSample Results:")
                    for i, item in enumerate(parsed['results'][:3], 1):
                        print(f"\n  [{i}] {item.get('title', 'No title')}")
                        if 'snippet' in item:
                            snippet = item['snippet'][:150] + "..." if len(item['snippet']) > 150 else item['snippet']
                            print(f"      {snippet}")
                        if 'url' in item:
                            print(f"      🔗 {item['url']}")
                else:
                    print(f"\n✅ Success in {elapsed:.2f}s")
                    print(json.dumps(parsed, indent=2)[:1000])
            else:
                print(json.dumps(parsed, indent=2)[:1000])
                
        except:
            print(res[:1000] if len(res) > 1000 else res)
        
        print(f"\n⏱️  Completed in {elapsed:.2f} seconds")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("-" * 70)

def check_sessions():
    """Check which session files exist"""
    print_header("Session Status Check")
    
    session_paths = [
        "src/utils/.sessions",
        ".sessions"
    ]
    
    platforms = ["facebook", "twitter", "linkedin", "instagram", "reddit"]
    found_sessions = []
    print("session_path: ", session_paths)
    
    for path in session_paths:
        if os.path.exists(path):
            print(f"\n📁 Checking {path}/")
            for platform in platforms:
                session_file = os.path.join(path, f"{platform}_storage_state.json")
                if os.path.exists(session_file):
                    size = os.path.getsize(session_file)
                    print(f"   ✅ {platform:12} ({size:,} bytes)")
                    found_sessions.append(platform)
                else:
                    print(f"   ❌ {platform:12} (not found)")
    
    if not found_sessions:
        print("\n⚠️  No session files found!")
        print("   Run 'python src/utils/session_manager.py' to create sessions.")
    
    print_separator()
    return found_sessions

def main():
    print_header("Roger Debug Runner - Comprehensive Tool Testing")
    
    print("\n📋 Available Test Categories:")
    print("  1. Weather & Alerts (No auth required)")
    print("  2. News & Government (No auth required)")
    print("  3. Financial Data (No auth required)")
    print("  4. Social Media (Requires auth)")
    print("  5. Check Sessions")
    print("  6. Run All Tests")
    print("  q. Quit")
    
    choice = input("\nSelect category (1-6 or q): ").strip()
    
    if choice == "q":
        return
    
    if choice == "5":
        check_sessions()
        return
    
    # === CATEGORY 1: Weather & Alerts ===
    if choice in ["1", "6"]:
        print_header("CATEGORY 1: Weather & Alerts")
        
        run_test(
            "Weather Nowcast",
            tool_weather_nowcast,
            "Comprehensive weather data from Department of Meteorology",
            location="Colombo"
        )
        
        run_test(
            "DMC Alerts",
            tool_dmc_alerts,
            "Disaster Management Centre severe weather alerts"
        )
    
    # === CATEGORY 2: News & Government ===
    if choice in ["2", "6"]:
        print_header("CATEGORY 2: News & Government")
        
        run_test(
            "Local News",
            scrape_local_news,
            "Scraping Daily Mirror, Daily FT, News First",
            keywords=["economy", "politics"],
            max_articles=5
        )
        
        run_test(
            "Government Gazette",
            scrape_government_gazette,
            "Latest gazette notifications",
            keywords=["regulation"],
            max_items=3
        )
    
    # === CATEGORY 3: Financial Data ===
    if choice in ["3", "6"]:
        print_header("CATEGORY 3: Financial Data")
        
        run_test(
            "CSE Stock Data",
            scrape_cse_stock_data,
            "Colombo Stock Exchange - ASPI Index",
            symbol="ASPI",
            period="1d"
        )
    
    # === CATEGORY 4: Social Media ===
    if choice in ["4", "6"]:
        print_header("CATEGORY 4: Social Media (Authentication Required)")
        
        available_sessions = check_sessions()
        
        if "facebook" in available_sessions:
            run_test(
                "Facebook",
                scrape_facebook,
                "Facebook search results",
                keywords=["Sri Lanka", "Elon musk", "business"],
                max_items=5
            )
        else:
            print("\n⚠️  Facebook session not found - skipping")

        if "instagram" in available_sessions:
            run_test(
                "Instagram",
                scrape_instagram,
                "Instagram search results",
                keywords=["Sri Lanka", "Elon musk", "business"],
                max_items=5
            )
        else:
            print("\n⚠️  Facebook session not found - skipping")

        if "linkedin" in available_sessions:
            run_test(
                "Linkedin",
                scrape_linkedin,
                "Linkedin search results",
                keywords=["Sri Lanka", "Elon musk", "business"],
                max_items=5
            )
        else:
            print("\n⚠️  Facebook session not found - skipping")

        
        if "twitter" in available_sessions:
            run_test(
                "Twitter",
                scrape_twitter,
                "Twitter/X search",
                query="Sri Lanka economy"
            )
        else:
            print("\n⚠️  Twitter session not found - skipping")
        
        # Reddit doesn't need session
        run_test(
            "Reddit",
            scrape_reddit,
            "Reddit posts (no auth needed)",
            keywords=["Sri Lanka"],
            limit=5
        )
    
    print_header("Testing Complete!")
    print(f"\n⏰ Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
