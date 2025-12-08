"""
test_all_tools.py
Comprehensive test script for all Roger agentic AI tools
Runs each tool and validates output format
"""
import json
import sys
import os

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

def test_tool(name, func, *args, **kwargs):
    """Test a tool and report results"""
    print(f"\n{'='*60}")
    print(f"[TEST] {name}")
    print(f"{'='*60}")
    try:
        result = func(*args, **kwargs)
        
        # Parse result if it's JSON string
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if "error" in parsed:
                    print(f"[WARN] TOOL RETURNED ERROR: {parsed['error']}")
                    if "solution" in parsed:
                        print(f"   [TIP] Solution: {parsed['solution']}")
                    return {"status": "error", "error": parsed["error"]}
                else:
                    print(f"[OK] SUCCESS")
                    # Print sample of results
                    if "results" in parsed:
                        print(f"   [DATA] Results count: {len(parsed['results'])}")
                        if parsed['results'] and len(parsed['results']) > 0:
                            print(f"   [SAMPLE] {str(parsed['results'][0])[:200]}...")
                    elif isinstance(parsed, dict):
                        for key in list(parsed.keys())[:3]:
                            val = str(parsed[key])[:100]
                            print(f"   - {key}: {val}...")
                    return {"status": "success", "data": parsed}
            except json.JSONDecodeError:
                print(f"[OK] SUCCESS (non-JSON response)")
                print(f"   [SAMPLE] {result[:200]}...")
                return {"status": "success", "data": result}
        else:
            print(f"[OK] SUCCESS")
            print(f"   [TYPE] Response type: {type(result)}")
            return {"status": "success", "data": result}
            
    except Exception as e:
        print(f"[FAIL] FAILED: {e}")
        return {"status": "failed", "error": str(e)}

def main():
    results = {}
    
    print("\n" + "="*70)
    print("[START] ROGER AGENTIC AI - COMPREHENSIVE TOOL TESTING")
    print(f"   Started: {datetime.now().isoformat()}")
    print("="*70)
    
    # =====================================================
    # 1. WEATHER & FLOOD TOOLS (No session required)
    # =====================================================
    print("\n\n[CATEGORY] WEATHER & FLOOD TOOLS")
    print("-"*50)
    
    try:
        from src.utils.utils import tool_dmc_alerts
        results["tool_dmc_alerts"] = test_tool("tool_dmc_alerts", tool_dmc_alerts)
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        results["tool_dmc_alerts"] = {"status": "import_error", "error": str(e)}
    
    try:
        from src.utils.utils import tool_weather_nowcast
        results["tool_weather_nowcast"] = test_tool("tool_weather_nowcast", tool_weather_nowcast)
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        results["tool_weather_nowcast"] = {"status": "import_error", "error": str(e)}
    
    try:
        from src.utils.utils import tool_rivernet_status
        results["tool_rivernet_status"] = test_tool("tool_rivernet_status", tool_rivernet_status)
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        results["tool_rivernet_status"] = {"status": "import_error", "error": str(e)}
    
    try:
        from src.utils.utils import tool_district_weather
        results["tool_district_weather"] = test_tool("tool_district_weather", tool_district_weather, "colombo")
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        results["tool_district_weather"] = {"status": "import_error", "error": str(e)}
    
    # =====================================================
    # 2. NEWS & OFFICIAL SOURCES (No session required)
    # =====================================================
    print("\n\n[CATEGORY] NEWS & OFFICIAL SOURCES")
    print("-"*50)
    
    try:
        from src.utils.tool_factory import create_tool_set
        tools = create_tool_set(include_profile_scrapers=False)
        
        # Local News
        local_news = tools.get("scrape_local_news")
        if local_news:
            results["scrape_local_news"] = test_tool("scrape_local_news", local_news.invoke, {"keywords": ["sri lanka"], "max_articles": 5})
        else:
            results["scrape_local_news"] = {"status": "not_found", "error": "Tool not found in ToolSet"}
        
        # CSE Stock Data
        cse_tool = tools.get("scrape_cse_stock_data")
        if cse_tool:
            results["scrape_cse_stock_data"] = test_tool("scrape_cse_stock_data", cse_tool.invoke, {"symbol": "ASPI", "period": "1d"})
        else:
            results["scrape_cse_stock_data"] = {"status": "not_found", "error": "Tool not found in ToolSet"}
        
        # Government Gazette
        gazette_tool = tools.get("scrape_government_gazette")
        if gazette_tool:
            results["scrape_government_gazette"] = test_tool("scrape_government_gazette", gazette_tool.invoke, {"keywords": None, "max_items": 5})
        else:
            results["scrape_government_gazette"] = {"status": "not_found", "error": "Tool not found in ToolSet"}
        
        # Parliament Minutes
        parliament_tool = tools.get("scrape_parliament_minutes")
        if parliament_tool:
            results["scrape_parliament_minutes"] = test_tool("scrape_parliament_minutes", parliament_tool.invoke, {"keywords": None, "max_items": 5})
        else:
            results["scrape_parliament_minutes"] = {"status": "not_found", "error": "Tool not found in ToolSet"}
        
        # Reddit (no session needed)
        reddit_tool = tools.get("scrape_reddit")
        if reddit_tool:
            results["scrape_reddit"] = test_tool("scrape_reddit", reddit_tool.invoke, {"keywords": ["sri lanka"], "limit": 5})
        else:
            results["scrape_reddit"] = {"status": "not_found", "error": "Tool not found in ToolSet"}
            
    except Exception as e:
        print(f"[FAIL] ToolSet creation error: {e}")
        results["tool_factory"] = {"status": "failed", "error": str(e)}
    
    # =====================================================
    # 3. SOCIAL MEDIA TOOLS (Session required)
    # =====================================================
    print("\n\n[CATEGORY] SOCIAL MEDIA TOOLS (Session Required)")
    print("-"*50)
    
    try:
        from src.utils.tool_factory import create_tool_set
        tools = create_tool_set(include_profile_scrapers=True)
        
        # Twitter Search
        twitter_tool = tools.get("scrape_twitter")
        if twitter_tool:
            results["scrape_twitter"] = test_tool("scrape_twitter", twitter_tool.invoke, {"query": "sri lanka", "max_items": 3})
        else:
            results["scrape_twitter"] = {"status": "not_found", "error": "Tool not found in ToolSet"}
        
        # Facebook Search
        fb_tool = tools.get("scrape_facebook")
        if fb_tool:
            results["scrape_facebook"] = test_tool("scrape_facebook", fb_tool.invoke, {"keywords": ["sri lanka"], "max_items": 3})
        else:
            results["scrape_facebook"] = {"status": "not_found", "error": "Tool not found in ToolSet"}
        
        # LinkedIn Search
        linkedin_tool = tools.get("scrape_linkedin")
        if linkedin_tool:
            results["scrape_linkedin"] = test_tool("scrape_linkedin", linkedin_tool.invoke, {"keywords": ["sri lanka"], "max_items": 3})
        else:
            results["scrape_linkedin"] = {"status": "not_found", "error": "Tool not found in ToolSet"}
        
        # Instagram Search
        instagram_tool = tools.get("scrape_instagram")
        if instagram_tool:
            results["scrape_instagram"] = test_tool("scrape_instagram", instagram_tool.invoke, {"keywords": ["srilanka"], "max_items": 3})
        else:
            results["scrape_instagram"] = {"status": "not_found", "error": "Tool not found in ToolSet"}
            
    except Exception as e:
        print(f"[FAIL] Social media tools error: {e}")
    
    # =====================================================
    # 4. PROFILE SCRAPERS (Session required)
    # =====================================================
    print("\n\n[CATEGORY] PROFILE SCRAPERS (Session Required)")
    print("-"*50)
    
    try:
        from src.utils.profile_scrapers import scrape_twitter_profile, scrape_facebook_profile
        
        # Twitter Profile
        results["scrape_twitter_profile"] = test_tool("scrape_twitter_profile", scrape_twitter_profile.invoke, {"username": "SLTMobitel", "max_items": 3})
        
        # Facebook Profile  
        results["scrape_facebook_profile"] = test_tool("scrape_facebook_profile", scrape_facebook_profile.invoke, {"profile_url": "https://www.facebook.com/DialogAxiata", "max_items": 3})
        
    except Exception as e:
        print(f"[FAIL] Profile scrapers error: {e}")
    
    # =====================================================
    # SUMMARY REPORT
    # =====================================================
    print("\n\n" + "="*70)
    print("[SUMMARY] TEST RESULTS")
    print("="*70)
    
    success_count = 0
    error_count = 0
    session_issues = []
    other_errors = []
    
    for tool_name, result in results.items():
        status = result.get("status", "unknown")
        if status == "success":
            success_count += 1
            print(f"[OK] {tool_name}: SUCCESS")
        elif status == "error":
            error = result.get("error", "Unknown error")
            if "session" in error.lower() or "Session" in error:
                session_issues.append(tool_name)
                print(f"[SESSION] {tool_name}: SESSION ISSUE - {error[:50]}")
            else:
                other_errors.append((tool_name, error))
                print(f"[WARN] {tool_name}: ERROR - {error[:50]}")
            error_count += 1
        else:
            error = result.get("error", "Unknown")
            other_errors.append((tool_name, error))
            print(f"[FAIL] {tool_name}: {status.upper()} - {error[:50]}")
            error_count += 1
    
    print(f"\n[TOTALS]:")
    print(f"   [OK] Successful: {success_count}")
    print(f"   [FAIL] Errors: {error_count}")
    
    if session_issues:
        print(f"\n[SESSION] TOOLS NEEDING SESSION REFRESH:")
        for tool in session_issues:
            print(f"   - {tool}")
    
    if other_errors:
        print(f"\n[WARN] TOOLS WITH OTHER ERRORS:")
        for tool, error in other_errors:
            print(f"   - {tool}: {error[:80]}")
    
    # Save results to file
    with open("tool_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "success": success_count,
                "errors": error_count,
                "session_issues": session_issues,
                "other_errors": [t[0] for t in other_errors]
            },
            "details": {k: {"status": v.get("status"), "error": v.get("error")} for k, v in results.items()}
        }, f, indent=2)
    print(f"\n[SAVED] Results saved to: tool_test_results.json")
    
    return results

if __name__ == "__main__":
    main()

