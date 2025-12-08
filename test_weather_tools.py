"""
test_weather_tools.py
Quick test for weather and flood tools only (no sessions required)
"""
import sys
import os
import io

# Force UTF-8 output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("[TEST] WEATHER & FLOOD TOOLS")
print("="*60)

# Test 1: DMC Alerts
print("\n[1] Testing tool_dmc_alerts...")
try:
    from src.utils.utils import tool_dmc_alerts
    result = tool_dmc_alerts()
    print(f"[OK] DMC Alerts: {str(result)[:300]}...")
except Exception as e:
    print(f"[FAIL] DMC Alerts: {e}")

# Test 2: Weather Nowcast
print("\n[2] Testing tool_weather_nowcast...")
try:
    from src.utils.utils import tool_weather_nowcast
    result = tool_weather_nowcast()
    print(f"[OK] Weather Nowcast: {str(result)[:300]}...")
except Exception as e:
    print(f"[FAIL] Weather Nowcast: {e}")

# Test 3: District Weather
print("\n[3] Testing tool_district_weather...")
try:
    from src.utils.utils import tool_district_weather
    result = tool_district_weather("colombo")
    print(f"[OK] District Weather: {str(result)[:300]}...")
except Exception as e:
    print(f"[FAIL] District Weather: {e}")

# Test 4: RiverNet (may take longer - uses Playwright)
print("\n[4] Testing tool_rivernet_status (may take 30-60 seconds)...")
try:
    from src.utils.utils import tool_rivernet_status
    result = tool_rivernet_status()
    print(f"[OK] RiverNet: {str(result)[:300]}...")
except Exception as e:
    print(f"[FAIL] RiverNet: {e}")

print("\n" + "="*60)
print("[DONE] Weather tools test complete")
print("="*60)
