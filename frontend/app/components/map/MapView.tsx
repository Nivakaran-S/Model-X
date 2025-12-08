'use client'
import { useState } from "react";
import SriLankaMap from "./SriLankaMap";
import DistrictInfoPanel from "./DistrictInfoPanel";
import { Card } from "../ui/card";
import { MapPin, Activity } from "lucide-react";
import { useRogerData } from "../../hooks/use-roger-data";
import { Badge } from "../ui/badge";

const MapView = () => {
  const [selectedDistrict, setSelectedDistrict] = useState<string | null>(null);
  const { events, isConnected } = useRogerData();

  // Count alerts per district (simplified - matches district names in event summaries)
  const districtAlertCounts: Record<string, number> = {};

  (events ?? []).forEach(event => {
    const summary = (event.summary ?? '').toLowerCase();
    // Check if district name is mentioned in the event
    ['colombo', 'gampaha', 'kandy', 'jaffna', 'galle', 'matara', 'hambantota',
      'anuradhapura', 'polonnaruwa', 'batticaloa', 'ampara', 'trincomalee',
      'kurunegala', 'puttalam', 'kalutara', 'ratnapura', 'kegalle', 'nuwara eliya',
      'badulla', 'monaragala', 'kilinochchi', 'mannar', 'vavuniya', 'mullaitivu', 'matale'
    ].forEach(district => {
      if (summary.includes(district)) {
        const capitalizedDistrict = district.charAt(0).toUpperCase() + district.slice(1);
        districtAlertCounts[capitalizedDistrict] = (districtAlertCounts[capitalizedDistrict] || 0) + 1;
      }
    });
  });

  // Count critical events
  const criticalEvents = events.filter(e => e.severity === 'critical' || e.severity === 'high');

  return (
    <div className="space-y-4">
      <Card className="p-4 sm:p-6 bg-card border-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-2">
            <MapPin className="w-5 h-5 text-primary" />
            <h2 className="text-base sm:text-lg font-bold">TERRITORY MAP</h2>
          </div>
          <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
            {isConnected ? (
              <Badge className="bg-success/20 text-success flex items-center gap-2 text-xs">
                <span className="w-2 h-2 rounded-full bg-success animate-pulse"></span>
                Live
              </Badge>
            ) : (
              <Badge className="bg-warning/20 text-warning text-xs">Reconnecting...</Badge>
            )}
            <Badge className="border border-border flex items-center gap-2 text-xs">
              <Activity className="w-3 h-3" />
              {criticalEvents.length} Critical
            </Badge>
            <span className="text-xs font-mono text-muted-foreground hidden sm:inline">
              Click any district for detailed intelligence
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
          <div className="lg:col-span-2 order-1">
            <div className="h-[350px] sm:h-[450px] lg:h-[550px] w-full">
              <SriLankaMap
                selectedDistrict={selectedDistrict}
                onDistrictSelect={setSelectedDistrict}
                alertCounts={districtAlertCounts}
                className="w-full h-full"
              />
            </div>
          </div>


          <div className="lg:col-span-1 order-2">
            <DistrictInfoPanel district={selectedDistrict} />
          </div>
        </div>
      </Card>
    </div>
  );
};

export default MapView;
