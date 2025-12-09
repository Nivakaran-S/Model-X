"use client";

import { Card } from "../ui/card";
import { Badge } from "../ui/badge";
import { TrendingUp, TrendingDown, Activity, AlertCircle } from "lucide-react";
import { motion } from "framer-motion";
import { useRogerData } from "../../hooks/use-roger-data";

const StockPredictions = () => {
  const { events, isConnected } = useRogerData();

  // Filter for economic/market events
  const marketEvents = events.filter(e =>
    e.domain === 'economical' || e.domain === 'market'
  );

  // Extract market insights from real events
  const marketInsights = marketEvents.map(event => {
    const isBullish = event.impact_type === 'opportunity' ||
      event.summary.toLowerCase().includes('bullish') ||
      event.summary.toLowerCase().includes('growth') ||
      event.summary.toLowerCase().includes('increase') ||
      event.summary.toLowerCase().includes('positive');

    const isBearish = event.summary.toLowerCase().includes('bearish') ||
      event.summary.toLowerCase().includes('contraction') ||
      event.summary.toLowerCase().includes('decline') ||
      event.summary.toLowerCase().includes('negative');

    return {
      id: event.event_id || `market-${Math.random().toString(36).substr(2, 9)}`,
      title: event.summary,
      sentiment: isBullish ? 'bullish' : isBearish ? 'bearish' : 'neutral',
      confidence: event.confidence || 0.7,
      severity: event.severity,
      timestamp: event.timestamp,
      source: event.domain || 'Market Analysis'
    };
  });

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-card border-border">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-success" />
            <h2 className="text-lg font-bold">MARKET INTELLIGENCE - CSE</h2>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-success animate-pulse' : 'bg-destructive'}`} />
            <Badge className="font-mono text-xs border">
              {isConnected ? 'LIVE AI ANALYSIS' : 'CONNECTING...'}
            </Badge>
          </div>
        </div>

        {/* AI-Generated Market Insights from Real Data */}
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase">
            AI Market Analysis ({marketInsights.length} insights)
          </h3>

          {marketInsights.length > 0 ? (
            <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
              {marketInsights.slice(0, 10).map((insight, idx) => (
                <motion.div
                  key={insight.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className={`p-4 rounded-lg border-l-4 ${insight.sentiment === 'bullish' ? 'border-l-success bg-success/10' :
                    insight.sentiment === 'bearish' ? 'border-l-destructive bg-destructive/10' :
                      'border-l-muted bg-muted/30'
                    }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    {insight.sentiment === 'bullish' && <TrendingUp className="w-4 h-4 text-success" />}
                    {insight.sentiment === 'bearish' && <TrendingDown className="w-4 h-4 text-destructive" />}
                    {insight.sentiment === 'neutral' && <Activity className="w-4 h-4 text-muted-foreground" />}
                    <Badge className={`text-xs ${insight.sentiment === 'bullish' ? 'bg-success/20 text-success' :
                      insight.sentiment === 'bearish' ? 'bg-destructive/20 text-destructive' :
                        'bg-muted'
                      }`}>
                      {insight.sentiment.toUpperCase()}
                    </Badge>
                    <span className="text-xs text-muted-foreground ml-auto">
                      {Math.round(insight.confidence * 100)}% confidence
                    </span>
                  </div>
                  <p className="text-sm">{insight.title}</p>
                  <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
                    <span>{insight.source}</span>
                    {insight.timestamp && (
                      <span>{new Date(insight.timestamp).toLocaleTimeString()}</span>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <AlertCircle className="w-12 h-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground mb-2">No market data available yet</p>
              <p className="text-xs text-muted-foreground">
                Waiting for economic events from the AI agents...
              </p>
            </div>
          )}
        </div>

        <div className="mt-4 p-3 bg-muted/20 rounded border border-border">
          <p className="text-xs text-muted-foreground font-mono">
            <span className="text-warning font-bold">⚠ DISCLAIMER:</span> AI analysis based on real-time data. Not financial advice.
          </p>
        </div>
      </Card>
    </div>
  );
};

export default StockPredictions;
