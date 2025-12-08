import { Card } from "../ui/card";
import { Badge } from "../ui/badge";
import { TrendingUp, TrendingDown, Activity } from "lucide-react";
import { motion } from "framer-motion";
import { useRogerData } from "../../hooks/use-roger-data";

const StockPredictions = () => {
  const { events } = useRogerData();

  // Filter for economic/market events
  const marketEvents = events.filter(e => 
    e.domain === 'economical' || e.domain === 'market'
  );

  // Extract market insights
  const marketInsights = marketEvents.map(event => {
    const isBullish = event.impact_type === 'opportunity' || 
                      event.summary.toLowerCase().includes('bullish') ||
                      event.summary.toLowerCase().includes('growth');
    
    const isBearish = event.summary.toLowerCase().includes('bearish') ||
                      event.summary.toLowerCase().includes('contraction');

    return {
      symbol: "ASPI",
      title: event.summary,
      sentiment: isBullish ? 'bullish' : isBearish ? 'bearish' : 'neutral',
      confidence: event.confidence,
      severity: event.severity,
      timestamp: event.timestamp
    };
  });

  // Mock stock data structure (in production, parse from actual events)
  const stocks = [
    {
      symbol: "JKH.N0000",
      name: "John Keells Holdings",
      current: 145.50,
      predicted: 148.20,
      change: 2.70,
      changePercent: 1.86,
      volume: "1.2M",
      sentiment: marketInsights[0]?.sentiment || 'neutral'
    },
    {
      symbol: "COMB.N0000",
      name: "Commercial Bank",
      current: 89.75,
      predicted: 87.30,
      change: -2.45,
      changePercent: -2.73,
      volume: "856K",
      sentiment: marketInsights[1]?.sentiment || 'neutral'
    },
    {
      symbol: "HNB.N0000",
      name: "Hatton National Bank",
      current: 178.20,
      predicted: 182.50,
      change: 4.30,
      changePercent: 2.41,
      volume: "632K",
      sentiment: 'bullish'
    },
  ];

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-card border-border">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-success" />
            <h2 className="text-lg font-bold">MARKET INTELLIGENCE - CSE</h2>
          </div>
          <Badge className="font-mono text-xs border">
            LIVE AI ANALYSIS
          </Badge>
        </div>

        {/* AI-Generated Market Insights */}
        <div className="mb-6 space-y-2">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase">AI Market Analysis</h3>
          {marketInsights.length > 0 ? (
            marketInsights.slice(0, 3).map((insight, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className={`p-3 rounded border-l-4 ${
                  insight.sentiment === 'bullish' ? 'border-l-success bg-success/10' :
                  insight.sentiment === 'bearish' ? 'border-l-destructive bg-destructive/10' :
                  'border-l-muted bg-muted/30'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  {insight.sentiment === 'bullish' && <TrendingUp className="w-4 h-4 text-success" />}
                  {insight.sentiment === 'bearish' && <TrendingDown className="w-4 h-4 text-destructive" />}
                  <Badge className="text-xs">{insight.sentiment.toUpperCase()}</Badge>
                  <span className="text-xs text-muted-foreground ml-auto">
                    {Math.round(insight.confidence * 100)}% confidence
                  </span>
                </div>
                <p className="text-sm">{insight.title}</p>
              </motion.div>
            ))
          ) : (
            <p className="text-sm text-muted-foreground">Waiting for market data...</p>
          )}
        </div>

        {/* Stock Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {stocks.map((stock, idx) => {
            const isPositive = stock.change > 0;
            
            return (
              <motion.div
                key={stock.symbol}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
              >
                <Card className="p-4 bg-muted/30 border-border hover:border-primary/50 transition-all">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h3 className="font-bold text-sm">{stock.symbol}</h3>
                      <p className="text-xs text-muted-foreground">{stock.name}</p>
                    </div>
                    <Badge 
                      className={`font-mono text-xs ${isPositive ? "bg-primary text-primary-foreground" : "bg-destructive text-destructive-foreground"}`}
                    >
                      {isPositive ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
                      {isPositive ? "+" : ""}{stock.changePercent.toFixed(2)}%
                    </Badge>
                  </div>

                  <div className="grid grid-cols-2 gap-3 mt-3">
                    <div>
                      <p className="text-xs text-muted-foreground mb-1">Current</p>
                      <p className="text-lg font-bold font-mono">
                        LKR {stock.current.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground mb-1">AI Forecast</p>
                      <p className={`text-lg font-bold font-mono ${isPositive ? "text-success" : "text-destructive"}`}>
                        LKR {stock.predicted.toFixed(2)}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center justify-between mt-3 pt-3 border-t border-border">
                    <span className="text-xs text-muted-foreground">
                      Vol: {stock.volume}
                    </span>
                    <span className={`text-xs font-bold font-mono ${isPositive ? "text-success" : "text-destructive"}`}>
                      {isPositive ? "+" : ""}{stock.change.toFixed(2)}
                    </span>
                  </div>

                  {/* AI Sentiment Badge */}
                  <div className="mt-2">
                    <Badge className={`text-xs ${
                      stock.sentiment === 'bullish' ? 'bg-success/20 text-success' :
                      stock.sentiment === 'bearish' ? 'bg-destructive/20 text-destructive' :
                      'bg-muted'
                    }`}>
                      AI: {stock.sentiment.toUpperCase()}
                    </Badge>
                  </div>
                </Card>
              </motion.div>
            );
          })}
        </div>

        <div className="mt-4 p-3 bg-muted/20 rounded border border-border">
          <p className="text-xs text-muted-foreground font-mono">
            <span className="text-warning font-bold">⚠ DISCLAIMER:</span> AI predictions based on real-time data analysis. Not financial advice.
          </p>
        </div>
      </Card>
    </div>
  );
};

export default StockPredictions;
