"use client";

import { Card } from "../ui/card";
import { Badge } from "../ui/badge";
import { ShoppingBasket, TrendingUp, TrendingDown, Minus } from "lucide-react";

interface Commodity {
    name: string;
    price: number;
    unit: string;
    change: number;
    category: string;
}

interface CommodityPricesProps {
    commodityData?: Record<string, unknown> | null;
}

const CommodityPrices = ({ commodityData }: CommodityPricesProps) => {
    const commodities = (commodityData?.commodities as Commodity[]) || [];
    const summary = (commodityData?.summary as Record<string, number>) || {};
    const fetchedAt = commodityData?.fetched_at as string;

    // Show top 8 essential items
    const essentialItems = commodities.slice(0, 8);

    const getTrendIcon = (change: number) => {
        if (change > 0) return <TrendingUp className="w-3 h-3 text-destructive" />;
        if (change < 0) return <TrendingDown className="w-3 h-3 text-success" />;
        return <Minus className="w-3 h-3 text-muted-foreground" />;
    };

    const getChangeColor = (change: number) => {
        if (change > 0) return "text-destructive";
        if (change < 0) return "text-success";
        return "text-muted-foreground";
    };

    return (
        <Card className="p-4 bg-card border-border">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <div className="p-2 rounded-lg bg-green-500/20">
                        <ShoppingBasket className="w-5 h-5 text-green-500" />
                    </div>
                    <div>
                        <h3 className="font-bold text-sm">🛒 COMMODITIES</h3>
                        <p className="text-xs text-muted-foreground">Essential goods prices</p>
                    </div>
                </div>
                <div className="flex gap-1">
                    {summary.items_increased > 0 && (
                        <Badge className="bg-destructive/20 text-destructive text-xs">
                            ↑{summary.items_increased}
                        </Badge>
                    )}
                    {summary.items_decreased > 0 && (
                        <Badge className="bg-success/20 text-success text-xs">
                            ↓{summary.items_decreased}
                        </Badge>
                    )}
                </div>
            </div>

            <div className="grid grid-cols-2 gap-1.5">
                {essentialItems.map((item, idx) => (
                    <div key={idx} className="p-2 rounded bg-muted/30 border border-border">
                        <div className="flex items-center justify-between">
                            <span className="text-xs text-muted-foreground truncate flex-1">{item.name}</span>
                            {getTrendIcon(item.change)}
                        </div>
                        <div className="flex items-baseline gap-1 mt-0.5">
                            <span className="text-sm font-bold">Rs.{item.price}</span>
                            {item.change !== 0 && (
                                <span className={`text-xs ${getChangeColor(item.change)}`}>
                                    {item.change > 0 ? '+' : ''}{item.change}
                                </span>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            {fetchedAt && (
                <p className="text-xs text-muted-foreground mt-3 text-center">
                    Source: Consumer Affairs Authority
                </p>
            )}
        </Card>
    );
};

export default CommodityPrices;
