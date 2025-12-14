/**
 * TrendingTopics.tsx
 * Dashboard component for displaying trending topics and spike alerts
 */

import React, { useEffect, useState } from 'react';

interface RelatedFeed {
    summary: string;
    domain: string;
    timestamp: string;
    source: string;
}

interface TrendingTopic {
    topic: string;
    momentum: number;
    is_spike: boolean;
    count_current_hour?: number;
    avg_count?: number;
    related_feeds?: RelatedFeed[];
}

interface TrendingData {
    status: string;
    trending_topics: TrendingTopic[];
    spike_alerts: TrendingTopic[];
    total_trending: number;
    total_spikes: number;
}
// ... (rest of imports and component setup unchanged)

// ... (inside the map function for rendering topics)
data.trending_topics.slice(0, 8).map((topic, idx) => (
    <div
        key={idx}
        className={`flex flex-col p-3 rounded-xl ${getMomentumBg(topic.momentum)} border border-gray-700/30 transition-all hover:scale-[1.02]`}
    >
        <div className="flex items-center justify-between w-full">
            <div className="flex items-center gap-3">
                <span className="text-lg font-bold text-gray-500">#{idx + 1}</span>
                <div>
                    <p className="font-semibold text-white capitalize">{topic.topic}</p>
                    <p className="text-xs text-gray-400">
                        {topic.is_spike ? '🔥 Spiking' : 'Trending'}
                    </p>
                </div>
            </div>
            <div className="text-right">
                <p className={`text-lg font-bold ${getMomentumColor(topic.momentum)}`}>
                    {topic.momentum.toFixed(0)}x
                </p>
                <p className="text-xs text-gray-500">momentum</p>
            </div>
        </div>

        {/* Related Feeds Context */}
        {topic.related_feeds && topic.related_feeds.length > 0 && (
            <div className="mt-3 pl-3 border-l-2 border-gray-600/30 space-y-2">
                {topic.related_feeds.map((feed, fIdx) => (
                    <div key={fIdx} className="text-xs text-gray-300/80 leading-relaxed">
                        <span className="text-gray-500 font-medium text-[10px] uppercase tracking-wider mr-2">[{feed.domain}]</span>
                        {feed.summary.length > 100 ? feed.summary.substring(0, 100) + '...' : feed.summary}
                    </div>
                ))}
            </div>
        )}
    </div>
))
                )}
            </div >

    {/* Footer */ }
    < div className = "mt-4 pt-4 border-t border-gray-700/50" >
        <p className="text-xs text-gray-500 text-center">
            Momentum = current hour mentions / avg last 6 hours
        </p>
            </div >
        </div >
    );
};

export default TrendingTopics;
