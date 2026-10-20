"""Plain-language wording for the dashboard."""

TIERS = {
    "High": ("High priority", "Strong, corroborated warning signs. Review first."),
    "Review": ("Worth a look", "Some warning signs. Review if capacity allows."),
    "Needs more data": ("Needs more data", "Too little history, or suspicion with nothing to back it up. Collect more records before judging."),
    "Low": ("Low priority", "No notable warning signs."),
}

EDGE_LABELS = {
    "OWNS": "owns", "DIRECTS": "is a director of", "ISSUES_INVOICE_TO": "sent invoices to",
    "REGISTERED_AT": "is registered at", "HAS_CONTACT": "uses phone", "TRANSFERS_TO": "paid",
    "HAS_ACCOUNT": "has account",
}

FEATURE_LABELS = {
    "n_inv_out": "Invoices issued", "n_inv_in": "Invoices received", "log_amt_out": "Value invoiced (log)",
    "log_amt_in": "Value received (log)", "round_share_out": "Share of round-number invoices",
    "mismatch_share_out": "Invoices for items outside its sector", "lag_mean": "Average days to get paid",
    "lag_std": "Variation in days to get paid", "flow_balance": "Money out vs money in (1 = equal)",
    "log_in_out_ratio": "Money in vs out (log ratio)", "median_dwell_days": "Typical days money stays before leaving",
    "short_dwell_share": "Share of payments sent on within 3 days", "n_cpty": "Number of trading partners",
    "cpty_hhi": "Dependence on few partners", "week_cv": "Unevenness of activity over time",
    "max_week_share": "Share of all activity in its busiest week", "dormancy_frac": "Share of weeks with no activity",
    "longest_dormant_frac": "Longest quiet stretch", "age_days": "Company age (days)",
    "active_span_days": "Days between first and last activity", "vol_vs_declared": "Trading volume vs declared size",
    "n_tx": "Number of payments", "in_deg": "Customers", "out_deg": "Suppliers",
    "reciprocal_partners": "Partners it both buys from and sells to", "pagerank": "Importance in the trade network",
    "clustering": "How interlinked its partners are", "cycles3": "Circular flows (3 companies)",
    "cycles4": "Circular flows (4 companies)", "tight_cycles": "Circular flows with near-identical amounts",
    "addr_group_size": "Other companies at same address", "contact_group_size": "Other companies with same phone",
    "idcluster_size": "Size of linked group (shared controller/address/phone)",
    "idcluster_internal_share": "Share of trade kept inside its linked group",
    "idcluster_density": "How tightly its linked group trades together",
    "ctrl_n_controllers": "Known owners/directors", "ctrl_max_reach": "Most companies one of its controllers is linked to",
    "ctrl_cluster_size": "Companies linked through shared controllers",
    "ctrl_cluster_internal_share": "Trade kept inside controller-linked group",
    "ev_n_links": "Supporting links on file", "ev_mean_conf": "Average link confidence",
    "ev_n_sources": "Independent evidence sources", "ev_recency_days": "Days since newest link",
    "ev_conflicts": "Conflicting owner records",
}

# code -> (title, what it means, innocent explanations to rule out)
CODE_INFO = {
    "TIGHT_CYCLE": ("Money going in circles",
                    "Invoices pass around a loop of 3-4 companies with almost the same amount each time. Round-tripping like this can fake turnover.",
                    "Genuine reciprocal trade between a few partners can look similar; check whether goods actually moved."),
    "ROUND_VALUES": ("Unusually round invoice amounts",
                     "A large share of its invoices are exact multiples of 1,000. Real invoices rarely land on round numbers this often.",
                     "Cash-heavy retail, fixed-price contracts and some wholesalers invoice in round amounts."),
    "PASS_THROUGH": ("Money in, straight back out",
                     "Almost everything it receives is sent on within a day or two. It looks like a conduit, not a business that keeps money.",
                     "Payment agents, wholesalers on thin margins and escrow-like services also pass funds quickly."),
    "DORMANT_THEN_SPIKE": ("Quiet, then a sudden burst",
                           "Months of little or no activity followed by a concentrated burst. Shell companies are often parked and then activated.",
                           "Seasonal businesses and newly launched products produce bursts too."),
    "SECTOR_MISMATCH": ("Sells things unrelated to its declared sector",
                        "Many of its invoices are for items that don't belong to the sector it registered under.",
                        "Diversified firms and trading companies legitimately sell across sectors."),
    "SHARED_CONTROLLER": ("One person behind several companies",
                          "A recorded owner or director is linked to three or more companies in this data.",
                          "Legitimate business groups and holding structures look the same; check whether the companies trade with each other at arm's length."),
    "INSULAR_CLUSTER": ("Trades mostly inside its own linked group",
                        "It is linked to a few other companies (shared controller, address or phone) and a large part of its trade stays inside that group.",
                        "Genuine supplier-customer groups under common ownership behave this way."),
    "SHARED_IDENTIFIER": ("Shares an address or phone number with others",
                          "Its registered address and/or phone number also belong to other companies.",
                          "Coworking spaces, registered-office services and accountants' addresses are shared by many unrelated firms."),
}
