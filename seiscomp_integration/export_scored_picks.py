#!/usr/bin/env python3
"""
Export recovar-scored picks from the SeisComP database to a CSV file.

Usage:
    python3 export_scored_picks.py
    python3 export_scored_picks.py --output my_picks.csv
    python3 export_scored_picks.py --start 2024-01-01 --end 2024-12-31
    python3 export_scored_picks.py --min-score 0.5
"""

import argparse
import csv
import sys
import pymysql as MySQLdb

DB_HOST = "localhost"
DB_USER = "sysop"
DB_PASS = "sysop"
DB_NAME = "seiscomp"

QUERY = """
SELECT
    po.publicID                                          AS pick_id,
    CONCAT(p.time_value, '.', LPAD(p.time_value_ms, 3, '0')) AS pick_time,
    p.waveformID_networkCode                             AS net,
    p.waveformID_stationCode                             AS sta,
    p.waveformID_locationCode                            AS loc,
    p.waveformID_channelCode                             AS cha,
    CAST(REPLACE(c.text, 'recovar_score:', '') AS DECIMAL(6,4)) AS score
FROM Pick p
JOIN PublicObject po ON po._oid = p._oid
JOIN Comment c       ON c._parent_oid = p._oid
WHERE c.text LIKE 'recovar_score:%%'
{time_filter}
ORDER BY p.time_value
"""


def parse_args():
    p = argparse.ArgumentParser(description="Export recovar-scored picks to CSV.")
    p.add_argument("--output", default="scored_picks.csv", help="Output CSV file (default: scored_picks.csv)")
    p.add_argument("--start",  metavar="YYYY-MM-DD", help="Only picks on or after this date")
    p.add_argument("--end",    metavar="YYYY-MM-DD", help="Only picks on or before this date")
    p.add_argument("--min-score", type=float, metavar="0-1", help="Only picks with score >= this value")
    return p.parse_args()


def build_query(args):
    clauses = []
    params = []
    if args.start:
        clauses.append("AND p.time_value >= %s")
        params.append(args.start)
    if args.end:
        clauses.append("AND p.time_value <= %s")
        params.append(args.end)
    if args.min_score is not None:
        clauses.append("AND CAST(REPLACE(c.text, 'recovar_score:', '') AS DECIMAL(6,4)) >= %s")
        params.append(args.min_score)
    return QUERY.format(time_filter="\n".join(clauses)), params


def main():
    args = parse_args()
    query, params = build_query(args)

    try:
        conn = MySQLdb.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)
    except MySQLdb.Error as e:
        sys.exit(f"DB connection failed: {e}")

    with conn:
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        columns = [d[0] for d in cur.description]

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    print(f"Exported {len(rows)} scored pick(s) to {args.output}")


if __name__ == "__main__":
    main()
