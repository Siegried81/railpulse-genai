# Text-to-SQL evaluation

Run on 2026-11-23 08:01, LLM_PROVIDER `groq`, SQL written by: groq · openai/gpt-oss-120b.
Execution accuracy: the model's query and the reference query are both run on
`data/railpulse_ai.db` and their results compared (see `scripts/eval_text_to_sql.py`).

| Subset | Passed |
|---|---|
| All cases | 40/40 (100%) |
| Not in the few-shot examples | 33/33 (100%) |
| Copied or paraphrased from the few-shot | 7/7 (100%) |

## Failures

None.
## All cases

| Case | Few-shot | Result |
|---|---|---|
| count_all | no | pass |
| avg_delay_all | no | pass |
| on_time_pct | yes | pass |
| busiest_station | no | pass |
| max_delay | no | pass |
| severe_count | no | pass |
| count_per_severity | no | pass |
| avg_bxl_central | no | pass |
| avg_anvers_today | yes | pass |
| worst_weekday | no | pass |
| count_on_date | no | pass |
| avg_per_hour | no | pass |
| worst5_stations | no | pass |
| distinct_stations | no | pass |
| distinct_vehicles | no | pass |
| avg_weekend | no | pass |
| over_30_min | no | pass |
| category_ranking | yes | pass |
| ic_count | no | pass |
| worst_date | no | pass |
| on_time_bxl_midi | no | pass |
| compare_two_brussels | yes | pass |
| hour_most_delayed | yes | pass |
| gand_morning_peak | no | pass |
| avg_last_7_days | no | pass |
| liege_top3 | no | pass |
| stations_over_5min | no | pass |
| date_range | no | pass |
| canceled_total | no | pass |
| severe_share_namur | no | pass |
| best_station | no | pass |
| avg_per_date | no | pass |
| monday_count | no | pass |
| zero_delay | no | pass |
| evening_avg | no | pass |
| no_weather | no | pass |
| no_direction | yes | pass |
| no_wheelchair | yes | pass |
| no_ticket_price | no | pass |
| no_staff | no | pass |
