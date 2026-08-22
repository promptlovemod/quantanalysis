#!/usr/bin/env python3
import csv
from collections import Counter
from pathlib import Path

p = Path('docs/qua3/sourcing_tracker.csv')
rows = list(csv.DictReader(p.open(encoding='utf-8', newline='')))

def filled(v):
    return bool(v and v.strip())

total = len(rows)
responses = sum(1 for r in rows if filled(r.get('response_date', '')))
intros = sum(1 for r in rows if filled(r.get('intro_date', '')))
design = sum(1 for r in rows if filled(r.get('design_date', '')))
technical = sum(1 for r in rows if filled(r.get('technical_date', '')))
finals = sum(1 for r in rows if filled(r.get('final_date', '')))

stage = Counter((r.get('stage') or 'unknown').strip() or 'unknown' for r in rows)
source = Counter((r.get('source_channel') or 'unknown').strip() or 'unknown' for r in rows)

def pct(n, d):
    return f"{(100*n/d):.1f}%" if d else '0.0%'

print('Hiring Funnel KPI Report')
print('========================')
print(f'Total prospects: {total}')
print(f'Responses: {responses} ({pct(responses,total)})')
print(f'Intro calls: {intros} ({pct(intros,total)})')
print(f'Design rounds: {design} ({pct(design,total)})')
print(f'Technical rounds: {technical} ({pct(technical,total)})')
print(f'Final rounds: {finals} ({pct(finals,total)})')
print('\nBy stage:')
for k,v in sorted(stage.items()):
    print(f'- {k}: {v}')
print('\nBy source:')
for k,v in sorted(source.items()):
    print(f'- {k}: {v}')
