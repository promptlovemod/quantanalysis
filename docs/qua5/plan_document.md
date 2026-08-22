# QUA-5: Core Data Persistence Layer Plan

## Objective
Implement idempotent historical + incremental ingestion with gate: no duplicates on rerun; quality report generated each run.

## Acceptance Criteria
- Historical data ingestion is idempotent (safe to rerun without duplicates)
- Incremental ingestion handles new data efficiently
- Quality report generated for each run with metrics and anomalies
- No duplicate records in storage after reruns
- Proper error handling and logging

## Implementation Steps

### Phase 1: Storage Foundation (Days 1-2)
1. Select storage solution (Parquet files, SQLite, or cloud storage)
2. Design schema for OHLCV data with proper partitioning
3. Implement basic read/write operations
4. Create connection/pool management

### Phase 2: Idempotent Historical Ingestion (Days 3-4)
1. Implement duplicate detection mechanism (based on ticker+date+timeframe)
2. Create upsert/merge functionality
3. Build batch ingestion pipeline for historical data
4. Add validation checks during ingestion

### Phase 3: Incremental Ingestion (Days 5-6)
1. Design checkpoint mechanism for tracking last ingested timestamp
2. Implement incremental fetch from providers
3. Handle data gaps and overlaps gracefully
4. Create merge logic for incremental updates

### Phase 4: Quality Reporting & Monitoring (Days 7-8)
1. Implement data quality checks (completeness, validity, consistency)
2. Generate quality reports with metrics (record counts, date ranges, anomalies)
3. Add logging and monitoring hooks
4. Create visualization/reporting interface if needed

### Phase 5: Testing & Optimization (Days 9-10)
1. Write comprehensive unit and integration tests
2. Test idempotency with multiple reruns
3. Performance testing for large datasets
4. Optimization and documentation

## Dependencies
- QUA-4 completion (Market Data API Integration)
- QUA-3 completion (branch protections enabled)

## Risks & Mitigations
1. **Storage corruption or data loss**
   - Mitigation: Implement backup and recovery procedures
   - Mitigation: Use atomic writes and transaction-like semantics
   
2. **Performance degradation with scale**
   - Mitigation: Proper indexing and partitioning
   - Mitigation: Batch processing optimization
   
3. **Schema evolution complexity**
   - Mitigation: Versioned schema design
   - Mitigation: Backward compatibility considerations

## Success Metrics
- Zero duplicate records after multiple ingestion runs
- Ingestion latency < 2 seconds per 1000 records
- Quality report generated for every run with <5% anomaly rate
- Storage efficiency > 80% compression ratio
