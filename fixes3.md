# Analysis and Fixes for Unrealistic Dashboard Metrics

### Executive Summary
The investigation confirmed that several key metrics on the "Business" tab were not being calculated from real, scored data, leading to a "cooked" or static appearance. The primary issues were hardcoded placeholder logic in both the backend and frontend for "Fraud by Category" and "Fraud Exposure."

The code has now been modified to perform real-time aggregation directly on the `*.parquet` score files produced by the pipeline. The dashboard now reflects the true output of the model, ensuring data is realistic and derived from actual inference results, as requested in `table.md` (D-01).

---

### Detailed Findings & Fixes

#### 1. Issue: "Fraud by Category" $$$ values are static and identical.
*   **Reason:** The frontend JavaScript in `dashboard.html` was not using real data. It was programmatically distributing a *total* fraud count across the 14 categories using a fixed weighting scheme (giving 3x weight to `shopping_net`, `travel`, and `misc_net`). This created the illusion of data but was disconnected from the model's actual output.
*   **Fix Implemented:**
    1.  **Backend (`pods/backend/metrics.py`):** The `_collect_fraud_metrics` function was updated to perform a `groupby('category')` on the real scored data. It now calculates the `sum` of transaction amounts (`amt`) for each category flagged as high-risk.
    2.  **Frontend (`pods/backend/static/dashboard.html`):** The fake weighting logic in the JavaScript was removed entirely. The "Fraud by Category" chart is now populated directly from the new `fraud_by_category` data sent by the backend, displaying the actual dollar amounts at risk per category.

#### 2. Issue: "Fraud Exposure" value looks cooked.
*   **Reason:** The backend (`pods/backend/metrics.py`) was not summing the real dollar amounts of fraudulent transactions. It was using a hardcoded `avg_fraud_amt = 250.0` and multiplying it by the total number of flagged transactions.
*   **Fix Implemented:** The `_collect_fraud_metrics` function was modified to calculate the `total_exposure_usd` by summing the `amt` column for all transactions with a `fraud_score > 0.8`. This value is now passed to the business KPIs and displayed on the dashboard.

#### 3. Issue: "High Risk Alert List" looks static.
*   **Reason:** This is a symptom of a stalled pipeline, not a bug in the dashboard logic itself. The backend correctly reads the last 10 score files from `/data/scores`. If the `scoring` pod is not processing new data (due to the Triton crashes we resolved earlier), no new score files are written, and this list will never change.
*   **Fix Implemented:** No code change was needed for this component. By restoring the stability of the `triton` and `scoring` pods, new score files are generated, and the "High Risk Alert List" now updates with fresh data as intended.

#### 4. Issue: Fraud alert percentage is always the same (~6%).
*   **Reason:** This is not a bug but a reflection of the model's consistent performance on the synthetically generated data. The data generator (`gather.py`) and the scoring model are stable, leading to a highly consistent prediction rate on the input data.
*   **Fix Implemented:** No code change is required, as this is the *correct* behavior of the system. For a more dynamic demo, the data generator (`gather.py`) could be modified to introduce more randomness or concept drift, but the model and backend are functioning as designed.

#### 5. Issue: Realism of transaction data.
*   **Reason:** The data generator (`gather.py`) uses realistic mechanisms, including category-specific amount caps (`CATEGORY_MAX_AMT`) and lognormal distributions, which are appropriate for this type of data. The *perception* of the data being unrealistic was caused by the placeholder logic in the dashboard.
*   **Fix Implemented:** No changes were needed in the data generator. The fixes to the backend and frontend to display *real* aggregated data have resolved this issue, making the dashboard an accurate and realistic representation of the pipeline's output.
