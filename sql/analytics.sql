-- These queries are also registered as views by churn_analysis.database.

SELECT Contract AS contract,
       COUNT(*) AS customers,
       ROUND(AVG(Churn) * 100, 2) AS churn_rate_percent,
       ROUND(AVG(MonthlyCharges), 2) AS average_monthly_charges
FROM customers
GROUP BY Contract
ORDER BY churn_rate_percent DESC;

SELECT CASE
         WHEN tenure <= 3 THEN '0-3 months'
         WHEN tenure <= 12 THEN '4-12 months'
         WHEN tenure <= 24 THEN '13-24 months'
         ELSE '25+ months'
       END AS tenure_band,
       COUNT(*) AS customers,
       ROUND(AVG(Churn) * 100, 2) AS churn_rate_percent
FROM customers
GROUP BY tenure_band
ORDER BY MIN(tenure);

SELECT Contract AS contract,
       COUNT(*) AS customers,
       SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS observed_churners,
       ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_exposure
FROM customers
GROUP BY Contract
ORDER BY monthly_revenue_exposure DESC;
