
import pandas as pd
import numpy as np

# load olist orders dataset
orders = pd.read_csv('data/olist_orders_dataset.csv')

# Display the first five rows
print("first 5 orders")
print(orders.head())

#check column namesand data types
print("/nDataset information")
print(orders.info())

#check for missing values
print("/nMissing values in each column")
print(orders.isnull().sum())


# STEP 2: DATA CLEANING

print("\n========== DATA CLEANING ==========")

# 1. Check dataset shape
print("\nDataset Shape:")
print(orders.shape)

# 2. Check duplicate rows
print("\nDuplicate Rows:")
print(orders.duplicated().sum())

# 3. Check missing values
print("\nMissing Values:")
print(orders.isnull().sum())

# 4. Check order status
print("\nOrder Status:")
print(orders["order_status"].value_counts())

# 5. Convert date columns into datetime format
date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for column in date_columns:
    orders[column] = pd.to_datetime(
        orders[column],
        errors="coerce"
    )

print("\nDate Conversion Completed!")

# 6. Check updated data types
print("\nUpdated Data Types:")
print(orders.dtypes)


# STEP 3: MERGING DATASETS

print("\n========== MERGING DATASETS ==========")

# Load order items
items = pd.read_csv("data/olist_order_items_dataset.csv")

# Load payments
payments = pd.read_csv("data/olist_order_payments_dataset.csv")

# Load reviews
reviews = pd.read_csv("data/olist_order_reviews_dataset.csv")

# Load customers
customers = pd.read_csv("data/olist_customers_dataset.csv")

# Combine item details by order
item_summary = items.groupby("order_id", as_index=False).agg(
    item_value=("price", "sum"),
    freight_value=("freight_value", "sum")
)

# Calculate total order value
item_summary["order_value"] = (
    item_summary["item_value"] + item_summary["freight_value"]
)

# Combine payment details by order
payment_summary = payments.groupby("order_id", as_index=False).agg(
    payment_value=("payment_value", "sum")
)

# Combine review details by order
review_summary = reviews.groupby("order_id", as_index=False).agg(
    review_score=("review_score", "mean")
)

# Merge datasets
analysis = orders.merge(
    item_summary, on="order_id", how="left"
)

analysis = analysis.merge(
    payment_summary, on="order_id", how="left"
)

analysis = analysis.merge(
    review_summary, on="order_id", how="left"
)

analysis = analysis.merge(
    customers[["customer_id", "customer_state"]],
    on="customer_id", how="left"
)

# Display final analysis table
print("\nAnalysis Table Created Successfully!")
print("\nTable Shape:")
print(analysis.shape)

print("\nFirst 5 Rows:")
print(analysis.head())

# Save the merged dataset
analysis.to_csv("olist_analysis.csv", index=False)

print("\nFile Saved: olist_analysis.csv")


# STEP 4: EXPLORATORY DATA ANALYSIS

print("\n========== STEP 4: DATA ANALYSIS ==========")

# 1. Basic information
print("\nTotal Orders:")
print(analysis["order_id"].nunique())

# 2. Average order value
print("\nAverage Order Value:")
print(round(analysis["order_value"].mean(), 2))

# 3. Highest order value
print("\nHighest Order Value:")
print(round(analysis["order_value"].max(), 2))

# 4. Lowest order value
print("\nLowest Order Value:")
print(round(analysis["order_value"].min(), 2))

# 5. Average review score
print("\nAverage Review Score:")
print(round(analysis["review_score"].mean(), 2))

# 6. Missing values
print("\nMissing Values:")
print(analysis.isnull().sum())



# STEP 5: DATA VISUALIZATION

import matplotlib.pyplot as plt

print("\n========== STEP 5: GRAPHS ==========")

# Remove missing and negative order values
order_values = analysis["order_value"].dropna()
order_values = order_values[order_values >= 0]

# Create graph
plt.figure(figsize=(8, 5))

plt.hist(order_values, bins=50)

plt.title("Olist Order Value Distribution")
plt.xlabel("Order Value")
plt.ylabel("Number of Orders")

plt.tight_layout()

# Save graph
plt.savefig("order_value_graph.png")

# Display graph
plt.show()

print("\nGraph created and saved successfully!")


# STEP 6: DELIVERY TIME ANALYSIS

print("\n========== DELIVERY TIME ANALYSIS ==========")

# Calculate delivery time in days
analysis["delivery_days"] = (
    analysis["order_delivered_customer_date"]
    - analysis["order_purchase_timestamp"]
).dt.total_seconds() / (24 * 60 * 60)

# Display average delivery time
print("\nAverage Delivery Time (Days):")
print(round(analysis["delivery_days"].mean(), 2))

# Display average customer review
print("\nAverage Review Score:")
print(round(analysis["review_score"].mean(), 2))

# Prepare data for graph
delivery_data = analysis.dropna(
    subset=["delivery_days", "review_score"]
)

# Remove invalid delivery times
delivery_data = delivery_data[
    delivery_data["delivery_days"] >= 0
]

# Create graph
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.scatter(
    delivery_data["delivery_days"],
    delivery_data["review_score"],
    alpha=0.3
)

plt.title("Delivery Time vs Customer Review Score")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Review Score")

plt.tight_layout()
plt.savefig("delivery_review_graph.png")
plt.show()

print("\nDelivery analysis completed successfully!")


# STEP 7: SQL ANALYSIS

import sqlite3

print("\n========== STEP 7: SQL ANALYSIS ==========")

# Create database
connection = sqlite3.connect("olist_database.db")

# Save analysis table into database
analysis.to_sql(
    "orders_analysis",
    connection,
    if_exists="replace",
    index=False
)

# QUERY 1: Total orders by status
query1 = """
SELECT order_status, COUNT(*) AS total_orders
FROM orders_analysis
GROUP BY order_status
ORDER BY total_orders DESC;
"""

print("\nQUERY 1: Orders by Status")
print(pd.read_sql_query(query1, connection))

# QUERY 2: Average order value by state
query2 = """
SELECT customer_state,
       ROUND(AVG(order_value), 2) AS average_order_value
FROM orders_analysis
GROUP BY customer_state
ORDER BY average_order_value DESC;
"""

print("\nQUERY 2: Average Order Value by State")
print(pd.read_sql_query(query2, connection))

# QUERY 3: Average review score by order status
query3 = """
SELECT order_status,
       ROUND(AVG(review_score), 2) AS average_review_score
FROM orders_analysis
GROUP BY order_status;
"""

print("\nQUERY 3: Review Score by Status")
print(pd.read_sql_query(query3, connection))

# QUERY 4: Rank states by number of orders (Window Function)
query4 = """
SELECT customer_state,
       COUNT(*) AS total_orders,
       DENSE_RANK() OVER (
           ORDER BY COUNT(*) DESC
       ) AS state_rank
FROM orders_analysis
GROUP BY customer_state
ORDER BY state_rank;
"""

print("\nQUERY 4: State Ranking")
print(pd.read_sql_query(query4, connection))

connection.close()

print("\nSQL ANALYSIS COMPLETED SUCCESSFULLY!")

# STEP 8: STATISTICAL ANALYSIS

print("\n========== STEP 8: STATISTICAL ANALYSIS ==========")

# 1. 95% Confidence Interval for Average Order Value

order_values = analysis["order_value"].dropna()
order_values = order_values[order_values >= 0]

mean_order_value = order_values.mean()
std_order_value = order_values.std()
sample_size = len(order_values)

standard_error = std_order_value / np.sqrt(sample_size)

# Approximate 95% confidence interval
lower_limit = mean_order_value - 1.96 * standard_error
upper_limit = mean_order_value + 1.96 * standard_error

print("\n95% CONFIDENCE INTERVAL")
print("Average Order Value:", round(mean_order_value, 2))
print("Lower Limit:", round(lower_limit, 2))
print("Upper Limit:", round(upper_limit, 2))

# 2. Weekend vs Weekday Order Counts

analysis["order_purchase_timestamp"] = pd.to_datetime(
    analysis["order_purchase_timestamp"], errors="coerce"
)

analysis["day_name"] = analysis["order_purchase_timestamp"].dt.day_name()

analysis["day_type"] = np.where(
    analysis["order_purchase_timestamp"].dt.dayofweek >= 5,
    "Weekend",
    "Weekday"
)

day_counts = analysis.groupby("day_type")["order_id"].nunique()

print("\nWEEKEND VS WEEKDAY ORDERS")
print(day_counts)

print("\nStatistical Analysis Completed Successfully!")

# STEP 9: SQL CTE ANALYSIS

import sqlite3

connection = sqlite3.connect("olist_database.db")

cte_query = """
WITH state_orders AS (
    SELECT
        customer_state,
        COUNT(DISTINCT order_id) AS total_orders
    FROM orders_analysis
    GROUP BY customer_state
)
SELECT customer_state, total_orders
FROM state_orders
ORDER BY total_orders DESC
LIMIT 10;
"""

print("\n========== STEP 9: SQL CTE ANALYSIS ==========")
print(pd.read_sql_query(cte_query, connection))

connection.close()

print("\nCTE Analysis Completed Successfully!")





