# ==============================================
# TCS Customer Segmentation & Recommendation System
# VSCode Implementation for Internship Submission
# ==============================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("🚀 TCS Customer Segmentation Project - VSCode")
print("=" * 50)

# STEP 1: Generate Raw Transaction Data (25K records)
print("\n📊 STEP 1: Generating 25K realistic transactions...")
np.random.seed(42)
n_transactions = 25000
customers = np.random.choice(range(1000), n_transactions)  # 1000 unique customers
dates = pd.date_range('2024-01-01', periods=n_transactions, freq='H')

raw_data = pd.DataFrame({
    'customer_id': customers,
    'transaction_date': dates,
    'amount': np.random.lognormal(3, 1, n_transactions).clip(1, 500),
    'merchant': np.random.choice(['Amazon', 'Flipkart', 'Myntra', 'Others'], n_transactions)
})

raw_data.to_csv('raw_transactions.csv', index=False)
print(f"✅ RAW DATA SAVED: {raw_data.shape[0]} rows, {raw_data.shape[1]} columns")

# STEP 2: Data Cleaning
print("\n🧹 STEP 2: Data Cleaning...")
df = pd.read_csv('raw_transactions.csv')
print(f"Raw shape: {df.shape}")

df = df.dropna()
df = df[df['amount'] > 0]
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
print(f"✅ CLEANED shape: {df.shape}")

# STEP 3: Feature Engineering (RFM + Behavioral Features)
print("\n🔧 STEP 3: RFM + Behavioral Feature Engineering...")

# RFM Features
customer_features = df.groupby('customer_id').agg({
    'transaction_date': ['count', 'max'],
    'amount': ['sum', 'mean']
}).round(2)

customer_features.columns = ['frequency', 'last_purchase', 'monetary', 'avg_amount']
customer_features['recency'] = (pd.Timestamp('2025-12-12') - customer_features['last_purchase']).dt.days

# Behavioral Features
customer_features['avg_hour'] = df.groupby('customer_id')['transaction_date'].apply(
    lambda x: x.dt.hour.mean()).values
customer_features['weekend_pct'] = df.groupby('customer_id').apply(
    lambda x: (x['transaction_date'].dt.weekday >= 5).mean()).values

# Final 5 features for K-Means
features = customer_features[['recency', 'frequency', 'monetary', 'avg_hour', 'weekend_pct']]
features = features.fillna(features.mean())

features.to_csv('customer_features.csv')
print(f"✅ FEATURES SAVED: {features.shape} (1000 customers x 5 features)")
print("\nFeature Statistics:")
print(features.describe())

# SAVE PROGRESS
print("\n💾 FILES CREATED:")
print("- raw_transactions.csv (25K transactions)")
print("- customer_features.csv (1000 customers x 5 features)")
print("\n🎉 STEPS 1-3 COMPLETE! Ready for K-Means Clustering")

print("\n" + "="*50)
print("✅ RUN SUCCESSFUL - Check VSCode Explorer for files")
 
# STEP 4: Standardization for K-Means
print("\n📏 STEP 4: Standardizing features for K-Means...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

print("✅ Standardization complete")
print(f"Scaled matrix shape: {X_scaled.shape}")

# STEP 5: Elbow Method to find optimal k
print("\n📈 STEP 5: Elbow Method (k from 1 to 10)...")

wcss = []
K_RANGE = range(1, 11)
for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_RANGE, wcss, marker='o', color='blue')
plt.title("Figure 2: Elbow Curve (Optimal k Selection)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.grid(True)
plt.axvline(x=5, color='red', linestyle='--', label='Chosen k=5')
plt.legend()
plt.tight_layout()
plt.savefig("elbow_curve.png")
plt.close()

print("✅ Elbow curve saved as elbow_curve.png")

# STEP 6: Final K-Means with k=5
print("\n🤖 STEP 6: Running K-Means with k=5...")

k_final = 5
kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_scaled)

# Attach cluster labels to features
features['cluster'] = clusters

# Optional: map cluster IDs to segment names
cluster_names = {
    0: "Loyal",
    1: "VIP",
    2: "At-Risk",
    3: "Occasional",
    4: "New"
}
features['segment_name'] = features['cluster'].map(cluster_names)

features.to_csv("customer_segments.csv")
print("✅ Customer segments saved to customer_segments.csv")

# STEP 7: Cluster size distribution (table + print)
print("\n📊 STEP 7: Final Cluster Distribution")
cluster_counts = features['segment_name'].value_counts().sort_values(ascending=False)
total_customers = len(features)

print("\nCluster Name | Count | Percentage")
for name, count in cluster_counts.items():
    pct = 100 * count / total_customers
    print(f"{name:10s} | {count:5d} | {pct:6.2f}%")

# Also save distribution as CSV for report
cluster_dist = cluster_counts.to_frame(name="count")
cluster_dist['percentage'] = 100 * cluster_dist['count'] / total_customers
cluster_dist.to_csv("cluster_distribution.csv")
print("\n✅ Cluster distribution saved to cluster_distribution.csv")

# STEP 8: Rule-based recommendation engine

def get_recommendation(segment_name):
    if segment_name == "VIP":
        return "Offer exclusive early-access deals, high-value bundles, and VIP loyalty rewards."
    elif segment_name == "Loyal":
        return "Send regular personalized offers, loyalty points, and cross-sell complementary products."
    elif segment_name == "At-Risk":
        return "Trigger win-back campaigns with strong discounts and reminder emails to re-engage."
    elif segment_name == "Occasional":
        return "Send gentle nudges, seasonal offers, and social-proof based recommendations."
    elif segment_name == "New":
        return "Provide onboarding discounts, best-seller recommendations, and welcome journeys."
    else:
        return "Send generic promotions and collect more behavior data."

features['recommendation'] = features['segment_name'].apply(get_recommendation)

# Save sample of 20 customers with segment + recommendation
sample_reco = features[['cluster', 'segment_name', 'recommendation']].head(20)
sample_reco.to_csv("sample_recommendations.csv")
print("\n✅ Sample recommendations saved to sample_recommendations.csv")
