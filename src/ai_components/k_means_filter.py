import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib
import matplotlib.dates as mdates

def assign_regime_labels(model, features):
    """
    Assign meaningful labels to clusters based on their characteristics
    Returns:
    - label_mapping (dict): Mapping from cluster number to regime label
    - labeled_data (DataFrame): Features with 'regime' column
    """
    # Analyze cluster characteristics
    cluster_stats = features.groupby(model.labels_).agg({
        'returns': 'mean',
        'volatility': 'mean',
        'close': 'mean',
        'price_range': 'mean'
    }).reset_index()

    # Sort clusters by key metrics for label assignment
    bullish_candidate = cluster_stats.sort_values('returns', ascending=False).iloc[0]['index']
    bearish_candidate = cluster_stats.sort_values('returns').iloc[0]['index']

    # Identify ranging/unfavorable regime
    range_threshold = cluster_stats['volatility'].median()
    ranging_candidates = cluster_stats[cluster_stats['volatility'] < range_threshold]['index']

    # Create label mapping
    label_mapping = {
        int(bullish_candidate): "Bullish Trend",
        int(bearish_candidate): "Bearish Trend"
    }

    # Assign remaining clusters to ranging/unfavorable
    for cluster in model.labels_:
        if cluster not in label_mapping:
            label_mapping[cluster] = "Unfavorable/Ranging"

    # Apply labels to data
    features['regime'] = [label_mapping[label] for label in model.labels_]

    return label_mapping, features


def enhanced_visualization(labeled_data):
    """Plot regime distribution with proper labels matching screenshot style"""
    plt.figure(figsize=(18, 8))

    # Create colormap matching your existing labels
    color_map = {
        "Bullish Trend": '#4CAF50',  # Green
        "Bearish Trend": '#F44336',  # Red
        "Unfavorable/Ranging": '#9E9E9E'  # Gray
    }

    # Plot each regime with proper formatting
    for regime, color in color_map.items():
        regime_data = labeled_data[labeled_data['regime'] == regime]
        plt.plot(regime_data.index,
                 regime_data['close'],
                 color=color,
                 linewidth=1.5,
                 label=regime)

    # Format x-axis like the screenshot
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')

    # Set labels and title matching the screenshot
    plt.title('Market Regimes (KMeans Clustering)', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12, labelpad=10)
    plt.ylabel('Close Price', fontsize=12, labelpad=10)

    # Add regime label on right axis
    plt.gca().yaxis.set_label_position("right")
    plt.gca().set_ylabel('Regime', rotation=-90, labelpad=20, fontsize=12)

    # Configure grid and borders
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Add legend in best location
    plt.legend(title="Regimes:",
               loc='upper left',
               frameon=False,
               fontsize=10)

    plt.tight_layout()
    plt.show()


def load_data(file_path):
    """Load and preprocess market data"""
    df = pd.read_csv(file_path, parse_dates=['time'])
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close']]  # Focus on OHLC
    return df.dropna()


def engineer_features(df, window=14):
    """Create technical features for clustering"""
    features = df.copy()

    # Price changes
    features['returns'] = df['close'].pct_change()
    features['volatility'] = df['close'].rolling(window).std()

    # Range features
    features['price_range'] = df['high'] - df['low']
    features['body_range'] = abs(df['open'] - df['close'])

    return features.dropna()


def train_model(data, n_clusters=3):
    """Train K-Means model with optimized parameters"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    model = KMeans(n_clusters=n_clusters,
                   init='k-means++',
                   n_init=10,
                   random_state=42)
    model.fit(scaled_data)

    # Save artifacts
    joblib.dump(model, r'C:\Users\User\PycharmProjects\PythonProject\models/kmeans_model.pkl')
    joblib.dump(scaler, r'C:\Users\User\PycharmProjects\PythonProject/models/scaler.pkl')

    return model, scaler


def evaluate_model(model, data):
    """Calculate clustering metrics"""
    print(f"Inertia: {model.inertia_:.2f}")
    print(f"Silhouette Score: {silhouette_score(data, model.labels_):.2f}")


def visualize_clusters(df, labels):
    """Plot cluster distributions over time"""
    plt.figure(figsize=(15, 7))

    for cluster in np.unique(labels):
        cluster_data = df[labels == cluster]
        plt.scatter(cluster_data.index,
                    cluster_data['close'],
                    label=f'Regime {cluster}',
                    alpha=0.7)

    plt.title('Market Regime Identification')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()



# Updated main execution
if name == "main":
    raw_data = load_data(r'C:\Users\User\PycharmProjects\PythonProject\data/GBPUSDm_D1_train.csv')
    features = engineer_features(raw_data)
    model, scaler = train_model(features)
    evaluate_model(model, scaler.transform(features))

    # New labeling functionality
    label_mapping, labeled_data = assign_regime_labels(model, features)
    print("\nCluster Label Mapping:")
    for cluster, label in label_mapping.items():
        print(f"Cluster {cluster}: {label}")

    enhanced_visualization(labeled_data)