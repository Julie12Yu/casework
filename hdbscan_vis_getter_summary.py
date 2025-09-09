import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from collections import Counter
import textwrap
import re
import hdbscan
from sklearn.metrics import silhouette_score

# ---------------------------
# Config
# ---------------------------
DIR_PATH = 'STEP 2: get embed/summaries/categorized_cases.json'
UMAP_EMBED = 'STEP 3: umap/summary/umap_embedding.npy'
TEXT_TYPE = "summary"
MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 10

# ---------------------------
# Helpers
# ---------------------------
def load_and_process_data():
    """Load JSON data and process it for visualization (new JSON format)."""
    with open(DIR_PATH, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    processed_cases = []
    for count, (casename, metadata) in enumerate(data.items(), start=1):
        try:
            summary = metadata.get("summary", casename)
            categories_obj = metadata.get("categories", {})
            categories = categories_obj.get("category", ["Unknown"])
            ai_material = categories_obj.get("ai_material", False)
            main_category = categories[0] if isinstance(categories, list) and categories else "Unknown"
        except Exception as e:
            print(f"COUNT {count} – Warning: could not parse {casename}: {e}")
            summary = casename
            main_category = "Unknown"
            ai_material = False
            categories = ["Unknown"]

        case_info = {
            "title": casename,
            "summary": summary,
            "main_category": main_category,
            "ai_material": ai_material,
            "all_categories": categories,
        }
        processed_cases.append(case_info)
    
    df = pd.DataFrame(processed_cases)
    return df, processed_cases


def wrap_text(text, width=60):
    """Helper function to wrap text for better display"""
    return '<br>'.join(textwrap.wrap(str(text), width=width))

def perform_hdbscan(embedding):
    """Perform HDBSCAN clustering on UMAP embeddings."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        gen_min_span_tree=True
    )
    labels = clusterer.fit_predict(embedding)
    return labels, clusterer

def calculate_hdbscan_silhouette(X, labels):
    """
    Calculate silhouette score for HDBSCAN clustering.
    Returns None if there are fewer than 2 clusters (ignoring noise).
    """
    # Filter out noise points (-1)
    mask = labels != -1
    if mask.sum() == 0:
        print("All points classified as noise, cannot compute silhouette.")
        return None
    
    unique_clusters = set(labels[mask])
    if len(unique_clusters) < 2:
        print(f"Only {len(unique_clusters)} cluster found (ignoring noise). Cannot compute silhouette.")
        return None

    score = silhouette_score(X[mask], labels[mask])
    print(f"HDBSCAN silhouette score: {score:.4f} (computed on {mask.sum()} points)")
    return score


def create_interactive_visualization_hdbscan(
    embedding, titles, ai_materials, all_categories, hdbscan_labels,
    silhouette_score_val=None, embedding_method="Legal-BERT"
):
    """Create UMAP visualization with HDBSCAN clustering on top, with silhouette score text."""
    output_file = ('hdbscan_' + embedding_method + '_' + TEXT_TYPE + '.png')

    # Convert -1 (noise) to string
    labels_str = ["Noise" if l == -1 else f"Cluster {l}" for l in hdbscan_labels]

    df_plot = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'cluster': labels_str,
        'title': titles,
        'ai_material': ai_materials,
        'all_categories': all_categories,
        'title_wrapped': [t[:100] + '...' if len(t) > 100 else t for t in titles]
    })

    unique_clusters = sorted(set(labels_str))

    # Assign colors
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Set3
    color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}

    fig = go.Figure()

    for cluster in unique_clusters:
        mask = df_plot['cluster'] == cluster
        cluster_data = df_plot[mask]

        if len(cluster_data) > 0:
            customdata = []
            for _, row in cluster_data.iterrows():
                ai_status = "Yes" if row['ai_material'] else "No"
                all_cats = ", ".join(row['all_categories']) if isinstance(row['all_categories'], list) else str(row['all_categories'])
                customdata.append([row['cluster'], ai_status, all_cats, wrap_text(row['title'], width=80)])

            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                marker=dict(
                    color=color_map[cluster],
                    size=12,
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                name=f"{cluster} ({len(cluster_data)})",
                customdata=customdata,
                hovertemplate='<b>Case:</b><br>%{customdata[3]}<br><br>' +
                              '<b>Cluster:</b> %{customdata[0]}<br>' +
                              '<b>All Categories:</b> %{customdata[2]}<br>' +
                              '<b>AI Material:</b> %{customdata[1]}<br>' +
                              '<extra></extra>'
            ))

    # Add silhouette score info
    subtitle = embedding_method
    if silhouette_score_val is not None:
        subtitle += f" | Silhouette score: {silhouette_score_val:.3f}"

    fig.update_layout(
        title={
            'text': f'Legal Cases Visualization with HDBSCAN<br><sub>{subtitle}</sub>',
            'x': 0.5,
            'font': {'size': 36}
        },
        xaxis_title=(embedding_method + ' Dimension 1'),
        yaxis_title=(embedding_method + ' Dimension 2'),
        font=dict(size=20),
        legend=dict(
            title='HDBSCAN Clusters',
            title_font=dict(size=24),
            font=dict(size=18),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=2
        ),
        width=2560,
        height=1440,
        margin=dict(r=400, l=100, t=150, b=100),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="black",
            font_size=20,
            font_family="Arial",
            align="left"
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    fig.write_image(output_file, width=2560, height=1440, scale=2)
    print(f"HDBSCAN visualization saved to: {output_file}")

    return fig



# ---------------------------
# Main entry
# ---------------------------
def main():
    print("Loading and processing data...")
    df, processed_cases = load_and_process_data()

    titles = [case['title'] for case in processed_cases]
    ai_materials = [case['ai_material'] for case in processed_cases]
    all_categories = [case['all_categories'] for case in processed_cases]

    print(f"Loaded {len(titles)} cases")

    embedding_umap = np.load(UMAP_EMBED)

    # Run HDBSCAN
    print("Running HDBSCAN clustering...")
    hdbscan_labels, clusterer = perform_hdbscan(embedding_umap)

    silhouette_score_hdb = calculate_hdbscan_silhouette(embedding_umap, hdbscan_labels)

    # Create visualization
    print("Creating interactive visualization with HDBSCAN...")
    fig = create_interactive_visualization_hdbscan(
        embedding_umap, titles, ai_materials, all_categories, hdbscan_labels,
        silhouette_score_val=silhouette_score_hdb
    )

    return embedding_umap, titles, hdbscan_labels, fig


if __name__ == "__main__":
    try:
        embedding_umap, titles, hdbscan_labels, fig = main()
        print("Interactive HDBSCAN visualization completed successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find the file '{DIR_PATH}'")
        print("Please make sure the file exists and the path is correct.")
