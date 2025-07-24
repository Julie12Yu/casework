import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from collections import Counter

def load_and_preprocess_data(json_file_path):
    """
    Load JSON data and extract summaries and categories
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Debug: Print the structure to understand the format
    print(f"Data type: {type(data)}")
    
    # Handle different JSON structures
    if isinstance(data, dict):
        print("Data is a dictionary. Looking for the list of cases...")
        print(f"Top-level keys: {list(data.keys())}")
        
        # Try to find the list of cases within the dictionary
        cases_list = None
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                cases_list = value
                print(f"Found cases list under key: '{key}'")
                break
        
        if cases_list is None:
            # If no list found, maybe each key is a case
            cases_list = list(data.values())
            print("Treating dictionary values as individual cases")
        
        data = cases_list
    
    elif isinstance(data, list):
        print(f"Data is a list with {len(data)} items")
    
    # Show structure of first item
    if len(data) > 0:
        print(f"First item type: {type(data[0])}")
        if isinstance(data[0], str):
            print(f"First item (string): '{data[0][:200]}...'")
            print(f"First item length: {len(data[0])}")
            print(f"First item repr: {repr(data[0][:100])}")
            
            # Check if strings are empty or whitespace
            non_empty_items = [item for item in data if item.strip()]
            print(f"Non-empty items: {len(non_empty_items)} out of {len(data)}")
            
            if len(non_empty_items) > 0:
                print(f"First non-empty item: {repr(non_empty_items[0][:200])}")
                
                # Try to parse each string as JSON
                print("Attempting to parse strings as JSON...")
                parsed_data = []
                for i, item in enumerate(data[:5]):  # Show first 5 for debugging
                    if item.strip():  # Only try non-empty strings
                        try:
                            parsed_item = json.loads(item)
                            parsed_data.append(parsed_item)
                            print(f"Successfully parsed item {i}")
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse item {i}: {e}")
                            print(f"Item content: {repr(item[:100])}")
                            continue
                    else:
                        print(f"Skipping empty item {i}")
                
                # Parse all items if some succeeded
                if len(parsed_data) > 0:
                    parsed_data = []
                    for item in data:
                        if item.strip():
                            try:
                                parsed_item = json.loads(item)
                                parsed_data.append(parsed_item)
                            except json.JSONDecodeError:
                                continue
                
                data = parsed_data
                print(f"Successfully parsed {len(data)} JSON objects")
                if len(data) > 0:
                    print(f"First parsed item keys: {list(data[0].keys())}")
                    print(f"First parsed item: {data[0]}")
            else:
                print("All items appear to be empty strings")
                data = []
        elif isinstance(data[0], dict):
            print(f"First item keys: {list(data[0].keys())}")
            print(f"First item sample: {str(data[0])[:300]}...")
    
    summaries = []
    categories = []
    titles = []
    
    for case in data:
        # Handle the case structure based on your sample
        if isinstance(case, dict):
            # Get title - try different possible field names
            title = case.get('title') or case.get('pdftitleofcase') or case.get('name') or 'Unknown Title'
            titles.append(title)
            
            # Get summary - try different possible field names  
            summary = case.get('summary') or case.get('casesummary') or case.get('description') or 'No summary'
            summaries.append(summary)
            
            # Get category
            if 'category' in case:
                cat_data = case['category']
                if isinstance(cat_data, dict) and 'category' in cat_data:
                    cat_list = cat_data['category']
                    if isinstance(cat_list, list):
                        categories.append(cat_list[0])
                    else:
                        categories.append(cat_list)
                else:
                    categories.append(str(cat_data))
            else:
                categories.append('Unknown Category')
        else:
            print(f"Warning: Unexpected case format: {type(case)}")
    
    print(f"Processed {len(summaries)} cases")
    print(f"Categories found: {set(categories)}")
    
    return summaries, categories, titles

def create_color_mapping():
    """
    Create a color mapping for each category
    """
    category_names = [
        'Consumer Protection',
        'Antitrust', 
        'IP Law',
        'Privacy and Data Protection',
        'Tort',
        'Justice and Equity',
        'Unrelated',
        'AI in Legal Proceedings'
    ]
    
    # Use a colorblind-friendly palette
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple  
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f'   # gray
    ]
    
    return dict(zip(category_names, colors))

def vectorize_summaries(summaries):
    """
    Convert text summaries to TF-IDF vectors
    """
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    vectors = vectorizer.fit_transform(summaries)
    return vectors.toarray(), vectorizer

def perform_tsne(vectors, perplexity=30, random_state=42):
    """
    Perform t-SNE dimensionality reduction
    """
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(vectors)-1),  # Ensure perplexity is valid
        random_state=random_state,
        init='pca',
        n_iter=1000
    )
    
    embeddings = tsne.fit_transform(vectors)
    return embeddings

def create_visualization(embeddings, categories, titles, color_mapping):
    """
    Create the t-SNE visualization with color coding
    """
    plt.figure(figsize=(15, 10))
    
    # Create scatter plot for each category
    for category in color_mapping.keys():
        mask = np.array(categories) == category
        if np.any(mask):
            plt.scatter(
                embeddings[mask, 0], 
                embeddings[mask, 1],
                c=color_mapping[category],
                label=f"{category} ({np.sum(mask)})",
                alpha=0.7,
                s=50
            )
    
    plt.title('t-SNE Visualization of Legal Cases by Category', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    
    # Customize legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
               fancybox=True, shadow=True)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

def print_statistics(categories):
    """
    Print basic statistics about the dataset
    """
    category_counts = Counter(categories)
    total_cases = len(categories)
    
    print("Dataset Statistics:")
    print(f"Total cases: {total_cases}")
    print("\nCategory distribution:")
    for category, count in category_counts.most_common():
        percentage = (count / total_cases) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")

def main(json_file_path, output_path=None, perplexity=30):
    """
    Main function to run the t-SNE visualization
    
    Parameters:
    json_file_path (str): Path to your JSON file
    output_path (str): Optional path to save the plot
    perplexity (int): t-SNE perplexity parameter (default: 30)
    """
    
    print("Loading and preprocessing data...")
    summaries, categories, titles = load_and_preprocess_data(json_file_path)
    
    print("Creating color mapping...")
    color_mapping = create_color_mapping()
    
    print("Vectorizing summaries...")
    vectors, vectorizer = vectorize_summaries(summaries)
    
    print("Performing t-SNE...")
    embeddings = perform_tsne(vectors, perplexity=perplexity)
    
    print("Creating visualization...")
    plt = create_visualization(embeddings, categories, titles, color_mapping)
    
    # Print statistics
    print_statistics(categories)
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return embeddings, categories, titles, vectorizer

# Example usage
if __name__ == "__main__":
    # Replace with your actual JSON file path
    json_file_path = "legal_cases.json"
    
    # Optional: specify output path to save the plot
    output_path = "legal_cases_tsne_visualization.png"
    
    # Run the visualization
    embeddings, categories, titles, vectorizer = main(
        json_file_path=json_file_path,
        output_path=output_path,
        perplexity=10  # Better for ~50 cases
    )
    
    # Optional: Print some example cases from each cluster
    print("\nExample cases by category:")
    unique_categories = list(set(categories))
    for category in unique_categories[:3]:  # Show first 3 categories as examples
        mask = np.array(categories) == category
        example_titles = np.array(titles)[mask][:2]  # Show 2 examples
        print(f"\n{category}:")
        for title in example_titles:
            print(f"  - {title}")