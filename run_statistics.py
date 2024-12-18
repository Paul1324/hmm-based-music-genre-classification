import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(json_path):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_parameter_heatmap(results):
    """Create a heatmap of accuracies for different parameter combinations."""
    # Extract parameters and accuracies
    data = []
    for config, result in results.items():
        n_components = result['model_params']['n_components']
        n_mfcc = result['model_params']['n_mfcc']
        accuracy = result['overall_accuracy']
        data.append([n_components, n_mfcc, accuracy])
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['n_components', 'n_mfcc', 'accuracy'])
    
    # Create pivot table for heatmap
    pivot_table = df.pivot(index='n_components', columns='n_mfcc', values='accuracy')
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Accuracy by Parameter Combination')
    plt.xlabel('Number of MFCC coefficients')
    plt.ylabel('Number of HMM Components')
    plt.savefig('parameter_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_confusion_matrix(results, best_config):
    """Create confusion matrix visualization for best configuration."""
    confusion_matrix = results[best_config]['confusion_matrix']
    
    # Convert to numpy array
    genres = sorted(confusion_matrix.keys())
    matrix = np.zeros((len(genres), len(genres)))
    for i, true_genre in enumerate(genres):
        for j, pred_genre in enumerate(genres):
            matrix[i, j] = confusion_matrix[true_genre][pred_genre]
    
    # Convert to percentages
    matrix = matrix / matrix.sum(axis=1, keepdims=True) * 100
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=genres, yticklabels=genres)
    plt.title(f'Confusion Matrix (%) - {best_config}')
    plt.xlabel('Predicted Genre')
    plt.ylabel('True Genre')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_genre_performance_chart(results, best_config):
    """Create bar chart of per-genre accuracies."""
    accuracies = results[best_config]['per_genre_accuracy']
    genres = sorted(accuracies.keys())
    values = [accuracies[g] for g in genres]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(genres, values)
    plt.axhline(y=np.mean(values), color='r', linestyle='--', label='Mean Accuracy')
    plt.title(f'Genre-wise Performance - {best_config}')
    plt.xlabel('Genre')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('genre_performance.png', bbox_inches='tight', dpi=300)
    plt.close()

def print_summary_statistics(results):
    """Print summary statistics for the best configuration."""
    # Find best configuration
    best_config = max(results.items(), key=lambda x: x[1]['overall_accuracy'])[0]
    best_results = results[best_config]
    
    print("Summary Statistics")
    print("=" * 50)
    print(f"Best Configuration: {best_config}")
    print(f"Overall Accuracy: {best_results['overall_accuracy']:.1f}%")
    print("\nPer-Genre Performance:")
    print("-" * 50)
    
    # Sort genres by performance
    genre_accuracies = best_results['per_genre_accuracy']
    sorted_genres = sorted(genre_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    print("\nBest Performing Genres:")
    for genre, acc in sorted_genres[:3]:
        print(f"{genre:10}: {acc:.1f}%")
    
    print("\nWorst Performing Genres:")
    for genre, acc in sorted_genres[-3:]:
        print(f"{genre:10}: {acc:.1f}%")
    
    return best_config

def main():
    # Load results
    results_path = "./trained_models/evaluation_results/parameter_evaluation_results.json"
    results = load_results(results_path)
    
    # Generate visualizations
    best_config = print_summary_statistics(results)
    create_parameter_heatmap(results)
    create_confusion_matrix(results, best_config)
    create_genre_performance_chart(results, best_config)

if __name__ == "__main__":
    main()