import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import joblib
import argparse
from joblib import Parallel, delayed
import itertools
import json

class MusicGenreClassifier:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.models = {}
        self.best_params = {}
        
    def extract_features(self, audio_path, n_mfcc):
        try:
            y, sr = librosa.load(audio_path, duration=30)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            return mfcc.T
        except:
            return None

    def parallel_feature_extraction(self, song_paths, n_mfcc):
        features = Parallel(n_jobs=self.n_jobs)(
            delayed(self.extract_features)(song_path, n_mfcc) 
            for song_path in song_paths
        )
        return [f for f in features if f is not None]

    def save_model(self, model, genre, n_components, n_mfcc, accuracy, output_dir, is_best=False):
        # Create genre subfolder
        genre_dir = os.path.join(output_dir, 'all_models', genre)
        if not os.path.exists(genre_dir):
            os.makedirs(genre_dir)
            
        # Create filename with parameters
        filename = f"{genre}_c{n_components}_m{n_mfcc}_acc{accuracy:.2f}.pkl"
        model_path = os.path.join(genre_dir, filename)
        
        # Save model
        joblib.dump(model, model_path)
        
        # If it's the best model, also save it in the main directory(TO DO: remove this becuase it will never save the correct best model)
        if is_best:
            best_path = os.path.join(output_dir, f"{genre}_model.pkl")
            joblib.dump(model, best_path)

    def optimize_genre_parameters(self, genre, train_paths, valid_paths, output_dir):
        best_accuracy = 0
        best_model = None
        best_params = None
        
        param_combinations = list(itertools.product(range(3, 11), range(20, 26)))
        
        for n_components, n_mfcc in param_combinations:
            try:
                train_features = self.parallel_feature_extraction(train_paths, n_mfcc)
                if not train_features:
                    continue
                    
                X_train = np.vstack(train_features)
                
                model = hmm.GaussianHMM(
                    n_components=n_components,
                    covariance_type='diag',
                    n_iter=100,
                    random_state=42
                )
                model.fit(X_train)
                
                # Validate
                correct = 0
                valid_features = self.parallel_feature_extraction(valid_paths, n_mfcc)
                
                for features in valid_features:
                    if features is not None:
                        score = model.score(features)
                        if score > float('-inf'):
                            correct += 1
                            
                accuracy = (correct / len(valid_features)) * 100 if valid_features else 0
                
                # Save every model
                self.save_model(
                    model, genre, n_components, n_mfcc,
                    accuracy, output_dir,
                    is_best=(accuracy > best_accuracy)
                )
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_params = {
                        'n_components': n_components,
                        'n_mfcc': n_mfcc,
                        'accuracy': accuracy
                    }
                    
            except:
                continue
        
        return genre, best_model, best_params

    def prepare_data(self, data_path, test_size=0.2, valid_size=0.2):
        train_data = {}
        valid_data = {}
        test_data = {}
        
        genres = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        
        for genre in genres:
            genre_path = os.path.join(data_path, genre)
            songs = [s for s in os.listdir(genre_path) if s.endswith('.wav')]
            
            train_valid_songs, test_songs = train_test_split(songs, test_size=test_size, random_state=42)
            train_songs, valid_songs = train_test_split(train_valid_songs, test_size=valid_size, random_state=42)
            
            train_data[genre] = [os.path.join(genre_path, s) for s in train_songs]
            valid_data[genre] = [os.path.join(genre_path, s) for s in valid_songs]
            test_data[genre] = [os.path.join(genre_path, s) for s in test_songs]
            
        return train_data, valid_data, test_data
    
    def train(self, data_path, output_dir):
        train_data, valid_data, test_data = self.prepare_data(data_path)
        
        # Ensure output directories exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        all_models_dir = os.path.join(output_dir, 'all_models')
        if not os.path.exists(all_models_dir):
            os.makedirs(all_models_dir)
        
        # Parallel optimization for all genres
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.optimize_genre_parameters)(
                genre, train_paths, valid_data[genre], output_dir
            )
            for genre, train_paths in train_data.items()
        )
        
        # Store successful models and their parameters
        for genre, model, params in results:
            if model is not None:
                self.models[genre] = model
                self.best_params[genre] = params
        
        # Save best parameters
        params_path = os.path.join(output_dir, "best_parameters.json")
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        # Print optimized parameters
        print("\nOptimal parameters for each genre:")
        print("-" * 50)
        for genre, params in self.best_params.items():
            print(f"{genre:10} | n_components: {params['n_components']:2d} | n_mfcc: {params['n_mfcc']:2d} | validation accuracy: {params['accuracy']:.2f}%")
        
        # Evaluation phase
        if test_data:
            self.evaluate(test_data)
    
    def evaluate(self, test_data):
        total = 0
        correct = 0
        confusion_matrix = {genre: {g: 0 for g in self.models.keys()} for genre in test_data.keys()}
        
        for true_genre, song_paths in test_data.items():
            for song_path in song_paths:
                features = self.extract_features(song_path, self.best_params[true_genre]['n_mfcc'])
                if features is not None:
                    predicted_genre, _ = self.predict_from_features(features)
                    confusion_matrix[true_genre][predicted_genre] += 1
                    total += 1
                    if predicted_genre == true_genre:
                        correct += 1
        
        print("\nTest Evaluation Results:")
        print("-" * 50)
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"Overall Accuracy: {accuracy:.2f}%")
        
        print("\nPer-genre Accuracy:")
        print("-" * 50)
        for genre in confusion_matrix:
            total_genre = sum(confusion_matrix[genre].values())
            correct_genre = confusion_matrix[genre][genre]
            genre_accuracy = (correct_genre / total_genre * 100) if total_genre > 0 else 0
            print(f"{genre:10}: {genre_accuracy:.2f}%")
        
        print("\nConfusion Matrix:")
        print("-" * 50)
        print(f"{'':10} | {''.join(g.ljust(10) for g in self.models.keys())}")
        print("-" * (10 + 1 + 10 * len(self.models.keys())))
        for true_genre in confusion_matrix:
            row = f"{true_genre:10} | "
            row += ''.join(f"{confusion_matrix[true_genre][pred]:10d}" for pred in self.models.keys())
            print(row)

    def predict_from_features(self, features):
        scores = {}
        for genre, model in self.models.items():
            try:
                scores[genre] = model.score(features)
            except:
                scores[genre] = float('-inf')
        
        predicted_genre = max(scores.items(), key=lambda x: x[1])[0]
        return predicted_genre, scores

    def predict(self, audio_path):
        predictions = []
        for genre in self.models.keys():
            features = self.extract_features(audio_path, self.best_params[genre]['n_mfcc'])
            if features is not None:
                pred, _ = self.predict_from_features(features)
                predictions.append(pred)
        
        if not predictions:
            return None, {}
        
        from collections import Counter
        predicted_genre = Counter(predictions).most_common(1)[0][0]
        features = self.extract_features(audio_path, self.best_params[predicted_genre]['n_mfcc'])
        return self.predict_from_features(features)
    
    def load_models(self, models_dir):
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory {models_dir} not found")
            
        params_path = os.path.join(models_dir, "best_parameters.json")
        with open(params_path, 'r') as f:
            self.best_params = json.load(f)
            
        for model_file in os.listdir(models_dir):
            if model_file.endswith('_model.pkl'):
                genre = model_file.replace('_model.pkl', '')
                model_path = os.path.join(models_dir, model_file)
                self.models[genre] = joblib.load(model_path)

    def load_all_models(self, models_dir):
        """Load all models from the all_models directory."""
        all_models = {}
        all_models_dir = os.path.join(models_dir, 'all_models')
        
        if not os.path.exists(all_models_dir):
            raise FileNotFoundError(f"All models directory {all_models_dir} not found")
        
        # Load each genre's models
        for genre in os.listdir(all_models_dir):
            genre_path = os.path.join(all_models_dir, genre)
            if os.path.isdir(genre_path):
                all_models[genre] = {}
                for model_file in os.listdir(genre_path):
                    if model_file.endswith('.pkl'):
                        model_path = os.path.join(genre_path, model_file)
                        try:
                            # Format: genreName_c[number]_m[number]_acc[float].pkl
                            parts = model_file.split('_')
                            if len(parts) != 4:  # 4 parts: genre, c[num], m[num], acc[float].pkl
                                print(f"Skipping {model_file} - incorrect format")
                                continue
                                
                            n_components = int(parts[1][1:])  # Remove 'c' prefix
                            n_mfcc = int(parts[2][1:])       # Remove 'm' prefix
                            
                            model = joblib.load(model_path)
                            all_models[genre][(n_components, n_mfcc)] = model
                            
                        except Exception as e:
                            print(f"Error loading model {model_file}: {str(e)}")
                            continue
        
        if not all_models:
            print("Warning: No models were loaded. Check if the model files follow the format: genreName_c[number]_m[number]_acc[float].pkl")
        
        return all_models

    def evaluate_parameter_set(self, n_components, n_mfcc, all_models, test_data, genres):
        """
        Evaluate a specific parameter combination (n_components, n_mfcc) across all genres.
        Uses one model per genre, all trained with the same parameters.
        """
        total = 0
        correct = 0
        confusion_matrix = {genre: {g: 0 for g in genres} for genre in genres}
        per_genre_accuracy = {}
        
        # Get the models for these specific parameters
        genre_models = {}
        for genre in genres:
            if (n_components, n_mfcc) in all_models[genre]:
                genre_models[genre] = all_models[genre][(n_components, n_mfcc)]
        
        if len(genre_models) != len(genres):
            return None
        
        # Evaluate each test sample
        for true_genre, song_paths in test_data.items():
            correct_genre = 0
            total_genre = 0
            
            for song_path in song_paths:
                features = self.extract_features(song_path, n_mfcc)
                if features is not None:
                    try:
                        # Get score from each genre's model
                        scores = {}
                        for genre, model in genre_models.items():
                            try:
                                scores[genre] = model.score(features)
                            except:
                                scores[genre] = float('-inf')
                        
                        # Predict genre with highest score
                        predicted_genre = max(scores.items(), key=lambda x: x[1])[0]
                        confusion_matrix[true_genre][predicted_genre] += 1
                        total += 1
                        total_genre += 1
                        
                        if predicted_genre == true_genre:
                            correct += 1
                            correct_genre += 1
                    except:
                        continue
            
            if total_genre > 0:
                per_genre_accuracy[true_genre] = (correct_genre / total_genre) * 100
        
        overall_accuracy = (correct / total) * 100 if total > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'per_genre_accuracy': per_genre_accuracy,
            'confusion_matrix': confusion_matrix,
            'total_samples': total,
            'model_params': {
                'n_components': n_components,
                'n_mfcc': n_mfcc
            }
        }

    def evaluate_all_models(self, data_path, models_dir):
        """Evaluate parameter combinations where n_components and n_mfcc are consistent across genres."""
        # Load all models
        print("Loading all models...")
        all_models = self.load_all_models(models_dir)
        
        # Prepare test data
        _, _, test_data = self.prepare_data(data_path)
        genres = list(all_models.keys())
        
        # Create results directory
        results_dir = os.path.join(models_dir, 'evaluation_results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Find all unique parameter combinations
        param_combinations = set()
        for genre in all_models:
            param_combinations.update(all_models[genre].keys())
        
        print(f"\nEvaluating {len(param_combinations)} parameter combinations in parallel...")
        
        # Prepare parameters for parallel evaluation
        eval_params = [(n_components, n_mfcc, all_models, test_data, genres) 
                    for n_components, n_mfcc in sorted(param_combinations)]
        
        # Run evaluations in parallel
        results_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self.evaluate_parameter_set)(*params)
            for params in eval_params
        )
        
        # Process results
        results = {}
        for params, eval_results in zip(eval_params, results_list):
            n_components, n_mfcc = params[0], params[1]
            if eval_results is not None:
                results[f"c{n_components}_m{n_mfcc}"] = eval_results
        
        # Save all results
        results_path = os.path.join(results_dir, "parameter_evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Print summary
        print("\nResults Summary:")
        print("-" * 50)
        best_accuracy = 0
        best_config = None
        
        for config, res in results.items():
            accuracy = res['overall_accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config
        
        if best_config:
            print(f"Best configuration: {best_config}")
            print(f"Best accuracy: {best_accuracy:.2f}%")
            
            # Print per-genre accuracy for best configuration
            print("\nPer-genre accuracy for best configuration:")
            for genre, accuracy in results[best_config]['per_genre_accuracy'].items():
                print(f"{genre:10}: {accuracy:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Music Genre Classification using HMM')
    parser.add_argument('mode', choices=['train', 'predict', 'evaluate_all'], 
                      help='Mode of operation: train, predict, or evaluate all models')
    parser.add_argument('--data_path', required=True,
                      help='Path to GTZAN dataset for training or song file for prediction')
    parser.add_argument('--model_dir', default='trained_models',
                      help='Directory for saving/loading models')
    parser.add_argument('--n_jobs', type=int, default=-1,
                      help='Number of parallel jobs (-1 for all cores)')
    
    args = parser.parse_args()
    
    classifier = MusicGenreClassifier(n_jobs=args.n_jobs)
    
    if args.mode == 'train':
        classifier.train(args.data_path, args.model_dir)
        print(f"\nModels saved to {args.model_dir}")
        print(f"All trained models saved to {os.path.join(args.model_dir, 'all_models')}")
        
    elif args.mode == 'predict':
        classifier.load_models(args.model_dir)
        predicted_genre, scores = classifier.predict(args.data_path)
        
        if predicted_genre is not None:
            print(f"\nPredicted genre: {predicted_genre}")
            print("\nScores for each genre:")
            for genre, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                print(f"{genre:10}: {score:.2f}")
                
    elif args.mode == 'evaluate_all':
        classifier.evaluate_all_models(args.data_path, args.model_dir)
        print(f"\nEvaluation results saved to {os.path.join(args.model_dir, 'evaluation_results')}")

if __name__ == "__main__":
    main()