import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import math
import copy
import os
import gc
import time

# Define the Config class, including all configuration parameters
class Config:
    # Data Directory
    data_dir = '.'
    data_path = 'combined_precipitation_temperature_data.npy'
    
    # Voformer configuration
    train_voformer = False # Whether to train the Voformer model
    load_voformer = True # Whether to load an existing Voformer model
    voformer_model_path = 'best_voformer.pth' # Voformer model save path
    
    # Classifier
    classifier_path = 'best_voformer_classifier.pth' # Classifier save path
    
    # Clustering configuration
    perform_clustering = True # Whether to perform clustering
    clustering_results_path = 'pseudo_labels.npy' # Clustering result save path
    extracted_features_path = 'extracted_features.npy' # Extracted feature save path
    
    # MMformer configuration
    train_mmformer = False # Whether to train the MMformer model
    load_mmformer = True # Whether to load an existing MMformer model
    drought_data = 'drought_data.npy'
    save_path = "best_Meta_MMformer_model.pth"
    
    # MVformer Visualization
    visualize_clusters = True       # Whether to visualize the clustering results
    visualize_predictions = True    # Whether to visualize MMformer prediction results
    visualization_output_dir = 'visualizations'  # Visualization output directory
    
    # Voformer parameters
    input_dim = 2
    d_model = 512
    n_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    batch_size_voformer = 256
    num_epochs = 30
    learning_rate = 1e-5
    neighborhood_radius = 22 # Clustering neighborhood radius
    ex_neighborhood_radius = 1.1
    DC_reference_distance = 0.08
    Noise_filtering_threshold = 0.05
    
    # MMformer parameters
    mm_d_model = 512
    mm_n_heads = 8
    mm_d_ff = 2048
    mm_e_layers = 6
    mm_d_layers = 6
    mm_seq_len = 30
    mm_pred_len = 10
    mm_dropout = 0.2
    mm_factor = 260
    mm_feature_size = 2
    batch_size_mmformer = 360
    mm_output_attention = True  
    scheduled_sampling = True  # Whether to use planned sampling
    sampling_ratio = 0.25       # Planned sampling rate
    outer_lr_max = 0.00005      # Maximum outer loop learning rate
    outer_lr_min = 1e-9       # Minimum outer loop learning rate
    inner_lr_init = 1e-7      # Initial inner loop learning rate
    inner_lr_min = 1e-9       # Minimum inner loop learning rate
    inner_steps = 5
    lr_inner = inner_lr_init
    warmup_epochs = 2         # Number of warmup epochs
    lr_scheduler_patience = 5  # Patience value for the learning rate scheduler
    weight_decay = 0.05        # Optimizer Weight Decay
    patience = 20              # Patience
    mm_num_epochs = 100
    
    epsilon = 1
    min_actual = 1e-1
    #Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
configs = Config()

##Extreme Clustering
def extreme_clustering(features, neighborhood_radius=0.2):
    # Handle both PyTorch tensors and NumPy arrays
    if isinstance(features, np.ndarray):
        data = features
    else:  # Assume PyTorch tensor
        data = features.detach().cpu().numpy()
    
    number = data.shape[0]
    dim = data.shape[1]

    # Calculate the distance matrix
    dist1 = pdist(data, metric='euclidean')
    dist = squareform(dist1)

    # Sorting distance and index
    sorted_indices = np.argsort(dist, axis=1)
    sorted_distances = np.sort(dist, axis=1)

    # Calculate distance
    position = int(round(number * configs.DC_reference_distance)) - 1
    sda = np.sort(dist, axis=0)
    dc = sda[position % number, position // number]

    # Calculate density
    density = np.zeros(number)
    for i in range(number - 1):
        for j in range(i + 1, number):
            tmp = np.exp(-((dist[i, j] / dc) ** 2))
            density[i] += tmp
            density[j] += tmp

    # Finding extreme points
    extreme_points = []
    state = np.zeros(number)
    for i in range(number):
        if state[i] == 0:
            j = 1
            while j < number and density[i] >= density[sorted_indices[i, j]] and sorted_distances[i, j] < neighborhood_radius:
                if density[i] == density[sorted_indices[i, j]]:
                    state[sorted_indices[i, j]] = 1
                j += 1
            if j < number and sorted_distances[i, j] >= neighborhood_radius:
                extreme_points.append(i)

    # Allocation Category
    clustering_result = np.zeros(number, dtype=int) - 1
    for idx, point in enumerate(extreme_points):
        clustering_result[point] = idx + 1
        j = 1
        while j < number and sorted_distances[point, j] < neighborhood_radius:
            if density[point] == density[sorted_indices[point, j]]:
                clustering_result[sorted_indices[point, j]] = idx + 1
            j += 1

    # Allocate the remaining points
    for i in range(number):
        if clustering_result[i] == -1:
            queue = [i]
            while True:
                current_point = queue[-1]
                j = 0
                while j < number and density[current_point] >= density[sorted_indices[current_point, j]]:
                    j += 1
                if j >= number:
                    break  # Prevent j from going out of range
                if clustering_result[sorted_indices[current_point, j]] == -1:
                    queue.append(sorted_indices[current_point, j])
                else:
                    break
                if len(queue) >= number:
                    break
            if j < number:
                label = clustering_result[sorted_indices[current_point, j]]
                for point in queue:
                    clustering_result[point] = label

    # Remove noise points
    unique_labels, counts = np.unique(clustering_result, return_counts=True)
    mean_count = np.mean(counts[unique_labels != -1])
    noise_labels = unique_labels[counts < mean_count * configs.Noise_filtering_threshold]
    for label in noise_labels:
        clustering_result[clustering_result == label] = -1

    # Renumber
    unique_labels = np.unique(clustering_result)
    label_map = {label: idx for idx, label in enumerate(unique_labels) if label != -1}
    for old_label, new_label in label_map.items():
        clustering_result[clustering_result == old_label] = new_label

    return clustering_result

##Voformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(torch.float32)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class Volatilite(nn.Module):
    def forward(self, x):
        mean_x = torch.mean(x, dim=1, keepdim=True)
        deviation = (mean_x - x) ** 2
        mean_deviation = torch.mean(deviation, dim=1, keepdim=True)  # Shape: (batch_size, 1, ...)
        volatility = torch.sqrt(mean_deviation)  # Shape: (batch_size, 1, ...)
        scaled_x = x * volatility
        return scaled_x

class ProbSparseAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, mask):
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape

        # Calculate scores (QK^T / sqrt(d_k)) and apply top-k sparsity
        scale = self.scale or 1.0 / (D ** 0.5)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))

        # Top-k selection (sparse attention focus)
        U_part = L_K
        top_k = max(1, int(U_part / 4))  # Use 25% sparsity
        idx = torch.topk(scores, top_k, dim=-1)[1]  # Get top-k indices
        mask_topk = torch.zeros_like(scores).scatter_(-1, idx, 1.0).bool()
        sparse_scores = scores.masked_fill(~mask_topk, -float('inf'))

        # Apply softmax, dropout, and compute attention outputs
        attn = self.dropout(torch.softmax(sparse_scores, dim=-1))
        outputs = torch.matmul(attn, values)

        return outputs

class InformerAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(InformerAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / (self.d_k ** 0.5)

        self.qkv_projection = nn.Linear(d_model, d_model * 3)  # Query, key, value
        self.out_projection = nn.Linear(d_model, d_model)

        self.attention = ProbSparseAttention(scale=self.scale, attention_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape  # Batch size, Sequence length, Feature dimension (d_model)

        # Compute Q, K, V
        qkv = self.qkv_projection(x).view(B, L, 3, self.n_heads, self.d_k)
        queries, keys, values = qkv.unbind(dim=2)  # Split Q, K, V (B, L, H, Dk)

        # Reshape for multi-head attention
        queries = queries.permute(0, 3, 1, 2).contiguous().view(-1, L, self.d_k)
        keys = keys.permute(0, 3, 1, 2).contiguous().view(-1, L, self.d_k)
        values = values.permute(0, 3, 1, 2).contiguous().view(-1, L, self.d_k)

        # Apply ProbSparseAttention
        outputs = self.attention(queries, keys, values, mask)

        # Reshape back to original dimensions
        outputs = outputs.view(B, self.n_heads, L, self.d_k).permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(B, L, -1)  # (B, L, d_model)

        # Final projections with dropout and residual connection
        outputs = self.dropout(self.out_projection(outputs))
        outputs = self.norm(outputs + x)  # Residual connection ensures same shape (B, L, d_model)

        return outputs

class VoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(VoformerEncoderLayer, self).__init__()
        self.self_attn = InformerAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            Volatilite(),  # Volatilite Activation Function
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, mask=src_mask)
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.ffn(src2)
        src = src + src2
        return src

class VoformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(VoformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.norm1.normalized_shape)

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask=src_mask)
        return self.norm(src)

class Voformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, d_ff, dropout=0.1):
        super(Voformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = VoformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.encoder = VoformerEncoder(encoder_layer, num_layers)

    def forward(self, src, src_mask=None):
        # Debugging shape
        assert src.dim() == 3, "Input source (src) must have 3 dimensions (batch_size, seq_len, input_dim)."
        assert src.size(-1) == self.input_linear.in_features, "Input feature mismatch for input_linear."
        
        src = self.input_linear(src)  # (batch_size, seq_len, d_model)
        assert src.size(-1) == self.input_linear.out_features, "Output dimension mismatch for input_linear."
        
        src = self.pos_encoder(src)  # (batch_size, seq_len, d_model)
        output = self.encoder(src, src_mask=src_mask)  # (batch_size, seq_len, d_model)
        
        return output

##Voformer Training
# Training Voformer
def train_voformer(configs):
    # Loading data
    data = np.load(configs.data_path)

    # Data preprocessing
    num_samples, seq_length, num_features = data.shape
    data_reshaped = data.reshape(num_samples, seq_length * num_features)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    
    pseudo_labels = extreme_clustering(data_scaled, neighborhood_radius=configs.neighborhood_radius)
    # check shape
    print(f'Data shape: {data.shape}')  # Out: (2415, 276, 2)
    # Check the shape of pseudo labels
    print(f'Pseudo labels shape: {pseudo_labels.shape}')  # should be (2415,)
    
    # Number of clusters
    num_clusters = len(np.unique(pseudo_labels))
    print(f'Number of Pseudo labels:{num_clusters}')
    
    # Reshape data_scaled back to (2415, 276, 2)
    data_scaled_3d = data_scaled.reshape(num_samples, seq_length, num_features)
    X = torch.tensor(data_scaled_3d, dtype=torch.float32) # (2415, 276, 2)
    y = torch.tensor(pseudo_labels, dtype=torch.long) # (2415,)

    # Creating a dataset and data loader
    dataset = TensorDataset(X, y)
    # Divide the training set, validation set and test set (for example, 70% training, 15% validation, 15% test)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size_voformer, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size_voformer, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size_voformer, shuffle=False)

    # Initialize the model
    model = Voformer(configs.input_dim, configs.d_model, configs.n_heads, configs.num_layers, configs.d_ff, configs.dropout)
    model = model.to(configs.device)
    classifier = nn.Linear(configs.d_model, len(torch.unique(y))).to(configs.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=configs.learning_rate)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(configs.num_epochs):
        model.train()
        classifier.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(configs.device)
            batch_y = batch_y.to(configs.device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            outputs = outputs.mean(dim=1)
            logits = classifier(outputs)

            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)

        # Verification
        model.eval()
        classifier.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(configs.device)
                batch_y = batch_y.to(configs.device)

                outputs = model(batch_X)
                outputs = outputs.mean(dim=1)
                logits = classifier(outputs)

                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        print(f'Epoch {epoch+1}/{configs.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), configs.voformer_model_path)
            print("Save the current best Voformer model")
        else:
            patience_counter += 1
            if patience_counter >= configs.patience:
                print("Early stop trigger, stop training...")
                break

    print("Voformer model training completed and saved")
    return model

##Voformer-EC
# Loading Voformer
def load_voformer_model(configs):
    model = Voformer(configs.input_dim, configs.d_model, configs.n_heads, configs.num_layers, configs.d_ff, configs.dropout)
    model.load_state_dict(torch.load(configs.voformer_model_path, map_location=configs.device))
    model = model.to(configs.device)
    model.eval()
    print("Voformer loaded")
    return model

# Extract features by Voformer
def extract_features_with_voformer(data, model, configs):
    num_samples, seq_length, num_features = data.shape
    print(f"Processing {num_samples} samples")
    data_reshaped = data.reshape(num_samples, seq_length * num_features)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_reshaped)
    data_scaled_3d = data_scaled.reshape(num_samples, seq_length, num_features)
    X = torch.tensor(data_scaled_3d, dtype=torch.float32).to(configs.device)
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=configs.batch_size_voformer, shuffle=False)

    features = []
    with torch.no_grad():
        for batch in loader:
            batch_X = batch[0].to(configs.device)
            output = model(batch_X)
            output = output.mean(dim=1)
            features.append(output.cpu().numpy())
    features = np.vstack(features)
    np.save(configs.extracted_features_path, features)
    print(f'Extracted feature shapes: {features.shape}')
    return features

# Extreme Clustering
def perform_extreme_clustering(data, voformer_model, configs):
    features = extract_features_with_voformer(data, voformer_model, configs)
    
    # Make sure the first dimension of features matches the first dimension of the original data
    if features.shape[0] != data.shape[0]:
        print(f"Warning: Features shape {features.shape} doesn't match data shape {data.shape}")
        
    pseudo_labels = extreme_clustering(features, neighborhood_radius=configs.ex_neighborhood_radius)
    # Ensure that the number of pseudo labels is consistent with the number of data samples
    assert len(pseudo_labels) == data.shape[0], f"Pseudo labels count ({len(pseudo_labels)}) doesn't match data samples ({data.shape[0]})"
    
    np.save(configs.clustering_results_path, pseudo_labels)
    print("Clustering is complete and the clustering results have been saved")
    print(f"Pseudo labels shape: {pseudo_labels.shape}")
    return pseudo_labels

# Extreme Clustering loaded
def load_clustering_results(configs):
    pseudo_labels = np.load(configs.clustering_results_path)
    print("Clustering results loaded")
    # Number of clusters
    num_unique_values = len(np.unique(pseudo_labels))
    print(f'Number of clusters:{num_unique_values}')
    return pseudo_labels

# Extract drought data function
def extract_drought_data(data, pseudo_labels):
    # Make sure the length of pseudo_labels matches the first dimension of data
    if len(pseudo_labels) != data.shape[0]:
        raise ValueError(f"Length mismatch: pseudo_labels ({len(pseudo_labels)}) vs data ({data.shape[0]})")
        
    # Calculate average temperature and precipitation
    average_temperature = data[:, :, 0].mean(axis=1)
    average_precipitation = data[:, :, 1].mean(axis=1)
    
    num_unique_values = len(np.unique(pseudo_labels))
    print(f'Number of clusters:{num_unique_values}')

    cluster_df = {
        'cluster': pseudo_labels,
        'avg_temp': average_temperature,
        'avg_precip': average_precipitation
    }
    df = pd.DataFrame(cluster_df)
    cluster_stats = df.groupby('cluster').agg({'avg_temp': 'mean', 'avg_precip': 'mean'})

    overall_avg_temp = average_temperature.mean()
    overall_avg_precip = average_precipitation.mean()
    print(f'ave_temp of data:{overall_avg_temp}')
    print(f'ave_precip of data:{overall_avg_precip}')

    temp_threshold = overall_avg_temp + 5  # Temperature above average 5℃
    precip_threshold = overall_avg_precip * 0.8  # Precipitation is 80% below average
    print(f'Temperature threshold (above 5-10℃ of the mean): {temp_threshold}')
    print(f'Precipitation threshold (below 80% of the mean): {precip_threshold}')
    temp_threshold = float(temp_threshold)
    precip_threshold = float(precip_threshold)

    drought_clusters = cluster_stats[
        (cluster_stats['avg_temp'] > temp_threshold) &
        (cluster_stats['avg_precip'] < precip_threshold)
    ].index.tolist()
    print(f'Cluster number of drought area: {drought_clusters}')
    print(f'Amount of drought area: {len(np.unique(drought_clusters))}')

    drought_labels = np.isin(pseudo_labels, drought_clusters).astype(int)
    drought_station_indices = np.where(drought_labels == 1)[0]
    print("Base station number belonging to the drought cluster：", drought_station_indices)
    print("There are {} base stations belonging to the drought cluster".format(len(drought_station_indices)))

    drought_data = data[drought_station_indices]
    np.save('drought_data.npy', drought_data)
    print("Drought data extracted and saved")
    return drought_data, drought_clusters

##MMformer
# Causal mask
def generate_causal_mask(seq_len, batch_size, n_heads, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    mask = mask.unsqueeze(0).unsqueeze(1)
    mask = mask.expand(batch_size, n_heads, seq_len, seq_len)  # 形状: (batch_size, n_heads, seq_len, seq_len)
    return mask

# Monte Carlo Dropout
class MCDropout(nn.Dropout):
    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)
    
# Probabilistic Attention
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=7, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = MCDropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, H, L_q, D = queries.shape  # Batch size, Heads, Length, Depth per head
        L_k = keys.shape[2]

        # Scale dot-product attention
        scale = self.scale or 1. / (D ** 0.5)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, values)
        return context, attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).to(torch.float32)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(1)
        x = x + self.pe[:, :L, :].to(x.device)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model, bias=True)
        self.position_embedding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.value_embedding(x)
        x = self.position_embedding(x)
        return self.dropout(x)

# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.attention = attention
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        B, L_q, _ = query.size()
        B, L_k, _ = key.size()
        query = self.query_projection(query).view(B, L_q, self.n_heads, self.d_k)
        key   = self.key_projection(key).view(B, L_k, self.n_heads, self.d_k)
        value = self.value_projection(value).view(B, L_k, self.n_heads, self.d_v)
    
        query = query.permute(0, 2, 1, 3)   # (B, n_heads, L_q, d_k)
        key   = key.permute(0, 2, 1, 3)     # (B, n_heads, L_k, d_k)
        value = value.permute(0, 2, 1, 3)   # (B, n_heads, L_k, d_v)
    
        context, attn = self.attention(query, key, value, attn_mask)
        context = context.permute(0, 2, 1, 3).contiguous()  # (B, L_q, n_heads, d_v)
        context = context.view(B, L_q, -1)                  # (B, L_q, d_model)
        out = self.out_projection(context)                  # (B, L_q, d_model)
        return out, attn

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, d_model, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = MCDropout(dropout)

    def forward(self, src, src_mask=None):
        src2, _ = self.attention_layer(src, src, src, attn_mask=src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

# Encoder
class Encoder(nn.Module):
    def __init__(self, layer, N_max, d_model):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N_max)])
        # self.norm = nn.LayerNorm(layer.size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return self.norm(src)

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, self_attention_layer, cross_attention_layer, d_model, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention_layer    # Self-attention layer
        self.cross_attention = cross_attention_layer  # Cross-attention layer for context from the encoder

        # Feed Forward Network (FFN)
        d_ff = d_ff or 4 * d_model
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.ELU(),
                                nn.Linear(d_ff, d_model))

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = MCDropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        out1, _ = self.self_attention(tgt, tgt, tgt, attn_mask=tgt_mask)
        x = tgt + self.dropout(out1)
        x = self.norm1(x)
        
        out2, _ = self.cross_attention(x, memory, memory, attn_mask=memory_mask)
        x = x + self.dropout(out2)
        x = self.norm2(x)
        
        out3 = self.ff(x)
        x = x + self.dropout(out3)
        x = self.norm3(x)
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, layers, d_model):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.norm(tgt)

# MMformer Model
class MMformer(nn.Module):
    def __init__(self, configs):
        super(MMformer, self).__init__()
        self.configs = configs
        # Embedding
        self.enc_embedding = DataEmbedding(configs.mm_feature_size, configs.mm_d_model, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.mm_feature_size, configs.mm_d_model, configs.dropout)
        
        # Encoder
        encoder_layer = EncoderLayer(
            AttentionLayer(
                ProbAttention(False, configs.mm_factor, attention_dropout=configs.dropout),
                configs.mm_d_model,
                configs.mm_n_heads
            ),
            configs.mm_d_model,
            configs.mm_d_ff,
            dropout=configs.mm_dropout
        )
        self.encoder = Encoder(encoder_layer, configs.mm_e_layers, configs.mm_d_model)
        
        # Decoder
        decoder_layers = [
            DecoderLayer(
                self_attention_layer=AttentionLayer(
                    ProbAttention(False, configs.mm_factor, attention_dropout=configs.mm_dropout),
                    configs.mm_d_model,
                    configs.mm_n_heads
                ),
                cross_attention_layer=AttentionLayer(
                    ProbAttention(False, configs.mm_factor, attention_dropout=configs.mm_dropout),
                    configs.mm_d_model,
                    configs.mm_n_heads
                ),
                d_model=configs.mm_d_model,
                d_ff=configs.mm_d_ff,
                dropout=configs.mm_dropout
            ) for _ in range(configs.mm_d_layers)
        ]
        self.decoder = Decoder(decoder_layers, configs.mm_d_model)
    
        self.projector = nn.Linear(configs.mm_d_model, configs.mm_feature_size, bias=True)
        self.pred_len = configs.mm_pred_len
        
    def encode(self, x):
        enc_out = self.enc_embedding(x)     # (B, L, d_model)
        enc_out = self.encoder(enc_out)     # (B, L, d_model)
        return enc_out

    def decode_step(self, tgt_embedding, memory, tgt_mask=None, memory_mask=None):
        dec_out = self.decoder(tgt_embedding, memory, tgt_mask, memory_mask)  # (B, 1, d_model)
        out =  self.projector(dec_out)  # (B, 1, feature_size)
        return out
    
    def forward_autoregressive(self, src, tgt=None, pred_len=None, sampling_prob=1.0):
        if pred_len is None:
            pred_len = self.pred_len
        B, L, C = src.shape
        device = src.device
        
        memory = self.encode(src)    # => (B, L, d_model)
        n_heads = self.encoder.layers[0].attention_layer.n_heads
    
        first_step_input = torch.zeros((B, 1, C), device=device)
        dec_emb_list = [self.dec_embedding(first_step_input)]  # 列表，后面会不断 append
        
        outputs = []
        
        for step in range(pred_len):
            tgt_emb = torch.cat(dec_emb_list, dim=1)  # => (B, step+1, d_model)
            
            seq_len_dec = tgt_emb.size(1)
            mask = torch.tril(torch.ones(seq_len_dec, seq_len_dec, device=device)).bool()
            mask = mask.unsqueeze(0).unsqueeze(1)  # => (1, 1, seq_len_dec, seq_len_dec)
            mask = mask.expand(B, n_heads, seq_len_dec, seq_len_dec)  # => (B, n_heads, seq_len_dec, seq_len_dec)
            
            dec_out = self.decoder(tgt_emb, memory, tgt_mask=mask, memory_mask=None)  # => (B, seq_len_dec, d_model)
            
            out_step = self.projector(dec_out[:, -1:, :])  # => (B, 1, feature_size)
            outputs.append(out_step)
        
            # (3.5) Teacher Forcing / Scheduled Sampling
            if self.training and (tgt is not None):
                use_real = torch.rand(B, device=device) < sampling_prob
                use_real = use_real.float().unsqueeze(1).unsqueeze(2)  # => (B, 1, 1)
                next_inp = use_real * tgt[:, step, :].unsqueeze(1) + \
                           (1 - use_real) * out_step
            else:
                next_inp = out_step
            
            dec_emb_list.append(self.dec_embedding(next_inp))
        
        outputs = torch.cat(outputs, dim=1)  # => (B, pred_len, C)
        return outputs
    
def forward_autoregressive_with_params(model, src, adapted_params, tgt=None, pred_len=None, sampling_prob=0.0):
    original_params_data = {}
    original_params_data = {}
    for name, param in model.named_parameters():
        original_params_data[name] = param.data.clone()
        
    try:
        for name, param in model.named_parameters():
            if name in adapted_params:
                param.data.copy_(adapted_params[name])
    
        outputs = model.forward_autoregressive(
            src, tgt=tgt, pred_len=pred_len, sampling_prob=sampling_prob)
    
    finally:
        for name, param in model.named_parameters():
            param.data.copy_(original_params_data[name])
    return outputs

def backup_model_params(model):
    backup = {}
    for name, param in model.named_parameters():
        backup[name] = param.data.clone()
    expected_params = set([name for name, _ in model.named_parameters()])
    backup_params = set(backup.keys())
    assert expected_params == backup_params, "Some parameters are missing in the backup."
    return backup


def restore_model_params(model, backup):
    for name, param in model.named_parameters():
        if name in backup:
            param.data.copy_(backup[name])
        else:
            raise KeyError(f"Parameter '{name}' not found in backup.")
    
# MAML (Meta-learning)
class MAML:
    def __init__(self, model, configs, resume_training=False, last_epoch=-1):
        self.model = model
        self.lr_inner = configs.inner_lr_init
        self.optimizer = optim.AdamW(model.parameters(), lr=configs.outer_lr_max, weight_decay=configs.weight_decay) 
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=configs.outer_lr_max,
            epochs=configs.mm_num_epochs,
            steps_per_epoch=1,
            pct_start=configs.warmup_epochs/configs.mm_num_epochs,
            anneal_strategy='cos',
            final_div_factor=configs.outer_lr_max/configs.outer_lr_min,
            last_epoch=last_epoch if resume_training else -1
        )
        
        dummy_optimizer = optim.SGD([torch.tensor(self.lr_inner, requires_grad=False)], lr=self.lr_inner)
        self.inner_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            dummy_optimizer,
            T_0=configs.warmup_epochs,
            T_mult=2,
            eta_min=configs.inner_lr_min,
            last_epoch=last_epoch if resume_training else -1
        )

    def adapt(self, loss):
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        adapted_params = {
            name: param - self.lr_inner * grad
            for (name, param), grad in zip(self.model.named_parameters(), grads)
        }
        return adapted_params

    def meta_update(self, meta_loss):
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        
        self.scheduler.step()
        self.inner_scheduler.step()
        self.lr_inner = self.inner_scheduler.optimizer.param_groups[0]['lr']
        
def _evaluate_on_global_val(model, val_loader, loss_function, eval_metrics, configs, device):
    model.eval()
    total_loss = 0.0
    total_mape = 0.0
    count = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            batch_inputs  = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            preds = model.forward_autoregressive(
                src=batch_inputs, tgt=None,
                pred_len=configs.mm_pred_len,
                sampling_prob=0.0
            )
            loss = loss_function(preds, batch_targets)
            total_loss += loss.item()
    
            mape = eval_metrics(batch_targets, preds)
            total_mape += mape.item()
            count += 1
    
    avg_loss = total_loss / (count if count>0 else 1)
    avg_mape = total_mape / (count if count>0 else 1)
    return avg_loss, avg_mape

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    loss_function = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model.forward_autoregressive(data)
            loss = loss_function(output, target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f"Evaluation - Average Loss: {avg_loss:.4f}")
    return avg_loss

def load_mmformer_model(configs):
    device = configs.device
    model = MMformer(configs).to(device)
    model.load_state_dict(torch.load(configs.save_path, map_location=device))
    model.eval()
    print(f"MMformer model loaded from {configs.save_path}")
    return model

# Define Evaluation Metrics
def eval_metrics(actual, predicted, epsilon=configs.epsilon, min_actual=configs.min_actual):
    mask = torch.abs(actual) > min_actual
    filtered_actual = actual[mask]
    filtered_predicted = predicted[mask]
    mape = torch.mean(torch.abs((filtered_actual - filtered_predicted) / (torch.abs(filtered_actual) + epsilon))) * 100
    return mape

##MMformer Training
def train_mmformer(data, configs):
    # Data preprocessing
    #data = np.load(data)
    
    # Customize how much data to use for predictions
    forecast_length = data.shape[1]
    data_forecast = data[:, :forecast_length, :]
    
    # StandardScaler for 3D data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_forecast.reshape(-1, data_forecast.shape[-1])).reshape(data_forecast.shape)

    # Divide the dataset
    train_days = 160
    val_days = 160 + 60
    train_data = scaled_data[:, :train_days, :]
    val_data = scaled_data[:, train_days:val_days, :]
    test_data = scaled_data[:, val_days:, :]
    
    # Create datasets
    def create_sequences_3d(data_3d, seq_len, pred_len):
        num_cities, num_days, num_features = data_3d.shape
        num_windows = num_days - seq_len - pred_len + 1
    
        if num_windows <= 0:
            raise ValueError("Not enough days for given seq_len and pred_len")
    
        sequences_x = []
        sequences_y = []
    
        for city in range(num_cities):
            city_data = data_3d[city]
            for i in range(num_windows):
                start_index = i
                end_index = i + seq_len + pred_len
                window = city_data[start_index:end_index, :]
                sequences_x.append(window[:seq_len, :])
                sequences_y.append(window[seq_len:, :])
    
        sequences_x = np.array(sequences_x)
        sequences_y = np.array(sequences_y)
    
        sequences_tensor = torch.from_numpy(sequences_x).float()
        targets_tensor = torch.from_numpy(sequences_y).float()
    
        print(f"num_cities: {num_cities}, num_days: {num_days}, num_features: {num_features}")
        print(f"seq_len: {seq_len}, pred_len: {pred_len}")
        print(f"Calculated num_windows: {num_windows}")
        print(f"sequences_x shape: {sequences_x.shape}")
        print(f"sequences_y shape: {sequences_y.shape}")
        print(f"sequences_tensor shape: {sequences_tensor.shape}")
        print(f"targets_tensor shape: {targets_tensor.shape}")
    
        return sequences_tensor, targets_tensor
    
    print(scaled_data.shape)
    
    # Convert data to PyTorch tensors
    train_sequences, train_targets = create_sequences_3d(train_data, configs.mm_seq_len, configs.mm_pred_len)
    val_sequences,   val_targets   = create_sequences_3d(val_data,   configs.mm_seq_len, configs.mm_pred_len)
    test_sequences,  test_targets  = create_sequences_3d(test_data,  configs.mm_seq_len, configs.mm_pred_len)
    
    train_dataset = TensorDataset(train_sequences, train_targets)
    val_dataset   = TensorDataset(val_sequences,   val_targets)
    test_dataset  = TensorDataset(test_sequences,  test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size_mmformer, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=configs.batch_size_mmformer, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=configs.batch_size_mmformer, pin_memory=True)

    # Initialize the model and optimizer
    device = configs.device
    model = MMformer(configs).to(device)
    loss_function = torch.nn.MSELoss()

    best_val_loss = float('inf')
    
    if configs.load_mmformer and os.path.exists(configs.save_path):
        print(f"Loading existing model from {configs.save_path}")
        model.load_state_dict(torch.load(configs.save_path, map_location=device))
        print("Model loaded successfully.")
        
        if not configs.train_mmformer:
            model.eval()
            print("Model loaded in evaluation mode (no training).")
            return model, test_loader, scaler
        
    else:
        if not configs.train_mmformer:
            print("Error: Cannot activate MMformer. Model file not found and training is disabled.")
            raise FileNotFoundError(f"Model file {configs.save_path} not found and training is disabled.")
        print("Initializing a new model for training.")
            
    if configs.train_mmformer:
        resume_training = configs.load_mmformer and os.path.exists(configs.save_path)
        last_epoch = -1
            
        maml = MAML(model, configs, resume_training=resume_training, last_epoch=last_epoch)
        loss_function = torch.nn.MSELoss()
        best_val_loss = float('inf')

        num_tasks = 1
        task_size = len(train_dataset) // num_tasks
        task_datasets = [torch.utils.data.Subset(train_dataset, range(i*task_size, (i+1)*task_size)) for i in range(num_tasks)]
        pred_len = configs.mm_pred_len
        seq_len = configs.mm_seq_len
        max_encoder_layers = configs.mm_e_layers
        max_decoder_layers = configs.mm_d_layers
        performance_dict = {}
        
        num_epochs = configs.mm_num_epochs
        best_val_loss = float('inf')
        best_model_state = None
        inner_steps = configs.inner_steps
        save_path = configs.save_path
        global_val_loader = val_loader
        patience = configs.patience
        trigger_times = 0
        tasks = [{"train_loader": train_loader, "val_loader": val_loader}]
        global_val_loader = val_loader
    
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            epoch_meta_loss = 0.0
    
            for task in tasks:
                train_loader = task["train_loader"]
                train_iter = iter(train_loader)
    
                original_params = {name: param.clone().detach() for name, param in model.named_parameters()}
                adapted_params = {name: param.clone().detach().requires_grad_(True) for name, param in model.named_parameters()}
    
                for _ in range(inner_steps):
                    try:
                        batch_inputs, batch_targets = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch_inputs, batch_targets = next(train_iter)
    
                    batch_inputs = batch_inputs.to(device)
                    batch_targets = batch_targets.to(device)
    
                    preds = forward_autoregressive_with_params(
                        model, batch_inputs, adapted_params, tgt=batch_targets,
                        pred_len=configs.mm_pred_len, sampling_prob=configs.sampling_ratio
                    )
                    loss = loss_function(preds, batch_targets)
    
                    grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True, allow_unused=True)
                    adapted_params = {
                        name: param - configs.inner_lr_init * (grad if grad is not None else torch.zeros_like(param))
                        for (name, param), grad in zip(adapted_params.items(), grads)
                    }
    
                try:
                    val_inputs, val_targets = next(iter(val_loader))
                except StopIteration:
                    val_inputs, val_targets = next(iter(val_loader))
    
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
    
                val_preds = forward_autoregressive_with_params(
                    model, val_inputs, adapted_params, tgt=None,
                    pred_len=configs.mm_pred_len, sampling_prob=0.0
                )
                val_loss = loss_function(val_preds, val_targets)
    
                epoch_meta_loss += val_loss
    
                for name, param in model.named_parameters():
                    param.data.copy_(original_params[name])
    
            avg_meta_loss = epoch_meta_loss / len(tasks)
    
            maml.optimizer.zero_grad()
            avg_meta_loss.backward()
            maml.optimizer.step()
    
            maml.scheduler.step()
            maml.inner_scheduler.step()
    
            val_loss_global, val_mape_global = _evaluate_on_global_val(
                model, global_val_loader, loss_function, eval_metrics, configs, device
            )
    
            print(f"[Epoch {epoch+1}/{num_epochs}] meta_loss={avg_meta_loss.item():.6f}, "
                  f"global_val_loss={val_loss_global:.6f}, global_val_mape={val_mape_global:.2f}%")
            
            if val_loss_global < best_val_loss:
                best_val_loss = val_loss_global
                best_model_state = model.state_dict()
                torch.save(best_model_state, save_path)
                trigger_times = 0
                print(f"==> Best global_val_loss={best_val_loss:.6f}，Saved to {save_path}")
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print(f'Early stopping triggered at epoch {epoch+1}! Best val loss: {best_val_loss}')
                    model.load_state_dict(best_model_state)
                    break

    return model, test_loader, scaler

##MVformer
# MMformer Loaded
def load_mmformer_model(configs):
    device = configs.device
    model = MMformer(configs).to(device)
    model.load_state_dict(torch.load(configs.save_path, map_location=device))
    model.eval()
    print("MMformer Loaded")
    return model

# MMformer to predict
def predict_with_mmformer(mmformer_model, drought_data, configs): 
    device = configs.device
    mmformer_model = mmformer_model.to(device)
    mmformer_model.eval()
    
    data = np.load(os.path.join(configs.data_dir, configs.data_path))
    forecast_length = data.shape[1]
    data_forecast = data[:, :forecast_length, :]
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_forecast.reshape(-1, data_forecast.shape[-1])).reshape(data_forecast.shape)
    
    device = configs.device
    
    # Create datasets
    def create_sequences_3d(data_3d, seq_len, pred_len):
        num_cities, num_days, num_features = data_3d.shape
        num_windows = num_days - seq_len - pred_len + 1
    
        if num_windows <= 0:
            raise ValueError("Not enough days for given seq_len and pred_len")
    
        sequences_x = []
        sequences_y = []
    
        for city in range(num_cities):
            city_data = data_3d[city]
            for i in range(num_windows):
                start_index = i
                end_index = i + seq_len + pred_len
                window = city_data[start_index:end_index, :]
                sequences_x.append(window[:seq_len, :])
                sequences_y.append(window[seq_len:, :])
    
        sequences_x = np.array(sequences_x)
        sequences_y = np.array(sequences_y)
    
        sequences_tensor = torch.from_numpy(sequences_x).float()
        targets_tensor = torch.from_numpy(sequences_y).float()
    
        print(f"num_cities: {num_cities}, num_days: {num_days}, num_features: {num_features}")
        print(f"seq_len: {seq_len}, pred_len: {pred_len}")
        print(f"Calculated num_windows: {num_windows}")
        print(f"sequences_x shape: {sequences_x.shape}")
        print(f"sequences_y shape: {sequences_y.shape}")
        print(f"sequences_tensor shape: {sequences_tensor.shape}")
        print(f"targets_tensor shape: {targets_tensor.shape}")
    
        return sequences_tensor, targets_tensor
    
    # Split the data into train, validation, and test sets
    train_days = 160  # 2 years for training
    val_days = 160+60    # 1 year for validation (year 3)
    
    train_data = scaled_data[:, :train_days, :]
    val_data = scaled_data[:, train_days:val_days, :]
    test_data = scaled_data[:, val_days:, :]
    
    # Convert data to PyTorch tensors
    train_sequences, train_targets = create_sequences_3d(train_data, configs.mm_seq_len, configs.mm_pred_len)
    val_sequences,   val_targets   = create_sequences_3d(val_data,   configs.mm_seq_len, configs.mm_pred_len)
    test_sequences,  test_targets  = create_sequences_3d(test_data,  configs.mm_seq_len, configs.mm_pred_len)
    
    train_dataset = TensorDataset(train_sequences, train_targets)
    val_dataset   = TensorDataset(val_sequences,   val_targets)
    test_dataset  = TensorDataset(test_sequences,  test_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size_mmformer, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=configs.batch_size_mmformer, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=configs.batch_size_mmformer, pin_memory=True)
    device = configs.device

    all_predictions = []
    all_targets = []
    
    # Calculating evaluation metrics
    total_mape, total_mse, total_mae, batches_count = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
    
            preds = mmformer_model.forward_autoregressive(
                src=batch_inputs,
                pred_len=configs.mm_pred_len,
                sampling_prob=0.0
            )  # shape: (B, pred_len, feature_size)
            
            all_predictions.append(preds.cpu())
            all_targets.append(batch_targets.cpu())
    
            B = batch_inputs.size(0)
            F_dim = batch_inputs.size(2) 
            pred_len = configs.mm_pred_len 
            
            preds_2d = preds.reshape(B * configs.mm_pred_len, F_dim).cpu().numpy()
            targets_2d = batch_targets.reshape(B * configs.mm_pred_len, F_dim).cpu().numpy()
    
            preds_torch = torch.tensor(preds_2d, dtype=torch.float32, device=device).reshape(B, configs.mm_pred_len, F_dim)
            targets_torch = torch.tensor(targets_2d, dtype=torch.float32, device=device).reshape(B, configs.mm_pred_len, F_dim)
    
            mse_val = F.mse_loss(preds_torch, targets_torch, reduction='mean').item()
            mae_val = F.l1_loss(preds_torch, targets_torch, reduction='mean').item()
    
            mape_val = eval_metrics(targets_torch, preds_torch).item()
    
            total_mse += mse_val
            total_mae += mae_val
            total_mape += mape_val
            batches_count += 1
    
        avg_mse = total_mse / (batches_count if batches_count > 0 else 1)
        avg_mae = total_mae / (batches_count if batches_count > 0 else 1)
        avg_mape = total_mape / (batches_count if batches_count > 0 else 1)
        
        print(f"[MMformer Test] MSE={avg_mse:.5f}, MAE={avg_mae:.5f}, MAPE={avg_mape:.5f}%")
    
    predictions = torch.cat(all_predictions, dim=0)
    return predictions, test_loader, scaler, all_predictions, all_targets

def reaggregate_predictions(predictions, num_cities, configs):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions).float()
    
    pred_len = configs.mm_pred_len
    feature_size = predictions.shape[-1]
    
    num_windows_per_city = predictions.shape[0] // num_cities
    if num_windows_per_city < 1:
        print(f"Warning: predictions shape is {predictions.shape}，num_cities={num_cities}")
        return predictions
        
    reaggregated_preds = torch.zeros((num_cities, pred_len, feature_size), device=predictions.device)
    for city_idx in range(num_cities):
        city_preds = predictions[city_idx * num_windows_per_city: (city_idx + 1) * num_windows_per_city]
        reaggregated_preds[city_idx] = torch.mean(city_preds, dim=0)
    
    return reaggregated_preds

def prepare_voformer_input(historical_data, predictions, configs):
    if isinstance(historical_data, np.ndarray):
        historical_data = torch.from_numpy(historical_data).float()
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions).float()
    
    print(f"Historical data shape: {historical_data.shape}")
    print(f"Predicting Data Shape: {predictions.shape}")
    
    if historical_data.shape[0] != predictions.shape[0]:
        print("Warning: The number of samples of historical data and forecast data does not match!")
        if historical_data.shape[0] > predictions.shape[0]:
            historical_data = historical_data[:predictions.shape[0]]
        else:
            predictions = predictions[:historical_data.shape[0]]
    
    combined_data = torch.cat([historical_data, predictions], dim=1)
    return combined_data

def perform_clustering_with_voformer(voformer_model, combined_data, configs):
    if isinstance(combined_data, torch.Tensor):
        combined_data_np = combined_data.cpu().numpy()
    else:
        combined_data_np = combined_data
    
    print(f"Clustering input data shape: {combined_data_np.shape}")
    
    features = extract_features_with_voformer(combined_data_np, voformer_model, configs)
    
    pseudo_labels = extreme_clustering(features, neighborhood_radius=configs.ex_neighborhood_radius)
    print(f"Clustering shape: {pseudo_labels.shape}, Clusters amount: {len(np.unique(pseudo_labels))}")
    
    np.save(configs.clustering_results_path, pseudo_labels)
    print(f"Clustering completed and the results saved to {configs.clustering_results_path}")
    
    evaluate_clustering(features, pseudo_labels)
    
    return pseudo_labels

def compute_inertia(features, labels):
    unique_labels = np.unique(labels)
    total_inertia = 0.0
    
    for lbl in unique_labels:
        if lbl == -1:
            continue
        cluster_points = features[labels == lbl]
        if len(cluster_points) < 2:
            continue
        center = np.mean(cluster_points, axis=0)
        inertia = np.sum((cluster_points - center) ** 2)
        total_inertia += inertia
    return total_inertia

def dunn_index(features, labels):
    unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]
    clusters = [features[labels == lbl] for lbl in unique_labels if len(features[labels == lbl]) > 0]
    if len(clusters) < 2:
        return 0
    
    intra_diameter_list = []
    for cluster in clusters:
        max_dist = 0
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                dist = np.linalg.norm(cluster[i] - cluster[j])
                if dist > max_dist:
                    max_dist = dist
        intra_diameter_list.append(max_dist)
    max_intra_diameter = np.max(intra_diameter_list)
    
    inter_distances = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            min_dist = np.inf
            for point_a in clusters[i]:
                for point_b in clusters[j]:
                    dist = np.linalg.norm(point_a - point_b)
                    if dist < min_dist:
                        min_dist = dist
            inter_distances.append(min_dist)
    min_inter_distance = np.min(inter_distances) if len(inter_distances) > 0 else 1e-9
    
    return min_inter_distance / (max_intra_diameter + 1e-9)

def evaluate_clustering(features, labels):
    filtered_features = features[labels != -1]
    filtered_labels = labels[labels != -1]

    # 1) Inertia
    inertia_val = compute_inertia(features, labels)

    # 2) Silhouette Score
    if len(np.unique(filtered_labels)) > 1 and len(filtered_features) > 1:
        silhouette_val = silhouette_score(filtered_features, filtered_labels)
    else:
        silhouette_val = None

    # 3) Davies-Bouldin Index
    if len(np.unique(filtered_labels)) > 1 and len(filtered_features) > 1:
        dbi_val = davies_bouldin_score(filtered_features, filtered_labels)
    else:
        dbi_val = None
    
    # 4) Calinski-Harabasz Index
    if len(np.unique(filtered_labels)) > 1 and len(filtered_features) > 1:
        ch_val = calinski_harabasz_score(filtered_features, filtered_labels)
    else:
        ch_val = None

    # 5) Dunn Index
    dunn_val = dunn_index(features, labels)

    print("=== Clustering metrics evaluation ===")
    print(f"• Inertia (SSE): {inertia_val:.4f}")
    
    print(f"• Dunn Index: {dunn_val:.4f}")
    print("=================\n")

# Voformer-EC visualization
def visualize_clusters(features, pseudo_labels, drought_clusters=None):
    """
    Visualize the clustering results and use PCA to reduce the feature dimension to 2 dimensions
    """
    if len(features.shape) > 2:
        features_reshaped = features.reshape(features.shape[0], -1)
    else:
        features_reshaped = features
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_reshaped)
    
    # Visualizing the distribution of clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=pseudo_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('Cluster distribution of MVformer')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()
    
    # Visualizing drought clusters
    plt.figure(figsize=(10, 8))
    colors = ['red' if label in drought_clusters else 'blue' for label in pseudo_labels]
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.6)
    plt.title('Drought Clusters Distribution of MVformer')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    # Add samples
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Drought areas'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal areas')
    ]
    plt.legend(handles=legend_elements)
    
    # Save
    plt.savefig('MVformer_drought_cluster_visualization.png')
    plt.show()

# MVformer visualization
def visualize_mmformer_predictions(mmformer_model, test_loader, scaler, configs, device, sample=0):
    """
    Visualize MMformer's predictions and actual values。
    """
    # Load best model
    device = configs.device
    model = MMformer(configs).to(device)
    model.load_state_dict(torch.load(configs.save_path, map_location=device))
    model.eval()
    
    # Get sample data in a batch and make predictions
    test_inputs, test_targets = next(iter(test_loader))
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    
    # Predictions from the model
    with torch.no_grad():
        src = test_inputs[:, :-configs.mm_pred_len, :] 
        tgt = test_inputs[:, -configs.mm_seq_len:-configs.mm_pred_len, :] 
        test_outputs = model.forward_autoregressive(
        src=src,
        tgt=tgt,
        pred_len=configs.mm_pred_len,
        sampling_prob=0.0)
    
    # Move data from the GPU to the CPU and convert to numpy arrays for plotting with Matplotlib
    test_inputs = test_inputs.cpu().numpy()
    test_targets = test_targets.cpu().numpy()
    test_outputs = test_outputs.cpu().numpy()
    
    # Draw a graph (you can choose a specific sample to draw)
    sample_index = 0  # select the sample
    input_seq_len = configs.mm_seq_len
    pred_len = configs.mm_pred_len
    
    plt.figure(figsize=(15, 5))
    
    # Plotting the input sequence
    plt.plot(range(input_seq_len), test_inputs[sample_index, :, 0], label='Input Sequence')
    
    # Draw the true target sequence (the target sequence starts after input_seq_len)
    plt.plot(range(input_seq_len, input_seq_len + pred_len), test_targets[sample_index, :, 0], label='True Target')
    
    # Draw the predicted target sequence (the predicted sequence starts after input_seq_len)
    plt.plot(range(input_seq_len, input_seq_len + pred_len), test_outputs[sample_index, :, 0], label='Predicted Target')
    
    plt.title('MVformer Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Feature Value')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Best model estimation
    model.eval()
    
    # Obtain data from the test loader
    test_iter = iter(test_loader)
    inputs, true_values = next(test_iter)
    inputs, true_values = inputs.to(device), true_values.to(device)
    
    # Best model prediction
    with torch.no_grad():
        predictions = model.forward_autoregressive(
            src=inputs,
            pred_len=configs.mm_pred_len,
            sampling_prob=0.0
        )
    
    inputs_np = inputs.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    true_values_np = true_values.cpu().numpy()
    
    # Inverse transform the standardized data
    true_values_np = scaler.inverse_transform(true_values.cpu().numpy().reshape(-1, configs.mm_feature_size)).reshape(true_values.shape)
    predictions_np = scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, configs.mm_feature_size)).reshape(predictions.shape)
    
    # Samples prediction MSE
    mse_list = []
    for i in range(true_values_np.shape[0]):
        mse = ((true_values_np[i] - predictions_np[i]) ** 2).mean()
        mse_list.append(mse)
    
    # Convert mse_list to NumPy array
    mse_array = np.array(mse_list)
    
    # Select the indices of the 5 samples with the lowest MSE
    num_samples = 5
    best_sample_indices = np.argsort(mse_array)[:num_samples]
    
    # Sort the best sample indices based on MSE in descending order
    sorted_sample_indices = best_sample_indices[np.argsort(mse_array[best_sample_indices])[::-1]]
    
    num_features = true_values_np.shape[2]
    
    fig, axes = plt.subplots(num_samples, num_features, figsize=(15, 10))
    
    for i, sample_idx in enumerate(sorted_sample_indices):
        for j in range(num_features):
            ax = axes[i, j]
            ax.plot(true_values_np[sample_idx, :, j], label='True Values' if i == 0 and j == 0 else None)
            ax.plot(predictions_np[sample_idx, :, j], label='Predictions' if i == 0 and j == 0 else None, linestyle='--')
            if i == 0:
                ax.set_title(f'Feature {j}')
            if j == 0:
                ax.set_ylabel(f'Sample {i+1}')
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, borderaxespad=0.1)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the spacing and positioning of the subplots
    plt.show()

def main():
    configs = Config()
    
    if configs.visualize_clusters or configs.visualize_predictions:
        os.makedirs(configs.visualization_output_dir, exist_ok=True)

    data_path = os.path.join(configs.data_dir, configs.data_path)
    if not os.path.exists(data_path):
        print(f"The file {data_path} does not exist, please check if the path is correct")
        return
    data = np.load(data_path)
    print(f'Data shape: {data.shape}')
    
    mmformer_available = False

    if configs.load_mmformer and os.path.exists(configs.save_path):
        mmformer_model = load_mmformer_model(configs)
        mmformer_available = True
        
        if configs.train_mmformer:
            mmformer_model, test_loader, scaler = train_mmformer(data, configs)
    
    elif configs.train_mmformer:
        mmformer_model, test_loader, scaler = train_mmformer(data, configs)
        mmformer_available = True
    
    else:
        print("MMformer is not activated. Skipping MMformer-related operations.")
    
    if mmformer_available:
        # MMformer Prediction
        predictions, test_loader, scaler,all_predictions, all_targets = predict_with_mmformer(mmformer_model, data, configs)
        
        num_cities = data.shape[0]
        reaggregated_preds = reaggregate_predictions(
            torch.cat(all_predictions), num_cities, configs
        )
        
        historical_data = data[:,:, :]
        
        if isinstance(historical_data, np.ndarray):
            historical_data = torch.from_numpy(historical_data).float()
        
        voformer_input = prepare_voformer_input(historical_data, reaggregated_preds, configs)

        # MMformer Visualization prediction results
        visualize_mmformer_predictions(mmformer_model, test_loader, scaler, configs, configs.device, sample=0)
    
    # Voformer
    if configs.train_voformer:
        voformer_model = train_voformer(configs)
    elif configs.load_voformer and os.path.exists(configs.voformer_model_path):
        voformer_model = load_voformer_model(configs)
    else:
        print("The Voformer model was not found. You need to train or provide the model path")
        return
    
    # Extreme Clustering
    if configs.perform_clustering:
        pseudo_labels = perform_clustering_with_voformer(voformer_model, voformer_input, configs)
    elif os.path.exists(configs.clustering_results_path):
        pseudo_labels = load_clustering_results(configs)
    else:
        print("No clustering results were found. Clustering needs to be performed or the clustering result path needs to be provided")
        return
    
    # Extracting drought data and drought_clusters
    drought_data, drought_clusters = extract_drought_data(voformer_input, pseudo_labels)
    
    # Clustering visualization
    if configs.visualize_clusters:
        features = np.load(configs.extracted_features_path)
        visualize_clusters(features, pseudo_labels, drought_clusters)

if __name__ == '__main__':
    main()
