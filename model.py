"""
HSGNN Model - Hybrid Stock Graph Neural Network

Implementation based on paper: "Modeling hybrid firm relationships with graph 
neural networks for stock investment decisions"

Architecture:
1. Structure-Aware Implicit Graph Learning (Section 4.1)
2. Explicit Graph Attention Learning (Section 4.2)  
3. Hybrid GNN Encoder (Section 4.3)
4. Prediction Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv


class StructureAwareImplicitGraphLearning(nn.Module):
    """
    Module 1: Structure-Aware Implicit Graph Learning (Section 4.1)
    
    This module learns a hidden graph structure from risk features (X_barra)
    with guidance from money flow graph and applies dual-path message passing.
    
    Key components:
    - Attention-based graph learning (Eq. 3)
    - Edge filtering & sign consistency with money flow (around Eq. 2-3)
    - Dual-path message passing for positive/negative edges (Eq. 4-9)
    """
    
    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 num_heads=4,
                 top_k=10,
                 alpha_threshold=0.3):
        """
        Args:
            input_dim: dimension of risk features
            hidden_dim: hidden dimension for embeddings
            num_heads: number of attention heads
            top_k: keep top-k edges per node
            alpha_threshold: threshold for edge filtering with money flow
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.alpha_threshold = alpha_threshold
        
        # Transform risk features to embeddings (before Eq. 3)
        self.risk_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism for graph learning (Eq. 3)
        # α_ij = σ(w^T [h_i || h_j])
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Dual-path message passing networks
        # Positive path (Eq. 4-6)
        self.msg_pos = nn.Linear(hidden_dim, hidden_dim)
        self.update_pos = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Negative path (Eq. 7-9)
        self.msg_neg = nn.Linear(hidden_dim, hidden_dim)
        self.update_neg = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Combine positive and negative embeddings
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def compute_attention_scores(self, h):
        """
        Compute pairwise attention scores (Eq. 3)
        
        α_ij = σ(w^T [h_i || h_j])
        
        Args:
            h: node embeddings (B, N, D)
        
        Returns:
            attention: attention scores (B, N, N)
        """
        B, N, D = h.shape
        
        # Expand for pairwise concatenation
        h_i = h.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)  # (B, N, N, D)
        
        # Concatenate [h_i || h_j]
        h_concat = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, 2D)
        
        # Compute attention scores (Eq. 3)
        attention = self.attention_net(h_concat).squeeze(-1)  # (B, N, N)
        attention = torch.sigmoid(attention)
        
        return attention
    
    def edge_filtering_with_money_flow(self, attention, money_flow_graph):
        """
        Edge Filtering & Sign Consistency (around Eq. 2-3 in paper)
        
        Filter edges based on:
        1. Top-K selection
        2. Sign consistency with money flow graph
        
        Args:
            attention: learned attention scores (B, N, N)
            money_flow_graph: money flow correlation matrix (B, N, N)
        
        Returns:
            adj_pos: positive edge adjacency (B, N, N)
            adj_neg: negative edge adjacency (B, N, N)
        """
        B, N, _ = attention.shape
        
        # 1. Top-K filtering: keep top-k strongest connections per node
        # Set diagonal to 0 (no self-loops in learned graph)
        attention_no_diag = attention.clone()
        attention_no_diag[:, torch.arange(N), torch.arange(N)] = 0
        
        # Keep top-k edges per node
        topk_values, topk_indices = torch.topk(attention_no_diag, k=self.top_k, dim=2)
        
        # Create mask for top-k edges
        mask_topk = torch.zeros_like(attention)
        for b in range(B):
            for i in range(N):
                mask_topk[b, i, topk_indices[b, i]] = 1
        
        # Apply top-k mask
        attention_filtered = attention * mask_topk
        
        # 2. Sign consistency check with money flow graph
        # Paper mentions comparing signs of learned edges with money flow
        # If money_flow > threshold: positive relationship expected
        # If money_flow < -threshold: negative relationship expected
        
        money_flow_pos_mask = (money_flow_graph > self.alpha_threshold).float()
        money_flow_neg_mask = (money_flow_graph < -self.alpha_threshold).float()
        
        # Split into positive and negative edges based on money flow guidance
        # Positive edges: high attention AND positive money flow
        adj_pos = attention_filtered * money_flow_pos_mask
        
        # Negative edges: high attention AND negative money flow
        adj_neg = attention_filtered * money_flow_neg_mask
        
        # Normalize adjacency matrices
        # Add small epsilon to avoid division by zero
        deg_pos = adj_pos.sum(dim=2, keepdim=True) + 1e-8
        adj_pos = adj_pos / deg_pos
        
        deg_neg = adj_neg.sum(dim=2, keepdim=True) + 1e-8
        adj_neg = adj_neg / deg_neg
        
        return adj_pos, adj_neg
    
    def dual_path_message_passing(self, h, adj_pos, adj_neg):
        """
        Dual-path message passing (Eq. 4-9)
        
        Positive path (Eq. 4-6):
            m_i^+ = Σ_j∈N_i^+ α_ij^+ W^+ h_j
            h_i^+ = GRU(m_i^+, h_i)
        
        Negative path (Eq. 7-9):
            m_i^- = Σ_j∈N_i^- α_ij^- W^- h_j
            h_i^- = GRU(m_i^-, h_i)
        
        Args:
            h: node embeddings (B, N, D)
            adj_pos: positive adjacency (B, N, N)
            adj_neg: negative adjacency (B, N, N)
        
        Returns:
            h_combined: combined embeddings (B, N, D)
        """
        B, N, D = h.shape
        
        # Positive path (Eq. 4-6)
        # Message aggregation: m_i^+ = Σ_j α_ij^+ W^+ h_j
        h_transformed_pos = self.msg_pos(h)  # (B, N, D)
        m_pos = torch.bmm(adj_pos, h_transformed_pos)  # (B, N, D)
        
        # GRU update: h_i^+ = GRU(m_i^+, h_i)
        h_pos = h.clone()
        for i in range(N):
            h_pos[:, i, :] = self.update_pos(m_pos[:, i, :], h[:, i, :])
        
        # Negative path (Eq. 7-9)
        # Message aggregation: m_i^- = Σ_j α_ij^- W^- h_j
        h_transformed_neg = self.msg_neg(h)  # (B, N, D)
        m_neg = torch.bmm(adj_neg, h_transformed_neg)  # (B, N, D)
        
        # GRU update: h_i^- = GRU(m_i^-, h_i)
        h_neg = h.clone()
        for i in range(N):
            h_neg[:, i, :] = self.update_neg(m_neg[:, i, :], h[:, i, :])
        
        # Combine positive and negative embeddings
        h_concat = torch.cat([h_pos, h_neg], dim=-1)  # (B, N, 2D)
        h_combined = self.combine(h_concat)  # (B, N, D)
        
        return h_combined
    
    def forward(self, x_risk, money_flow_graph):
        """
        Forward pass for implicit graph learning
        
        Args:
            x_risk: risk features (B, T, N, F_risk)
            money_flow_graph: money flow correlation (B, N, N)
        
        Returns:
            h_implicit: node embeddings from implicit graph (B, N, D)
        """
        B, T, N, F_dim = x_risk.shape
        
        # Aggregate temporal information (use last timestep or mean)
        # Here we use the last timestep as in paper
        x_risk_current = x_risk[:, -1, :, :]  # (B, N, F_risk)
        
        # Encode risk features to embeddings
        h = self.risk_encoder(x_risk_current)  # (B, N, D)
        
        # Compute attention-based graph structure (Eq. 3)
        attention = self.compute_attention_scores(h)  # (B, N, N)
        
        # Edge filtering & sign consistency with money flow
        adj_pos, adj_neg = self.edge_filtering_with_money_flow(
            attention, money_flow_graph
        )
        
        # Dual-path message passing (Eq. 4-9)
        h_implicit = self.dual_path_message_passing(h, adj_pos, adj_neg)
        
        return h_implicit


class ExplicitGraphAttentionLearning(nn.Module):
    """
    Module 2: Explicit Graph Attention Learning (Section 4.2)
    
    Uses alpha158 features and sector adjacency matrix (supply chain proxy)
    with Graph Attention Network (GAT).
    
    Based on Eq. 10-12:
        α_ij = exp(LeakyReLU(a^T [W h_i || W h_j])) / Σ_k exp(...)
        h_i' = σ(Σ_j∈N_i α_ij W h_j)
    """
    
    def __init__(self, 
                 input_dim,
                 hidden_dim,
                 num_layers=2,
                 num_heads=4,
                 dropout=0.1):
        """
        Args:
            input_dim: dimension of alpha158 features
            hidden_dim: hidden dimension
            num_layers: number of GAT layers
            num_heads: number of attention heads
            dropout: dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers (Eq. 10-12)
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer
                self.gat_layers.append(
                    GATConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True  # Concatenate heads
                    )
                )
            else:
                # Subsequent layers
                self.gat_layers.append(
                    GATConv(
                        in_channels=hidden_dim,  # Input is concatenated from previous layer
                        out_channels=hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=False if i == num_layers - 1 else True  # Average on last layer
                    )
                )
        
        # Project output to ensure correct dimension
        # If last layer uses concat=False, output is hidden_dim // num_heads
        last_layer_out_dim = (hidden_dim // num_heads) if num_layers > 0 and not self.gat_layers[-1].concat else hidden_dim
        self.output_proj = nn.Linear(last_layer_out_dim, hidden_dim) if num_layers > 0 else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_alpha, sector_graph):
        """
        Forward pass for explicit graph learning
        
        Args:
            x_alpha: alpha158 features (B, T, N, F_alpha)
            sector_graph: sector adjacency matrix (B, N, N)
        
        Returns:
            h_explicit: node embeddings from explicit graph (B, N, D)
        """
        B, T, N, F_dim = x_alpha.shape
        
        # Aggregate temporal information (use last timestep)
        x_alpha_current = x_alpha[:, -1, :, :]  # (B, N, F_alpha)
        
        # Project input features
        h = self.input_proj(x_alpha_current)  # (B, N, D)
        
        # Convert adjacency matrix to edge_index for PyG
        # Process each batch separately
        h_list = []
        for b in range(B):
            h_b = h[b]  # (N, D)
            adj_b = sector_graph[b]  # (N, N)
            
            # Convert to edge_index (COO format)
            edge_index = adj_b.nonzero().t()  # (2, E)
            edge_weight = adj_b[edge_index[0], edge_index[1]]  # (E,)
            
            # Apply GAT layers (Eq. 10-12)
            for i, gat_layer in enumerate(self.gat_layers):
                h_b_new = gat_layer(h_b, edge_index)
                
                # Apply activation and dropout (except last layer)
                if i < self.num_layers - 1:
                    h_b_new = F.elu(h_b_new)
                    h_b_new = self.dropout(h_b_new)
                
                # Residual connection only if dimensions match
                if h_b.shape[-1] == h_b_new.shape[-1]:
                    h_b = h_b + h_b_new
                else:
                    h_b = h_b_new
            
            # Ensure output has correct dimension
            h_b = self.output_proj(h_b)
            
            h_list.append(h_b)
        
        # Stack batch
        h_explicit = torch.stack(h_list, dim=0)  # (B, N, D)
        
        return h_explicit


class HybridGNNEncoder(nn.Module):
    """
    Module 3: Hybrid GNN Encoder (Section 4.3)
    
    Fuses implicit and explicit graph embeddings using gated attention mechanism.
    
    Based on Eq. 13-14:
        g = σ(W_g [h_implicit || h_explicit])
        h_fused = g ⊙ h_implicit + (1 - g) ⊙ h_explicit
    """
    
    def __init__(self, hidden_dim):
        """
        Args:
            hidden_dim: dimension of embeddings
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Gated attention for fusion (Eq. 13)
        # g = σ(W_g [h_implicit || h_explicit])
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Optional: additional transformation after fusion
        self.output_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, h_implicit, h_explicit):
        """
        Fuse implicit and explicit embeddings (Eq. 13-14)
        
        Args:
            h_implicit: embeddings from implicit graph (B, N, D)
            h_explicit: embeddings from explicit graph (B, N, D)
        
        Returns:
            h_fused: fused embeddings (B, N, D)
        """
        # Concatenate embeddings
        h_concat = torch.cat([h_implicit, h_explicit], dim=-1)  # (B, N, 2D)
        
        # Compute gate values (Eq. 13)
        # g = σ(W_g [h_implicit || h_explicit])
        gate = self.gate_net(h_concat)  # (B, N, D)
        
        # Gated fusion (Eq. 14)
        # h_fused = g ⊙ h_implicit + (1 - g) ⊙ h_explicit
        h_fused = gate * h_implicit + (1 - gate) * h_explicit  # (B, N, D)
        
        # Optional transformation
        h_fused = self.output_transform(h_fused)
        
        return h_fused


class HSGNN(nn.Module):
    """
    Complete HSGNN Model
    
    Combines all three modules:
    1. Structure-Aware Implicit Graph Learning
    2. Explicit Graph Attention Learning
    3. Hybrid GNN Encoder
    4. Prediction Layer
    """
    
    def __init__(self,
                 num_alpha_features,
                 num_risk_features,
                 hidden_dim=64,
                 num_gat_layers=2,
                 num_heads=4,
                 top_k=10,
                 dropout=0.1):
        """
        Args:
            num_alpha_features: number of alpha158 features
            num_risk_features: number of risk features (3)
            hidden_dim: hidden dimension for all modules
            num_gat_layers: number of GAT layers
            num_heads: number of attention heads
            top_k: top-k edges for implicit graph
            dropout: dropout rate
        """
        super().__init__()
        
        self.num_alpha_features = num_alpha_features
        self.num_risk_features = num_risk_features
        self.hidden_dim = hidden_dim
        
        # Module 1: Structure-Aware Implicit Graph Learning (Section 4.1)
        self.implicit_graph_module = StructureAwareImplicitGraphLearning(
            input_dim=num_risk_features,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            top_k=top_k
        )
        
        # Module 2: Explicit Graph Attention Learning (Section 4.2)
        self.explicit_graph_module = ExplicitGraphAttentionLearning(
            input_dim=num_alpha_features,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Module 3: Hybrid GNN Encoder (Section 4.3)
        self.hybrid_encoder = HybridGNNEncoder(
            hidden_dim=hidden_dim
        )
        
        # Prediction Layer
        # Maps from fused embeddings to stock returns
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, batch):
        """
        Forward pass of HSGNN
        
        Args:
            batch: dictionary with keys:
                - x_alpha: (B, T, N, F_alpha)
                - x_risk: (B, T, N, F_risk)
                - sector_graph: (B, N, N)
                - money_flow_graph: (B, N, N)
        
        Returns:
            predictions: (B, N) - predicted returns for each stock
        """
        x_alpha = batch['x_alpha']
        x_risk = batch['x_risk']
        sector_graph = batch['sector_graph']
        money_flow_graph = batch['money_flow_graph']
        
        # Module 1: Learn implicit graph from risk features (Section 4.1)
        h_implicit = self.implicit_graph_module(x_risk, money_flow_graph)
        
        # Module 2: Learn from explicit sector graph (Section 4.2)
        h_explicit = self.explicit_graph_module(x_alpha, sector_graph)
        
        # Module 3: Fuse implicit and explicit embeddings (Section 4.3)
        h_fused = self.hybrid_encoder(h_implicit, h_explicit)
        
        # Prediction Layer
        predictions = self.predictor(h_fused).squeeze(-1)  # (B, N)
        
        return predictions
    
    def get_embeddings(self, batch):
        """
        Get intermediate embeddings for analysis
        
        Returns:
            dict with h_implicit, h_explicit, h_fused
        """
        x_alpha = batch['x_alpha']
        x_risk = batch['x_risk']
        sector_graph = batch['sector_graph']
        money_flow_graph = batch['money_flow_graph']
        
        h_implicit = self.implicit_graph_module(x_risk, money_flow_graph)
        h_explicit = self.explicit_graph_module(x_alpha, sector_graph)
        h_fused = self.hybrid_encoder(h_implicit, h_explicit)
        
        return {
            'h_implicit': h_implicit,
            'h_explicit': h_explicit,
            'h_fused': h_fused
        }


def create_model(dataset_info, 
                 hidden_dim=64,
                 num_gat_layers=2,
                 num_heads=4,
                 top_k=10,
                 dropout=0.1):
    """
    Create HSGNN model from dataset info
    
    Args:
        dataset_info: dict with num_alpha_features, num_risk_features
        hidden_dim: hidden dimension
        num_gat_layers: number of GAT layers
        num_heads: number of attention heads
        top_k: top-k edges for implicit graph
        dropout: dropout rate
    
    Returns:
        model: HSGNN model
    """
    model = HSGNN(
        num_alpha_features=dataset_info['num_alpha_features'],
        num_risk_features=dataset_info['num_risk_features'],
        hidden_dim=hidden_dim,
        num_gat_layers=num_gat_layers,
        num_heads=num_heads,
        top_k=top_k,
        dropout=dropout
    )
    
    return model


if __name__ == '__main__':
    """Test the model"""
    print("Testing HSGNN Model...")
    
    # Create dummy data
    B, T, N, F_alpha, F_risk = 2, 300, 100, 158, 3
    
    batch = {
        'x_alpha': torch.randn(B, T, N, F_alpha),
        'x_risk': torch.randn(B, T, N, F_risk),
        'sector_graph': torch.randint(0, 2, (B, N, N)).float(),
        'money_flow_graph': torch.randn(B, N, N),
        'y_target': torch.randn(B, N),
        'mask': torch.ones(B, N)
    }
    
    print("\n" + "="*80)
    print("Input shapes:")
    print("="*80)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {value.shape}")
    
    # Create model
    dataset_info = {
        'num_alpha_features': F_alpha,
        'num_risk_features': F_risk,
        'num_stocks': N
    }
    
    model = create_model(dataset_info, hidden_dim=64)
    
    print("\n" + "="*80)
    print("Model architecture:")
    print("="*80)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Forward pass
    print("\n" + "="*80)
    print("Testing forward pass...")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
        print(f"Output shape: {predictions.shape}")
        print(f"Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # Get embeddings
        embeddings = model.get_embeddings(batch)
        print(f"\nEmbedding shapes:")
        for key, value in embeddings.items():
            print(f"  {key:20s}: {value.shape}")
    
    print("\n✓ Model test completed successfully!")
