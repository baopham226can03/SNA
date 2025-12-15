"""
HSGNN with LLM-based Dynamic Graph Construction

This is an enhanced version of the original HSGNN model (model.py)
that uses LLM to build dynamic stock relationship graphs.

Key Differences from Original:
1. Graph structure changes over time (not fixed sector/money-flow graphs)
2. LLM reasoning captures complex, non-obvious relationships
3. Adapts to market regime and news events
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

# Import original HSGNN components
from model import (
    StructureAwareImplicitGraphLearning,
    HybridGNNEncoder,
    HSGNN
)

from llm_graph_builder import LLMGraphBuilder


class DynamicExplicitGraphAttentionLearning(nn.Module):
    """
    Enhanced Explicit Graph Module with Dynamic LLM-generated graphs
    
    Unlike the original that uses fixed sector graphs, this version:
    - Uses LLM to build graphs dynamically per time period
    - Adapts graph structure based on market context
    - Captures time-varying stock relationships
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_llm: bool = True,
        llm_provider: str = 'local',
        graph_cache_dir: str = 'data/graph_cache'
    ):
        """
        Args:
            input_dim: Input feature dimension (F_alpha)
            hidden_dim: Hidden dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_llm: Whether to use LLM for dynamic graph
            llm_provider: LLM provider ('local', 'openai', 'anthropic')
            graph_cache_dir: Directory to cache LLM-generated graphs
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_llm = use_llm
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers (same as original)
        from torch_geometric.nn import GATConv
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gat_layers.append(
                    GATConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True
                    )
                )
            else:
                self.gat_layers.append(
                    GATConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=False if i == num_layers - 1 else True
                    )
                )
        
        # Output projection
        last_layer_out_dim = (hidden_dim // num_heads) if num_layers > 0 and not self.gat_layers[-1].concat else hidden_dim
        self.output_proj = nn.Linear(last_layer_out_dim, hidden_dim) if num_layers > 0 else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        
        # LLM Graph Builder
        if self.use_llm:
            self.graph_builder = LLMGraphBuilder(
                llm_provider=llm_provider,
                cache_dir=graph_cache_dir,
                use_cache=True
            )
            print("✓ DynamicExplicitGAT initialized with LLM graph builder")
        else:
            self.graph_builder = None
            print("✓ DynamicExplicitGAT initialized (using sector graphs)")
    
    def forward(
        self,
        x_alpha: torch.Tensor,
        sector_graph: torch.Tensor,
        date: Optional[str] = None,
        tickers: Optional[list] = None
    ) -> torch.Tensor:
        """
        Forward pass with dynamic graph construction
        
        Args:
            x_alpha: Alpha158 features (B, T, N, F_alpha)
            sector_graph: Static sector graph (B, N, N) - used as fallback
            date: Current date for LLM graph generation
            tickers: List of tickers for LLM queries
        
        Returns:
            h_explicit: Node embeddings (B, N, D)
        """
        B, T, N, F_dim = x_alpha.shape
        
        # Aggregate temporal information
        x_alpha_current = x_alpha[:, -1, :, :]  # (B, N, F_alpha)
        
        # Project input
        h = self.input_proj(x_alpha_current)  # (B, N, D)
        
        # Build/retrieve dynamic graph
        if self.use_llm and date is not None and tickers is not None:
            # Get LLM-generated dynamic graph
            dynamic_graph = self.graph_builder.build_dynamic_graph(
                tickers=tickers,
                date=date,
                top_k_per_stock=10
            )
            # Expand for batch
            dynamic_graph = dynamic_graph.unsqueeze(0).expand(B, -1, -1).to(h.device)
        else:
            # Fallback to sector graph
            dynamic_graph = sector_graph
        
        # Process each batch with GAT
        h_list = []
        for b in range(B):
            h_b = h[b]  # (N, D)
            adj_b = dynamic_graph[b]  # (N, N)
            
            # Convert to edge_index
            edge_index = adj_b.nonzero().t()  # (2, E)
            edge_weight = adj_b[edge_index[0], edge_index[1]]  # (E,)
            
            # Apply GAT layers
            for i, gat_layer in enumerate(self.gat_layers):
                h_b_new = gat_layer(h_b, edge_index)
                
                # Activation and dropout (except last layer)
                if i < self.num_layers - 1:
                    h_b_new = F.elu(h_b_new)
                    h_b_new = self.dropout(h_b_new)
                
                # Residual connection
                if h_b.shape[-1] == h_b_new.shape[-1]:
                    h_b = h_b + h_b_new
                else:
                    h_b = h_b_new
            
            # Output projection
            h_b = self.output_proj(h_b)
            h_list.append(h_b)
        
        # Stack batch
        h_explicit = torch.stack(h_list, dim=0)  # (B, N, D)
        
        return h_explicit


class HSGNN_LLM_DynamicGraph(HSGNN):
    """
    Enhanced HSGNN with LLM-based Dynamic Graph Construction
    
    This extends the original HSGNN by replacing the fixed explicit graph
    module with a dynamic one that uses LLM reasoning.
    
    Architecture:
    1. Implicit Graph Learning (same as original)
    2. Dynamic Explicit Graph Learning (NEW - LLM-based)
    3. Hybrid Encoder (same as original)
    """
    
    def __init__(
        self,
        num_alpha_features: int = 119,
        num_risk_features: int = 3,
        hidden_dim: int = 64,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        top_k: int = 10,
        dropout: float = 0.1,
        use_llm: bool = True,
        llm_provider: str = 'local',
        graph_cache_dir: str = 'data/graph_cache'
    ):
        """
        Initialize HSGNN with LLM Dynamic Graph
        
        Args:
            num_alpha_features: Number of alpha158 features
            num_risk_features: Number of risk features
            hidden_dim: Hidden dimension
            num_gat_layers: Number of GAT layers
            num_heads: Number of attention heads
            top_k: Top-K edges for implicit graph
            dropout: Dropout rate
            use_llm: Whether to use LLM for dynamic graphs
            llm_provider: LLM provider
            graph_cache_dir: Cache directory for graphs
        """
        # Don't call super().__init__() - we'll rebuild components
        nn.Module.__init__(self)
        
        self.num_alpha_features = num_alpha_features
        self.num_risk_features = num_risk_features
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.use_llm = use_llm
        
        # Module 1: Implicit Graph Learning (same as original)
        self.implicit_graph_module = StructureAwareImplicitGraphLearning(
            input_dim=num_risk_features,
            hidden_dim=hidden_dim,
            top_k=top_k,
            dropout=dropout
        )
        
        # Module 2: Dynamic Explicit Graph Learning (NEW)
        self.explicit_graph_module = DynamicExplicitGraphAttentionLearning(
            input_dim=num_alpha_features,
            hidden_dim=hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_llm=use_llm,
            llm_provider=llm_provider,
            graph_cache_dir=graph_cache_dir
        )
        
        # Module 3: Hybrid Encoder (same as original)
        self.hybrid_encoder = HybridGNNEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Final predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        date: Optional[str] = None,
        tickers: Optional[list] = None
    ) -> torch.Tensor:
        """
        Forward pass with dynamic graph
        
        Args:
            batch: Dictionary with keys:
                - x_alpha: (B, T, N, F_alpha)
                - x_risk: (B, T, N, F_risk)
                - sector_graph: (B, N, N)
                - money_flow_graph: (B, N, N)
                - mask: (B, N)
            date: Current date for LLM graph generation
            tickers: List of tickers
        
        Returns:
            predictions: (B, N) predicted returns
        """
        x_alpha = batch['x_alpha']
        x_risk = batch['x_risk']
        sector_graph = batch['sector_graph']
        money_flow_graph = batch['money_flow_graph']
        
        # Module 1: Implicit Graph Learning
        h_implicit = self.implicit_graph_module(x_risk, money_flow_graph)
        
        # Module 2: Dynamic Explicit Graph Learning (with LLM)
        h_explicit = self.explicit_graph_module(
            x_alpha,
            sector_graph,
            date=date,
            tickers=tickers
        )
        
        # Module 3: Hybrid Fusion
        h_fused = self.hybrid_encoder(h_implicit, h_explicit)
        
        # Final prediction
        predictions = self.predictor(h_fused).squeeze(-1)  # (B, N)
        
        return predictions


def create_model_llm_dynamic(
    num_alpha_features: int = 119,
    num_risk_features: int = 3,
    hidden_dim: int = 64,
    num_gat_layers: int = 2,
    num_heads: int = 4,
    top_k: int = 10,
    dropout: float = 0.1,
    use_llm: bool = True,
    llm_provider: str = 'local',
    device: str = 'cpu'
) -> HSGNN_LLM_DynamicGraph:
    """
    Create HSGNN model with LLM Dynamic Graph
    
    Args:
        num_alpha_features: Number of alpha158 features
        num_risk_features: Number of risk features
        hidden_dim: Hidden dimension
        num_gat_layers: Number of GAT layers
        num_heads: Number of attention heads
        top_k: Top-K edges for implicit graph
        dropout: Dropout rate
        use_llm: Whether to use LLM
        llm_provider: LLM provider
        device: Device to use
    
    Returns:
        model: HSGNN_LLM_DynamicGraph model
    """
    model = HSGNN_LLM_DynamicGraph(
        num_alpha_features=num_alpha_features,
        num_risk_features=num_risk_features,
        hidden_dim=hidden_dim,
        num_gat_layers=num_gat_layers,
        num_heads=num_heads,
        top_k=top_k,
        dropout=dropout,
        use_llm=use_llm,
        llm_provider=llm_provider
    )
    
    model = model.to(device)
    
    return model
