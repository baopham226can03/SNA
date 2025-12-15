"""
LLM-based Dynamic Graph Construction Module

This module uses LLM to build dynamic stock relationship graphs
based on market context, news, and fundamental analysis.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import pandas as pd
from functools import lru_cache
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load from .env file
    print("✓ Loaded environment variables from .env")
except ImportError:
    print("ℹ python-dotenv not installed. Using system environment variables.")
    print("  Install: pip install python-dotenv")


class LLMGraphBuilder:
    """
    Build dynamic stock graphs using LLM reasoning
    
    The graph structure changes over time based on:
    - Market regime (bull/bear/volatile)
    - Sector momentum
    - News events
    - Fundamental correlations
    """
    
    def __init__(
        self,
        llm_provider: str = 'local',  # 'local', 'openai', 'anthropic'
        model_name: str = 'gpt-3.5-turbo',
        cache_dir: str = 'data/graph_cache',
        use_cache: bool = True
    ):
        """
        Args:
            llm_provider: Which LLM to use
            model_name: Model name (for API providers)
            cache_dir: Directory to cache LLM responses
            use_cache: Whether to use cached results
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        
        # Initialize LLM client
        self._init_llm()
        
                # Get API key from environment
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    print("⚠ OPENAI_API_KEY not found in environment!")
                    print("  Create .env file or set: export OPENAI_API_KEY='sk-...'")
                    self.llm_provider = 'rule_based'
                    return
                
                self.llm_client = openai.OpenAI(api_key=api_key)
                print("✓ OpenAI client initialized")
                
            except ImportError:
                print("⚠ OpenAI not installed. Run: pip install openai")
                self.llm_provider = 'rule_based'
        
        elif self.llm_provider == 'anthropic':
            try:
                import anthropic
                
                # Get API key from environment
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    print("⚠ ANTHROPIC_API_KEY not found in environment!")
                    print("  Create .env file or set: export ANTHROPIC_API_KEY='sk-ant-...'")
                    self.llm_provider = 'rule_based'
                    return
                
                self.llm_client = anthropic.Anthropic(api_key=api_key)
                print("✓ Anthropic client initialized")
                
            except ImportError:
                print("⚠ OpenAI not installed. Run: pip install openai")
                self.llm_provider = 'rule_based'
        
        elif self.llm_provider == 'anthropic':
            try:
                import anthropic
                # Set your API key: export ANTHROPIC_API_KEY="sk-..."
                self.llm_client = anthropic.Anthropic()
                print("✓ Anthropic client initialized")
            except ImportError:
                print("⚠ Anthropic not installed. Run: pip install anthropic")
                self.llm_provider = 'rule_based'
        
        elif self.llm_provider == 'local':
            # Use local LLM or rule-based fallback
            print("ℹ Using rule-based graph construction (LLM optional)")
            self.llm_client = None
        
        else:
            print(f"✓ Using rule-based graph construction")
            self.llm_client = None
    
    def build_dynamic_graph(
        self,
        tickers: List[str],
        date: str,
        market_features: Optional[Dict] = None,
        top_k_per_stock: int = 10
    ) -> torch.Tensor:
        """
        Build dynamic adjacency matrix for given date
        
        Args:
            tickers: List of stock tickers
            date: Date string (YYYY-MM-DD)
            market_features: Market context features
            top_k_per_stock: Keep only top-k connections per stock
        
        Returns:
            adj_matrix: (N, N) adjacency matrix with edge weights
        """
        N = len(tickers)
        
        # Check cache first
        cache_key = f"{date}_{N}stocks_{top_k_per_stock}k"
        cached_graph = self._load_from_cache(cache_key)
        if cached_graph is not None:
            return cached_graph
        
        # Get market context
        if market_features is None:
            market_features = self._get_market_context(date)
        
        # Initialize adjacency matrix
        adj_matrix = torch.zeros(N, N)
        
        print(f"\n🔨 Building dynamic graph for {date} ({N} stocks)...")
        
        # Strategy: Build graph in chunks to manage LLM calls
        # Option 1: Sector-based (connect within and between sectors)
        adj_matrix = self._build_sector_aware_graph(
            tickers, date, market_features, top_k_per_stock
        )
        
        # Option 2: LLM-enhanced (if LLM available, refine edges)
        if self.llm_client is not None:
            adj_matrix = self._refine_with_llm(
                adj_matrix, tickers, date, market_features, top_k_per_stock
            )
        
        # Sparsify: Keep top-k edges per node
        adj_matrix = self._sparsify_graph(adj_matrix, top_k_per_stock)
        
        # Save to cache
        self._save_to_cache(cache_key, adj_matrix)
        
        print(f"✓ Graph built: {adj_matrix.nonzero().shape[0]} edges")
        
        return adj_matrix
    
    def _build_sector_aware_graph(
        self,
        tickers: List[str],
        date: str,
        market_features: Dict,
        top_k: int
    ) -> torch.Tensor:
        """Build base graph using sector relationships"""
        N = len(tickers)
        adj = torch.zeros(N, N)
        
        # Group by sector
        sector_groups = {}
        for i, ticker in enumerate(tickers):
            sector = self.sector_info.get(ticker, 'Unknown')
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(i)
        
        # Within-sector connections (high weight)
        for sector, indices in sector_groups.items():
            for i in indices:
                for j in indices:
                    if i != j:
                        adj[i, j] = 0.8  # Strong within-sector connection
        
        # Cross-sector connections (moderate weight)
        # Related sectors get higher weights
        sector_relations = {
            'Technology': ['Communication Services', 'Consumer Discretionary'],
            'Financials': ['Real Estate', 'Consumer Discretionary'],
            'Health Care': ['Consumer Staples'],
            'Energy': ['Materials', 'Industrials'],
            'Industrials': ['Materials', 'Energy'],
        }
        
        for sector1, related_sectors in sector_relations.items():
            if sector1 in sector_groups:
                for sector2 in related_sectors:
                    if sector2 in sector_groups:
                        for i in sector_groups[sector1]:
                            for j in sector_groups[sector2]:
                                adj[i, j] = 0.4  # Moderate cross-sector
        
        # Market regime adjustment
        regime = market_features.get('regime', 'neutral')
        if regime == 'risk_off':
            # Increase defensive sector connections
            defensive = ['Consumer Staples', 'Health Care', 'Utilities']
            for sector in defensive:
                if sector in sector_groups:
                    for i in sector_groups[sector]:
                        for j in sector_groups[sector]:
                            if i != j:
                                adj[i, j] *= 1.3
        
        elif regime == 'risk_on':
            # Increase growth sector connections
            growth = ['Technology', 'Consumer Discretionary', 'Communication Services']
            for sector in growth:
                if sector in sector_groups:
                    for i in sector_groups[sector]:
                        for j in sector_groups[sector]:
                            if i != j:
                                adj[i, j] *= 1.3
        
        return adj
    
    def _refine_with_llm(
        self,
        adj_matrix: torch.Tensor,
        tickers: List[str],
        date: str,
        market_features: Dict,
        top_k: int
    ) -> torch.Tensor:
        """Refine graph edges using LLM reasoning"""
        
        # Sample edges to query (too expensive to query all pairs)
        # Focus on borderline edges (weight ~0.3-0.6)
        edges_to_refine = self._select_edges_for_llm(adj_matrix, tickers, n_edges=50)
        
        print(f"   Refining {len(edges_to_refine)} edges with LLM...")
        
        for i, j in edges_to_refine:
            weight = self._query_llm_for_edge_weight(
                tickers[i], tickers[j], date, market_features
            )
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight  # Symmetric
        
        return adj_matrix
    
    def _select_edges_for_llm(
        self,
        adj_matrix: torch.Tensor,
        tickers: List[str],
        n_edges: int
    ) -> List[Tuple[int, int]]:
        """Select edges that would benefit most from LLM refinement"""
        
        # Focus on edges with moderate weights (uncertain connections)
        weights = adj_matrix.clone()
        weights[torch.eye(len(tickers)).bool()] = 0  # Remove self-loops
        
        # Select edges with weights in [0.3, 0.6] range
        mask = (weights > 0.3) & (weights < 0.6)
        candidates = mask.nonzero()
        
        # Sample n_edges
        if len(candidates) > n_edges:
            indices = torch.randperm(len(candidates))[:n_edges]
            candidates = candidates[indices]
        
        return [(i.item(), j.item()) for i, j in candidates]
    
    def _query_llm_for_edge_weight(
        self,
        ticker1: str,
        ticker2: str,
        date: str,
        market_features: Dict
    ) -> float:
        """Query LLM for relationship strength between two stocks"""
        
        sector1 = self.sector_info.get(ticker1, 'Unknown')
        sector2 = self.sector_info.get(ticker2, 'Unknown')
        
        prompt = f"""Analyze the relationship between two stocks for portfolio construction:

Stock A: {ticker1} (Sector: {sector1})
Stock B: {ticker2} (Sector: {sector2})
Date: {date}
Market Context: {market_features.get('regime', 'neutral')}

Consider:
1. Are they in the same or related sectors?
2. Do they share supply chain dependencies?
3. Are they substitutes or complements?
4. Do they respond similarly to market events?
5. Are they held by similar funds?

Rate the connection strength from 0 to 1:
- 0.0: No meaningful relationship
- 0.3: Weak relationship (different sectors, some macro correlation)
- 0.5: Moderate relationship (related sectors or supply chain)
- 0.7: Strong relationship (same sector or close substitutes)
- 1.0: Very strong relationship (direct competitors or partners)

Respond with ONLY a number between 0 and 1."""

        try:
            if self.llm_provider == 'openai':
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=10
                )
                weight_str = response.choices[0].message.content.strip()
                weight = float(weight_str)
            
            elif self.llm_provider == 'anthropic':
                response = self.llm_client.messages.create(
                    model=self.model_name,
                    max_tokens=10,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                weight_str = response.content[0].text.strip()
                weight = float(weight_str)
            
            else:
                # Fallback: Use sector similarity
                weight = 0.7 if sector1 == sector2 else 0.3
            
            return max(0.0, min(1.0, weight))  # Clamp to [0, 1]
        
        except Exception as e:
            print(f"   ⚠ LLM query failed: {e}, using fallback")
            return 0.5  # Default moderate weight
    
    def _get_market_context(self, date: str) -> Dict:
        """Get market context for the given date"""
        
        # Simplified market regime detection
        # In production, load VIX, S&P 500 returns, etc.
        
        context = {
            'date': date,
            'regime': 'neutral',  # 'risk_on', 'risk_off', 'volatile', 'neutral'
            'vix': 18.0,
            'sp500_change': 0.0
        }
        
        # TODO: Load actual market data
        # For now, use simple heuristics
        
        return context
    
    def _sparsify_graph(self, adj_matrix: torch.Tensor, top_k: int) -> torch.Tensor:
        """Keep only top-k edges per node"""
        N = adj_matrix.shape[0]
        sparse_adj = torch.zeros_like(adj_matrix)
        
        for i in range(N):
            # Get top-k neighbors for node i
            weights = adj_matrix[i].clone()
            weights[i] = -1  # Exclude self
            
            if top_k < N - 1:
                topk_values, topk_indices = torch.topk(weights, top_k)
                sparse_adj[i, topk_indices] = topk_values
            else:
                sparse_adj[i] = weights
        
        # Make symmetric
        sparse_adj = (sparse_adj + sparse_adj.T) / 2
        
        return sparse_adj
    
    def _load_from_cache(self, cache_key: str) -> Optional[torch.Tensor]:
        """Load cached graph"""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pt"
        if cache_file.exists():
            print(f"   ✓ Loaded from cache: {cache_key}")
            return torch.load(cache_file, weights_only=False)
        return None
    
    def _save_to_cache(self, cache_key: str, adj_matrix: torch.Tensor):
        """Save graph to cache"""
        if self.use_cache:
            cache_file = self.cache_dir / f"{cache_key}.pt"
            torch.save(adj_matrix, cache_file)


def build_dynamic_graphs_for_dataset(
    dates: List[str],
    tickers: List[str],
    llm_provider: str = 'local',
    top_k: int = 10,
    cache_dir: str = 'data/graph_cache'
) -> Dict[str, torch.Tensor]:
    """
    Pre-build dynamic graphs for all dates in dataset
    
    This is more efficient than building on-the-fly during training
    
    Args:
        dates: List of dates to build graphs for
        tickers: List of stock tickers
        llm_provider: LLM provider to use
        top_k: Top-k edges per stock
        cache_dir: Cache directory
    
    Returns:
        graphs: Dict mapping date -> adjacency matrix
    """
    builder = LLMGraphBuilder(
        llm_provider=llm_provider,
        cache_dir=cache_dir,
        use_cache=True
    )
    
    graphs = {}
    
    print(f"\n{'='*80}")
    print(f"Building dynamic graphs for {len(dates)} dates...")
    print(f"{'='*80}\n")
    
    for date in dates:
        graphs[date] = builder.build_dynamic_graph(
            tickers=tickers,
            date=date,
            top_k_per_stock=top_k
        )
    
    print(f"\n✓ Built {len(graphs)} dynamic graphs")
    
    return graphs
