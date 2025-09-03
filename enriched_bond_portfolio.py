#!/usr/bin/env python3
"""
Enhanced Bond Portfolio with Real FRED Data and Detailed Specifications
"""
import pandas as pd
import numpy as np
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import sys
import os
import re
import glob

# Import existing bond/loan classes and functions from test.py
sys.path.append('.')
from bond_utilities import (
    LoanSpec, BulletBondSpec, loan_schedule, loan_IRR_and_MOIC,
    bond_price_from_ytm, bond_ytm_from_price, bond_duration_convexity,
    simulate_random_portfolios_from_returns, best_portfolios, 
    plot_efficient_frontier, print_weights,
    plot_cashflow_waterfall, plot_outstanding_enhanced,
    calculate_year_fraction as year_fraction, calculate_loan_irr_and_moic
)

# Import centralized prepayment risk module
from prepayment_risk_module import (
    PrepaymentRiskEngine, PrepaymentFactors, get_prepayment_factors_from_loan,
    prepayment_engine
)

@dataclass
class EnhancedLoanSpec:
    """Enhanced loan specification with all attributes"""
    loan_id: str
    borrower: str
    sector: str
    principal: float
    rate_type: str  # 'fixed' or 'floating'
    base_rate: str  # 'SOFR', 'EFFR', 'DGS2', 'DGS10', or 'none'
    spread: float
    origination_date: str
    maturity_months: int
    io_months: int
    amort_style: str
    origination_rate: float
    current_spread: float
    ltv: float  # loan-to-value
    dscr: float  # debt service coverage ratio
    credit_rating: str
    prepay_penalty: float
    default_prob: float
    seniority: str
    collateral_type: str
    geography: str
    vintage_yield: float
    coupon_freq: str
    floating_reset_freq: str

@dataclass  
class EnhancedBondSpec:
    """Enhanced bond specification with all attributes"""
    bond_id: str
    issuer: str
    sector: str
    face_value: float
    coupon_rate: float
    rate_type: str  # 'fixed' or 'floating'
    base_rate: str  # 'SOFR', 'EFFR', 'DGS2', 'DGS5', 'DGS10', or 'none'
    spread: float
    issue_date: str
    maturity_date: str
    freq: int  # payment frequency
    day_count: str
    call_protection: int  # years
    credit_rating: str
    duration: float
    convexity: float
    oas_spread: float  # option-adjusted spread
    liquidity_score: float
    seniority: str
    collateral: str
    geography: str
    vintage_yield: float
    callable: str  # 'yes' or 'no'
    puttable: str  # 'yes' or 'no'

def calculate_returns_from_cashflows(loans, bonds, cashflow_dir="cashflows_20250902_161401"):
    """
    Calculate actual IRR/MOIC from generated cashflow files for sophisticated return modeling
    """
    returns_dict = {}
    
    # Process loans
    for loan in loans:
        loan_id = loan.loan_id
        
        # Find the cashflow file - check multiple possible patterns
        possible_files = [
            os.path.join(cashflow_dir, f"{loan_id}_cashflow.txt"),
            os.path.join(cashflow_dir, f"{loan_id}_cashflows.txt")
        ]
        
        # Also check for files that start with loan_id
        if cashflow_dir and os.path.exists(cashflow_dir):
            import glob
            pattern_files = glob.glob(os.path.join(cashflow_dir, f"{loan_id}_*_cashflows.txt"))
            possible_files.extend(pattern_files)
        
        cashflow_file = None
        for pf in possible_files:
            if os.path.exists(pf):
                cashflow_file = pf
                break
        
        if cashflow_file:
            try:
                # Read the actual IRR and MOIC from the cashflow analysis file
                with open(cashflow_file, 'r') as f:
                    content = f.read()
                    
                # Extract both base and adjusted IRRs for proper analysis
                import re
                base_irr_match = re.search(r'IRR \(base\): ([\d.]+)%', content)
                adj_irr_match = re.search(r'IRR \(adjusted\): ([\d.]+)%', content)
                moic_match = re.search(r'MOIC: ([\d.]+)x', content)
                
                if base_irr_match and adj_irr_match and moic_match:
                    base_irr = float(base_irr_match.group(1)) / 100  # Base IRR (before credit losses)
                    adj_irr = float(adj_irr_match.group(1)) / 100   # Adjusted IRR (after credit losses)
                    moic = float(moic_match.group(1))
                    
                    # Use base IRR for portfolio optimization (before credit adjustments)
                    # Credit risk is handled separately in volatility/correlation modeling
                    returns_dict[loan_id] = {
                        'irr': adj_irr,  # Keep adjusted for reference
                        'base_irr': base_irr,
                        'moic': moic,
                        'annual_return': base_irr,  # Use base IRR for portfolio optimization
                        'credit_risk': loan.default_prob,
                        'sector': loan.sector,
                        'rating': loan.credit_rating,
                        'ltv': loan.ltv,
                        'dscr': loan.dscr
                    }
                    
                    print(f"  Base IRR: {base_irr:.2%}, Adjusted IRR: {adj_irr:.2%}, MOIC: {moic:.2f}x for {loan_id}")
                else:
                    raise ValueError("Could not parse IRR/MOIC from cashflow file")
                    
                
                
            except Exception as e:
                print(f"Warning: Could not calculate returns for {loan_id}: {e}")
                # Fallback to vintage yield
                returns_dict[loan_id] = {
                    'irr': loan.vintage_yield,
                    'moic': 1.0 + loan.vintage_yield * (loan.maturity_months / 12),
                    'annual_return': loan.vintage_yield,
                    'credit_risk': loan.default_prob,
                    'sector': loan.sector,
                    'rating': loan.credit_rating,
                    'ltv': loan.ltv,
                    'dscr': loan.dscr
                }
        else:
            # Use vintage yield if no cashflow file
            returns_dict[loan_id] = {
                'irr': loan.vintage_yield,
                'moic': 1.0 + loan.vintage_yield * (loan.maturity_months / 12),
                'annual_return': loan.vintage_yield,
                'credit_risk': loan.default_prob,
                'sector': loan.sector,
                'rating': loan.credit_rating,
                'ltv': loan.ltv,
                'dscr': loan.dscr
            }
    
    # Process bonds (if any)
    for bond in bonds:
        bond_id = bond.bond_id
        # For bonds, use yield-based returns with safe attribute access
        returns_dict[bond_id] = {
            'irr': bond.coupon_rate,
            'moic': 1.0 + bond.coupon_rate * (getattr(bond, 'duration', 5)),
            'annual_return': bond.coupon_rate,
            'credit_risk': getattr(bond, 'default_prob', 0.03),  # Safe access with default
            'sector': bond.sector,
            'rating': getattr(bond, 'credit_rating', 'BBB'),  # Safe access with default
            'duration': getattr(bond, 'duration', 5),
            'convexity': getattr(bond, 'convexity', 25)
        }
    
    return returns_dict

def calculate_private_loan_volatilities(returns_dict, data_source='deals_data'):
    """
    Calculate sophisticated volatility estimates for private loans using multiple risk factors
    """
    volatilities = {}
    
    if data_source == 'deals_data':
        # For private loans, use multi-factor volatility model
        for asset_id, metrics in returns_dict.items():
            # Base volatility from credit risk and sector
            base_vol = 0.05  # 5% base volatility for private loans
            
            # Credit risk adjustment
            credit_risk_factor = metrics['credit_risk'] * 2.0  # Default prob impact
            
            # Sector volatility adjustment
            sector_vol_map = {
                'Consumer Staples': 0.08,
                'Energy (Midstream)': 0.15, 
                'Healthcare Services': 0.12,
                'Transportation & Logistics': 0.14,
                'Industrial Services': 0.11,
                'Industrials': 0.10,
                'Retail (Apparel)': 0.16,
                'Semicap Supply Chain': 0.18,
                'Retail (General)': 0.15,
                'Materials': 0.13
            }
            
            sector_vol = sector_vol_map.get(metrics['sector'], 0.12)
            
            # LTV/DSCR adjustment
            leverage_factor = max(0.01, (metrics['ltv'] / 0.75) * 0.03)  # Higher LTV = higher vol
            dscr_factor = max(0.01, (1.5 / metrics['dscr']) * 0.02)  # Lower DSCR = higher vol
            
            # Rating adjustment
            rating_vol_map = {
                'AAA': 0.02, 'AA': 0.03, 'A': 0.04, 'BBB': 0.06,
                'BB+': 0.08, 'BB': 0.09, 'BB-': 0.10, 
                'B+': 0.12, 'B': 0.14, 'B-': 0.16
            }
            
            rating_vol = rating_vol_map.get(metrics['rating'], 0.12)
            
            # Combine factors more realistically (not additive)
            # Use max of factors rather than sum to avoid unrealistic high volatility
            fundamental_vol = max(sector_vol, rating_vol)  # Take the higher of sector or rating vol
            adjustment_vol = (credit_risk_factor + leverage_factor + dscr_factor) / 3  # Average the adjustments
            
            total_volatility = base_vol + fundamental_vol + adjustment_vol
            
            volatilities[asset_id] = min(total_volatility, 0.25)  # More realistic cap at 25%
            
    else:
        # For synthetic data, use simpler model
        for asset_id, metrics in returns_dict.items():
            if 'duration' in metrics:  # Bond
                volatilities[asset_id] = 0.08 + (metrics['duration'] * 0.01)
            else:  # Loan
                volatilities[asset_id] = 0.10 + (metrics['credit_risk'] * 1.5)
    
    return volatilities

def create_correlation_matrix(returns_dict, data_source='deals_data'):
    """
    Create realistic correlation matrix for private loan portfolio
    """
    asset_ids = list(returns_dict.keys())
    n_assets = len(asset_ids)
    
    # Initialize correlation matrix
    corr_matrix = np.eye(n_assets)
    
    if data_source == 'deals_data':
        # Private loan correlations based on sector, rating, and macro factors
        for i, asset_i in enumerate(asset_ids):
            for j, asset_j in enumerate(asset_ids):
                if i != j:
                    metrics_i = returns_dict[asset_i]
                    metrics_j = returns_dict[asset_j]
                    
                    # Base correlation (market risk)
                    base_corr = 0.15
                    
                    # Sector correlation
                    if metrics_i['sector'] == metrics_j['sector']:
                        sector_corr = 0.25
                    else:
                        sector_corr = 0.05
                    
                    # Rating correlation (credit cycle)
                    rating_diff = abs(ord(metrics_i['rating'][0]) - ord(metrics_j['rating'][0]))
                    rating_corr = max(0.02, 0.15 - (rating_diff * 0.03))
                    
                    # Geographic correlation (if different)
                    geo_corr = 0.08 if asset_i.startswith('DEAL_') and asset_j.startswith('PORT_') else 0.12
                    
                    # Combined correlation
                    total_corr = min(0.6, base_corr + sector_corr + rating_corr + geo_corr)
                    corr_matrix[i, j] = total_corr
                    corr_matrix[j, i] = total_corr
    else:
        # Synthetic data correlations
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                corr = np.random.uniform(0.1, 0.4)  # Random correlations
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
    
    return pd.DataFrame(corr_matrix, index=asset_ids, columns=asset_ids)

def simulate_loan_paths(loans, returns_dict, volatilities, rates_df, num_simulations=10000, time_horizon_years=5):
    """
    Simulate multiple stochastic paths for each individual loan with credit events
    """
    results = {}
    
    for loan in loans:
        loan_id = loan.loan_id
        paths = []
        
        # Loan parameters
        base_return = returns_dict[loan_id]['annual_return']
        volatility = volatilities[loan_id]
        default_prob = loan.default_prob
        recovery_rate = 0.40  # 40% recovery on default
        
        print(f"Simulating {num_simulations} FIXED INCOME paths for {loan_id} ({loan.borrower})")
        print(f"  Base coupon: {base_return:.2%}, Default prob: {default_prob:.1%}")
        
        for sim in range(num_simulations):
            np.random.seed(42 + sim)  # Different seed per simulation
            
            # Initialize loan state for fixed income simulation
            outstanding_principal = loan.principal
            cumulative_cashflow = 0.0
            defaulted = False
            prepaid = False
            current_spread = getattr(loan, 'current_spread', 0.02)
            
            # Initialize SOFR path for floating rate loans using actual FRED data
            if loan.rate_type == 'floating':
                # Get current SOFR from FRED data dynamically
                try:
                    if rates_df is not None:
                        current_sofr = rates_df['SOFR'].dropna().iloc[-1]
                    else:
                        # Fallback to current EFFR
                        current_sofr = 0.0433
                except:
                    # Final fallback
                    current_sofr = 0.0433
            else:
                current_sofr = 0.0
            
            path_cashflows = []
            
            for year in range(time_horizon_years):
                if defaulted or prepaid or outstanding_principal <= 0:
                    # No more cashflows after default/prepayment/maturity
                    annual_cashflow = 0.0
                else:
                    # 1. Hazard-based default modeling (proper survival analysis)
                    # Convert annual PD to hazard rate: λ = -ln(1-PD)
                    hazard_rate = -np.log(1 - default_prob) if default_prob < 0.99 else 3.0
                    survival_prob = np.exp(-hazard_rate)  # Survival probability for this year
                    
                    if np.random.random() > survival_prob:
                        # Default occurs - apply recovery 
                        recovery_amount = outstanding_principal * recovery_rate
                        annual_cashflow = recovery_amount
                        outstanding_principal = 0.0
                        defaulted = True
                        
                    else:
                        # 2. Use centralized prepayment risk module
                        loan_factors = get_prepayment_factors_from_loan(loan)
                        prepayment_result = prepayment_engine.simulate_prepayment_event(
                            outstanding_principal, loan_factors, loan.prepay_penalty, time_period=1.0
                        )
                        
                        if prepayment_result['prepayment_occurred']:
                            # Prepayment occurs - use centralized calculation
                            annual_interest = outstanding_principal * base_return
                            annual_cashflow = annual_interest + prepayment_result['net_cashflow']
                            outstanding_principal = prepayment_result['remaining_principal']
                            prepaid = True
                            
                        else:
                            # 3. Normal operation - proper cashflow calculation
                            
                            # Update SOFR path for floating rate loans
                            if loan.rate_type == 'floating':
                                # SOFR follows realistic path with mean reversion
                                long_term_sofr = 0.035  # Long-term SOFR target
                                mean_reversion_speed = 0.1
                                sofr_vol = 0.012  # 120bps annual volatility
                                
                                # Mean-reverting SOFR process
                                sofr_change = mean_reversion_speed * (long_term_sofr - current_sofr) + np.random.normal(0, sofr_vol)
                                current_sofr = max(0.005, current_sofr + sofr_change)  # Floor at 50bps
                                
                                # Current loan rate = SOFR + spread (reset quarterly)
                                current_loan_rate = current_sofr + loan.spread
                            else:
                                current_loan_rate = base_return  # Fixed rate loans
                            
                            # Calculate interest payment
                            annual_interest = outstanding_principal * current_loan_rate
                            
                            # Principal payment based on actual loan structure
                            io_years = loan.io_months / 12
                            
                            if loan.amort_style == 'interest_only':
                                # Pure IO loan - bullet payment at maturity
                                if year == time_horizon_years - 1:
                                    principal_payment = outstanding_principal
                                    outstanding_principal = 0.0
                                else:
                                    principal_payment = 0.0
                                    
                            elif loan.amort_style == 'annuity':
                                if year < io_years:
                                    # Still in IO period
                                    principal_payment = 0.0
                                else:
                                    # Amortizing period started
                                    years_left_to_amort = time_horizon_years - year
                                    if years_left_to_amort > 0:
                                        principal_payment = outstanding_principal / years_left_to_amort
                                        outstanding_principal = max(0, outstanding_principal - principal_payment)
                                    else:
                                        principal_payment = outstanding_principal
                                        outstanding_principal = 0.0
                            else:
                                # Term loan or other - assume bullet
                                if year == time_horizon_years - 1:
                                    principal_payment = outstanding_principal
                                    outstanding_principal = 0.0
                                else:
                                    principal_payment = 0.0
                            
                            annual_cashflow = annual_interest + principal_payment
                
                path_cashflows.append(annual_cashflow)
                cumulative_cashflow += annual_cashflow
            
            # Calculate IRR and MOIC from actual cashflows (proper fixed income approach)
            total_inflow = sum(path_cashflows)
            if total_inflow > 0 and loan.principal > 0:
                path_moic = total_inflow / loan.principal
                path_irr = (path_moic ** (1/time_horizon_years)) - 1
            else:
                path_moic = 0.0
                path_irr = -1.0  # Total loss
            
            paths.append({
                'simulation': sim,
                'cashflows': path_cashflows,
                'final_irr': path_irr,
                'moic': path_moic,
                'defaulted': defaulted,
                'prepaid': prepaid,
                'final_outstanding': outstanding_principal
            })
        
        # Calculate path statistics
        final_irrs = [p['final_irr'] for p in paths]
        moics = [p['moic'] for p in paths]
        default_rate = sum(1 for p in paths if p['defaulted']) / len(paths)
        prepay_rate = sum(1 for p in paths if p['prepaid']) / len(paths)
        
        results[loan_id] = {
            'loan': loan,
            'paths': paths,
            'stats': {
                'mean_irr': np.mean(final_irrs),
                'median_irr': np.median(final_irrs),
                'irr_vol': np.std(final_irrs),
                'mean_moic': np.mean(moics),
                'default_rate': default_rate,
                'prepay_rate': prepay_rate,
                'percentiles': {
                    'p10': np.percentile(final_irrs, 10),
                    'p25': np.percentile(final_irrs, 25),
                    'p75': np.percentile(final_irrs, 75),
                    'p90': np.percentile(final_irrs, 90)
                }
            }
        }
        
        print(f"  Mean IRR: {results[loan_id]['stats']['mean_irr']:.2%}")
        print(f"  IRR Volatility: {results[loan_id]['stats']['irr_vol']:.2%}")
        print(f"  Default Rate: {default_rate:.1%}, Prepay Rate: {prepay_rate:.1%}")
        
    return results

def create_portfolio_efficient_frontier_from_paths(path_results, returns_dict, risk_free_rate=0.0433, num_portfolios=5000):
    """
    Create efficient frontier using Monte Carlo portfolio combinations from individual loan paths
    """
    loan_ids = list(path_results.keys())
    n_loans = len(loan_ids)
    
    print(f"\\nCreating efficient frontier from {n_loans} loans with {len(path_results[loan_ids[0]]['paths'])} paths each...")
    
    # Extract returns for each simulation across all loans
    n_sims = len(path_results[loan_ids[0]]['paths'])
    
    # Create returns matrix: rows = simulations, columns = loans
    returns_matrix = np.zeros((n_sims, n_loans))
    
    for j, loan_id in enumerate(loan_ids):
        for i in range(n_sims):
            returns_matrix[i, j] = path_results[loan_id]['paths'][i]['final_irr']
    
    # Use base IRRs from the already-parsed returns_dict (leveraging existing logic)
    mean_returns = []
    volatilities = []
    
    for loan_id in loan_ids:
        # Get base IRR from returns_dict which already parsed cashflow files
        if loan_id in returns_dict and 'base_irr' in returns_dict[loan_id]:
            base_return = returns_dict[loan_id]['base_irr']  # Base IRR from cashflow file
        elif loan_id in returns_dict:
            base_return = returns_dict[loan_id]['annual_return']  # Fallback to annual_return
        else:
            # Final fallback - use loan origination rate
            loan = path_results[loan_id]['loan']
            base_return = loan.origination_rate
            print(f"Warning: Using origination rate for {loan_id}")
            
        mean_returns.append(base_return)
        
        # Dynamic volatility cap based on loan credit quality
        loan = path_results[loan_id]['loan']
        default_prob = loan.default_prob
        
        # Volatility ceiling based on credit quality (higher default prob = higher vol cap)
        if default_prob <= 0.02:  # High quality (<=2% default)
            vol_cap = 0.08
        elif default_prob <= 0.05:  # Medium quality (2-5% default)  
            vol_cap = 0.12
        else:  # Lower quality (>5% default)
            vol_cap = 0.18
            
        # Apply cap to Monte Carlo volatility
        mc_vol = path_results[loan_id]['stats']['irr_vol']
        reasonable_vol = min(mc_vol, vol_cap)
        volatilities.append(reasonable_vol)
        
        print(f"  {loan_id}: Base IRR {base_return:.2%}, Vol {reasonable_vol:.2%} (cap: {vol_cap:.2%})")
    
    mean_returns = np.array(mean_returns)
    volatilities = np.array(volatilities)
    
    # Create correlation matrix from loan characteristics
    correlations = np.eye(n_loans)
    for i in range(n_loans):
        for j in range(i+1, n_loans):
            loan_i = path_results[loan_ids[i]]['loan']
            loan_j = path_results[loan_ids[j]]['loan']
            
            # Sector correlation
            if loan_i.sector == loan_j.sector:
                corr = 0.25
            else:
                corr = 0.12
                
            correlations[i,j] = correlations[j,i] = corr
    
    # Create covariance matrix
    vol_matrix = np.outer(volatilities, volatilities)
    covariance_matrix = vol_matrix * correlations
    
    # Generate random portfolio weights
    np.random.seed(42)
    portfolio_results = []
    
    for p in range(num_portfolios):
        # Generate random weights that sum to 1
        weights = np.random.dirichlet(np.ones(n_loans))
        
        # Portfolio statistics using mean-variance
        portfolio_return = np.sum(weights * mean_returns)
        portfolio_variance = weights @ covariance_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        portfolio_results.append({
            'Return': portfolio_return,
            'Volatility': portfolio_vol, 
            'Sharpe': portfolio_sharpe,
            'Weights': {loan_ids[i]: weights[i] for i in range(n_loans)}
        })
    
    # Convert to DataFrame
    portfolios_df = pd.DataFrame(portfolio_results)
    
    # Find best portfolios
    max_sharpe_idx = portfolios_df['Sharpe'].idxmax()
    min_vol_idx = portfolios_df['Volatility'].idxmin()
    
    max_sharpe = {
        'Return': portfolios_df.loc[max_sharpe_idx, 'Return'],
        'Volatility': portfolios_df.loc[max_sharpe_idx, 'Volatility'],
        'Sharpe': portfolios_df.loc[max_sharpe_idx, 'Sharpe'],
        **portfolio_results[max_sharpe_idx]['Weights']
    }
    
    min_vol = {
        'Return': portfolios_df.loc[min_vol_idx, 'Return'],
        'Volatility': portfolios_df.loc[min_vol_idx, 'Volatility'], 
        'Sharpe': portfolios_df.loc[min_vol_idx, 'Sharpe'],
        **portfolio_results[min_vol_idx]['Weights']
    }
    
    print(f"✅ Generated {num_portfolios} portfolio combinations from Monte Carlo paths")
    print(f"✅ Max-Sharpe Portfolio: {max_sharpe['Return']:.2%} return, {max_sharpe['Volatility']:.2%} vol, {max_sharpe['Sharpe']:.2f} Sharpe")
    print(f"✅ Min-Vol Portfolio: {min_vol['Return']:.2%} return, {min_vol['Volatility']:.2%} vol, {min_vol['Sharpe']:.2f} Sharpe")
    
    return portfolios_df, max_sharpe, min_vol

def create_path_simulation_plots(path_results, output_prefix="loan_path_simulation"):
    """
    Create visualizations for loan path simulations
    """
    import matplotlib.pyplot as plt
    
    n_loans = len(path_results)
    
    # Create subplots for each loan
    fig, axes = plt.subplots(2, 5, figsize=(20, 12))
    fig.suptitle('Private Credit Fixed Income Simulations\n(10,000 Cashflow Scenarios per Loan)', fontsize=16)
    
    loan_ids = list(path_results.keys())
    
    for i, loan_id in enumerate(loan_ids):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        paths = path_results[loan_id]['paths']
        stats = path_results[loan_id]['stats']
        loan = path_results[loan_id]['loan']
        
        # Plot distribution of final IRRs
        final_irrs = [p['final_irr'] for p in paths[:1000]]  # Sample 1000 for visualization
        
        ax.hist(final_irrs, bins=50, alpha=0.7, color='blue', density=True)
        ax.axvline(stats['mean_irr'], color='red', linestyle='--', label=f"Mean: {stats['mean_irr']:.1%}")
        ax.axvline(stats['percentiles']['p10'], color='orange', linestyle=':', label=f"P10: {stats['percentiles']['p10']:.1%}")
        ax.axvline(stats['percentiles']['p90'], color='orange', linestyle=':', label=f"P90: {stats['percentiles']['p90']:.1%}")
        
        ax.set_title(f"{loan.borrower}\n({loan.sector})", fontsize=10)
        ax.set_xlabel('Final IRR')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to plots directory
    os.makedirs("plots", exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plots/{output_prefix}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Path simulation plots saved to: {filename}")
    return filename

def get_current_risk_free_rate(rates_df):
    """Get current risk-free rate from FRED data (using EFFR)"""
    if rates_df is None:
        return 0.0433  # Fallback to current EFFR 4.33%
    
    try:
        # Use most recent EFFR (Effective Federal Funds Rate) as risk-free rate
        current_effr = rates_df['EFFR'].dropna().iloc[-1]
        print(f"✅ Using current EFFR as risk-free rate: {current_effr:.2%}")
        return current_effr
    except:
        # Fallback to hardcoded current rate
        return 0.0433

def load_fred_rates():
    """Load FRED interest rate data"""
    try:
        rates_df = pd.read_csv('fred_rates.csv', index_col=0, parse_dates=True)
        print(f"Loaded FRED rates: {rates_df.shape[0]} days, {rates_df.shape[1]} series")
        return rates_df
    except FileNotFoundError:
        print("FRED rates file not found. Please run download_fred_data.py first.")
        return None

def load_loans(data_source='synthetic'):
    """Load enhanced loan specifications from file"""
    loans = []
    
    if data_source == 'deals_data':
        # Load from deals_data CSV files
        import pandas as pd
        
        # Load from enhanced metadata files
        try:
            # Load enhanced portfolio loans
            with open('deals_data/enhanced_portfolio.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('|')
                        if len(parts) >= 24:  # Ensure we have all enhanced fields
                            loan = EnhancedLoanSpec(
                                loan_id=parts[0],
                                borrower=parts[1],
                                sector=parts[2], 
                                principal=float(parts[3]),
                                rate_type=parts[4],
                                base_rate=parts[5],
                                spread=float(parts[6]),
                                origination_date=parts[7],
                                maturity_months=int(parts[8]),
                                io_months=int(parts[9]),
                                amort_style=parts[10],
                                origination_rate=float(parts[11]),
                                current_spread=float(parts[12]),
                                ltv=float(parts[13]),
                                dscr=float(parts[14]),
                                credit_rating=parts[15],
                                prepay_penalty=float(parts[16]),
                                default_prob=float(parts[17]),
                                seniority=parts[18],
                                collateral_type=parts[19],
                                geography=parts[20],
                                vintage_yield=float(parts[21]),
                                coupon_freq=parts[22],
                                floating_reset_freq=parts[23]
                            )
                            loans.append(loan)
                            
            # Load enhanced deal start loans
            try:
                with open('deals_data/enhanced_deal_start.txt', 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split('|')
                            if len(parts) >= 24:  # Ensure we have all enhanced fields
                                loan = EnhancedLoanSpec(
                                    loan_id=parts[0],
                                    borrower=parts[1],
                                    sector=parts[2], 
                                    principal=float(parts[3]),
                                    rate_type=parts[4],
                                    base_rate=parts[5],
                                    spread=float(parts[6]),
                                    origination_date=parts[7],
                                    maturity_months=int(parts[8]),
                                    io_months=int(parts[9]),
                                    amort_style=parts[10],
                                    origination_rate=float(parts[11]),
                                    current_spread=float(parts[12]),
                                    ltv=float(parts[13]),
                                    dscr=float(parts[14]),
                                    credit_rating=parts[15],
                                    prepay_penalty=float(parts[16]),
                                    default_prob=float(parts[17]),
                                    seniority=parts[18],
                                    collateral_type=parts[19],
                                    geography=parts[20],
                                    vintage_yield=float(parts[21]),
                                    coupon_freq=parts[22],
                                    floating_reset_freq=parts[23]
                                )
                                loans.append(loan)
            except Exception as e:
                print(f"Note: Could not load enhanced deal start data: {e}")
                
        except Exception as e:
            print(f"Error loading deals_data: {e}")
            print("Falling back to synthetic data...")
            data_source = 'synthetic'
    
    if data_source == 'synthetic':
        # Original synthetic data loading
        with open('synthetic_data/synthetic_loans.csv', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) >= 23:  # Ensure we have all fields
                        loan = EnhancedLoanSpec(
                            loan_id=parts[0],
                            borrower=parts[1],
                            sector=parts[2], 
                            principal=float(parts[3]),
                            rate_type=parts[4],
                            base_rate=parts[5],
                            spread=float(parts[6]),
                            origination_date=parts[7],
                            maturity_months=int(parts[8]),
                            io_months=int(parts[9]),
                            amort_style=parts[10],
                            origination_rate=float(parts[11]),
                            current_spread=float(parts[12]),
                            ltv=float(parts[13]),
                            dscr=float(parts[14]),
                            credit_rating=parts[15],
                            prepay_penalty=float(parts[16]),
                            default_prob=float(parts[17]),
                            seniority=parts[18],
                            collateral_type=parts[19],
                            geography=parts[20],
                        vintage_yield=float(parts[21]),
                        coupon_freq=parts[22],
                        floating_reset_freq=parts[23] if len(parts) > 23 else 'none'
                    )
                    loans.append(loan)
    
    print(f"Loaded {len(loans)} loan specifications")
    return loans

def load_bonds(data_source='synthetic'):
    """Load enhanced bond specifications from file"""
    bonds = []
    
    if data_source == 'deals_data':
        # Skip loading bonds when using deals_data - portfolio.csv is being used for loans
        print("Using deals_data mode - skipping bond loading (portfolio.csv used for loans)")
        return bonds  # Return empty bonds list
    
    if data_source == 'synthetic':
        # Original synthetic data loading  
        with open('synthetic_data/synthetic_bonds.csv', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) >= 23:  # Ensure we have all fields
                        bond = EnhancedBondSpec(
                            bond_id=parts[0],
                            issuer=parts[1],
                            sector=parts[2],
                        face_value=float(parts[3]),
                        coupon_rate=float(parts[4]),
                        rate_type=parts[5],
                        base_rate=parts[6],
                        spread=float(parts[7]),
                        issue_date=parts[8],
                        maturity_date=parts[9],
                        freq=int(parts[10]),
                        day_count=parts[11],
                        call_protection=int(parts[12]),
                        credit_rating=parts[13],
                        duration=float(parts[14]),
                        convexity=float(parts[15]),
                        oas_spread=float(parts[16]),
                        liquidity_score=float(parts[17]),
                        seniority=parts[18],
                        collateral=parts[19],
                        geography=parts[20],
                        vintage_yield=float(parts[21]),
                        callable=parts[22],
                        puttable=parts[23] if len(parts) > 23 else 'no'
                    )
                    bonds.append(bond)
    
    print(f"Loaded {len(bonds)} bond specifications")
    return bonds

def get_current_rate(base_rate: str, rates_df: pd.DataFrame, date: str = None):
    """Get current floating rate from FRED data"""
    if rates_df is None or base_rate == 'none':
        return 0.0
        
    if date is None:
        date = rates_df.index[-1]  # Most recent
    else:
        date = pd.Timestamp(date)
        
    # Map rate names to FRED columns
    rate_mapping = {
        'SOFR': 'SOFR',
        'EFFR': 'EFFR', 
        'DGS2': 'DGS2',
        'DGS5': 'DGS5',
        'DGS10': 'DGS10',
        'DGS30': 'DGS30'
    }
    
    if base_rate in rate_mapping:
        col = rate_mapping[base_rate]
        if col in rates_df.columns:
            # Get rate closest to date
            idx = rates_df.index.get_indexer([date], method='nearest')[0]
            return rates_df.iloc[idx][col]
    
    # Use last known rate or interpolate instead of arbitrary default
    if rates_df is not None and not rates_df.empty:
        # Use most recent available rate for this series
        available_cols = [c for c in ['SOFR', 'EFFR', 'DGS2', 'DGS5', 'DGS10'] if c in rates_df.columns]
        if available_cols:
            recent_rate = rates_df[available_cols[0]].dropna().iloc[-1]
            return recent_rate
    
    # This should never happen if FRED data is properly loaded
    raise ValueError(f"No FRED rate data available for {base_rate}. Check FRED data loading.")

def calculate_loan_current_rate(loan: EnhancedLoanSpec, rates_df: pd.DataFrame):
    """Calculate current all-in rate for a loan"""
    if loan.rate_type == 'fixed':
        return loan.origination_rate
    else:
        # Floating rate: base_rate + spread
        base_rate = get_current_rate(loan.base_rate, rates_df)
        return base_rate + loan.current_spread

def get_historical_rate(rate_type: str, date: str, rates_df: pd.DataFrame) -> float:
    """Get historical rate from FRED data for a specific date"""
    try:
        target_date = pd.Timestamp(date)
        
        # Find closest available date in FRED data
        available_dates = rates_df.index
        closest_idx = available_dates.get_indexer([target_date], method='nearest')[0]
        closest_date = available_dates[closest_idx]
        
        # Map rate types to FRED columns
        rate_mapping = {
            'DGS1': 'DGS2',   # Use 2Y as proxy for 1Y
            'DGS2': 'DGS2',
            'DGS5': 'DGS5', 
            'DGS10': 'DGS10',
            'DGS30': 'DGS30',
            'SOFR': 'SOFR',
            'EFFR': 'EFFR'
        }
        
        col = rate_mapping.get(rate_type, 'DGS10')
        if col in rates_df.columns:
            rate = rates_df.loc[closest_date, col]
            print(f"Historical {rate_type} on {target_date.strftime('%Y-%m-%d')}: {rate:.3%}")
            return rate
            
    except Exception as e:
        print(f"Error getting historical rate for {rate_type} on {date}: {e}")
    
    # Fallback using current rate if historical lookup fails
    return get_current_rate(rate_type, rates_df)

def calculate_floating_rate_for_period(bond_or_loan, period_date: pd.Timestamp, rates_df: pd.DataFrame) -> float:
    """
    Calculate floating rate for a specific period using forward rate projections when available
    Falls back to current rates if forward projections not available
    """
    
    # Try to use forward rate lookup first
    try:
        from forward_rate_lookup import get_global_forward_rate_lookup
        forward_lookup = get_global_forward_rate_lookup()
        
        if forward_lookup.forward_rates_dir and hasattr(bond_or_loan, 'base_rate'):
            # Use historical for past dates, forward for future dates
            forward_base_rate = forward_lookup.get_forward_rate(
                bond_or_loan.base_rate, 
                period_date.strftime('%Y-%m-%d'),
                rates_df  # Pass FRED data for historical lookups
            )
            spread = bond_or_loan.current_spread if hasattr(bond_or_loan, 'current_spread') else bond_or_loan.spread
            return forward_base_rate + spread
    except Exception as e:
        print(f"Forward rate lookup failed, using current rate fallback: {e}")
    
    # Fallback to current rate approach
    # Determine reset date based on frequency
    if hasattr(bond_or_loan, 'floating_reset_freq'):
        reset_freq = bond_or_loan.floating_reset_freq
    else:
        reset_freq = 'quarterly'  # Default
    
    # Calculate most recent reset date
    if reset_freq == 'monthly':
        reset_date = period_date.replace(day=1)  # First of month
    elif reset_freq == 'quarterly':
        quarter_starts = [1, 4, 7, 10]  # Jan, Apr, Jul, Oct
        current_quarter = ((period_date.month - 1) // 3) * 3 + 1
        reset_date = period_date.replace(month=current_quarter, day=1)
    elif reset_freq == 'semi_annual':
        reset_date = period_date.replace(month=1 if period_date.month <= 6 else 7, day=1)
    elif reset_freq == 'annual':
        reset_date = period_date.replace(month=1, day=1)
    else:
        reset_date = period_date  # Default to period date
    
    # Get base rate at reset date using current rate approach
    if hasattr(bond_or_loan, 'base_rate'):
        base_rate = get_current_rate(bond_or_loan.base_rate, rates_df, reset_date.strftime('%Y-%m-%d'))
        spread = bond_or_loan.current_spread if hasattr(bond_or_loan, 'current_spread') else bond_or_loan.spread
        return base_rate + spread
    
    return 0.05  # Should not reach here

def calculate_bond_current_ytm(bond: EnhancedBondSpec, rates_df: pd.DataFrame):
    """Calculate current YTM for a bond with convexity adjustments using historical data"""
    if bond.rate_type == 'fixed':
        # Get actual historical treasury rate when bond was issued
        issue_date = bond.issue_date
        
        # Determine appropriate benchmark based on bond duration
        if bond.duration <= 3:
            benchmark_rate = 'DGS2'
        elif bond.duration <= 7:
            benchmark_rate = 'DGS5'
        elif bond.duration <= 20:
            benchmark_rate = 'DGS10'
        else:
            benchmark_rate = 'DGS30'
        
        # Get historical rate when issued (vintage treasury)
        vintage_treasury = get_historical_rate(benchmark_rate, issue_date, rates_df)
        
        # Get current treasury rate
        current_treasury = get_current_rate(benchmark_rate, rates_df)
        
        # Interest rate change since issuance
        rate_change = current_treasury - vintage_treasury
        
        # Duration adjustment (first-order price sensitivity)
        duration_adjustment = -rate_change * bond.duration / 100
        
        # Convexity adjustment (second-order) - reduces duration impact for rate increases
        convexity_adjustment = 0.5 * (rate_change ** 2) * bond.convexity / 10000
        
        # Credit spread adjustment based on current market conditions
        current_ig_spread = get_current_rate('IG_CREDIT', rates_df) if 'IG_CREDIT' in rates_df.columns else 0.015
        vintage_ig_spread = get_historical_rate('IG_CREDIT', issue_date, rates_df) if 'IG_CREDIT' in rates_df.columns else 0.012
        credit_spread_change = current_ig_spread - vintage_ig_spread
        
        # Liquidity adjustment
        liquidity_adjustment = bond.oas_spread * (10 - bond.liquidity_score) / 10
        
        # Total yield adjustment
        total_adjustment = (duration_adjustment + convexity_adjustment + 
                          credit_spread_change + liquidity_adjustment)
        
        return bond.vintage_yield + total_adjustment
    else:
        # Floating rate bond - calculate current rate based on most recent reset
        today = pd.Timestamp.today()
        current_rate = calculate_floating_rate_for_period(bond, today, rates_df)
        return current_rate

def calculate_prepayment_speed(loan: EnhancedLoanSpec) -> float:
    """Calculate expected prepayment speed based on loan characteristics"""
    
    # Base prepayment speed (annual CPR - Conditional Prepayment Rate)
    base_speed = 0.05  # 5% annual base rate
    
    # Adjust based on loan-to-value (higher LTV = lower prepayment ability)
    if loan.ltv > 0.8:
        ltv_adjustment = -0.02  # High LTV reduces prepayment
    elif loan.ltv < 0.6:
        ltv_adjustment = 0.015  # Low LTV increases prepayment ability
    else:
        ltv_adjustment = 0
    
    # Adjust based on debt service coverage ratio (higher DSCR = more ability to prepay)
    if loan.dscr > 1.5:
        dscr_adjustment = 0.01  # Strong cash flow increases prepayment
    elif loan.dscr < 1.2:
        dscr_adjustment = -0.015  # Weak cash flow reduces prepayment
    else:
        dscr_adjustment = 0
    
    # Adjust based on credit rating (better credit = more refinancing options)
    rating_adjustments = {
        'A': 0.02, 'A-': 0.015, 'BBB+': 0.01, 'BBB': 0.005, 'BBB-': 0,
        'BB+': -0.005, 'BB': -0.01, 'BB-': -0.015, 
        'B+': -0.02, 'B': -0.025, 'B-': -0.03
    }
    rating_adjustment = rating_adjustments.get(loan.credit_rating, -0.01)
    
    # Sector-based adjustments (some sectors prepay more than others)
    sector_adjustments = {
        'Technology': 0.01,  # Tech companies often have more cash
        'Healthcare': 0.005,
        'Real Estate': -0.005,  # Real estate is illiquid
        'Energy': -0.01,  # Cyclical, less predictable cash flows
        'Transportation': -0.015  # Capital intensive, less cash
    }
    sector_adjustment = sector_adjustments.get(loan.sector, 0)
    
    # Geography adjustments (some markets more active)
    if loan.geography in ['California', 'New York', 'Texas']:
        geo_adjustment = 0.005  # Active markets
    else:
        geo_adjustment = 0
    
    # Calculate final prepayment speed
    total_speed = (base_speed + ltv_adjustment + dscr_adjustment + 
                  rating_adjustment + sector_adjustment + geo_adjustment)
    
    # Ensure reasonable bounds
    return max(0.01, min(0.25, total_speed))  # Between 1% and 25% annual

def apply_prepayment_and_default_adjustments(returns: dict, loans: list, bonds: list):
    """Apply comprehensive prepayment speeds and credit losses to expected returns"""
    adjusted_returns = returns.copy()
    
    # Adjust loan returns for prepayments and defaults
    for loan in loans:
        loan_id = loan.loan_id
        if loan_id in adjusted_returns:
            # Enhanced prepayment modeling based on loan characteristics
            base_prepay_speed = calculate_prepayment_speed(loan)
            
            # Prepayment penalty impact on returns
            if loan.prepay_penalty > 0:
                # Higher penalties discourage prepayment, extending duration
                penalty_effect = loan.prepay_penalty * 0.5  # Penalty provides some income
                # But reduces expected prepayment speed
                adjusted_prepay_speed = base_prepay_speed * (1 - loan.prepay_penalty)
            else:
                penalty_effect = 0
                adjusted_prepay_speed = base_prepay_speed
            
            # Duration impact of prepayment
            # Faster prepayment = shorter duration = less interest rate risk but less total interest
            duration_impact = -adjusted_prepay_speed * 0.3  # Negative impact on yield
            
            # Net prepayment adjustment
            prepay_adjustment = penalty_effect + duration_impact
            
            # Default probability reduces expected return
            default_adjustment = -loan.default_prob * 0.6  # Recovery assumption ~40%
            
            # Combine adjustments
            total_adjustment = prepay_adjustment + default_adjustment
            adjusted_returns[loan_id] = max(0.01, adjusted_returns[loan_id] + total_adjustment)
    
    # Adjust bond returns for credit risk
    for bond in bonds:
        bond_id = bond.bond_id
        if bond_id in adjusted_returns:
            # Credit adjustment based on rating and liquidity
            credit_risk_map = {
                'AAA': 0.0001, 'AA+': 0.0002, 'AA': 0.0003, 'AA-': 0.0005,
                'A+': 0.0008, 'A': 0.0012, 'A-': 0.0018,
                'BBB+': 0.0025, 'BBB': 0.0035, 'BBB-': 0.0050,
                'BB+': 0.0080, 'BB': 0.0120, 'BB-': 0.0180,
                'B+': 0.0250, 'B': 0.0350, 'B-': 0.0500,
                'CCC+': 0.0800, 'CCC': 0.1200, 'CCC-': 0.1800
            }
            
            default_prob = credit_risk_map.get(bond.credit_rating, 0.02)
            liquidity_adjustment = (10 - bond.liquidity_score) * 0.0005
            
            total_adjustment = -default_prob * 0.5 - liquidity_adjustment
            adjusted_returns[bond_id] = max(0.01, adjusted_returns[bond_id] + total_adjustment)
    
    return adjusted_returns

# ============================================================================
# Advanced OU Modeling and MTM Return Calculations
# ============================================================================

@dataclass
class OUParameters:
    """Ornstein-Uhlenbeck process parameters for yield modeling"""
    mu: float      # Long-term mean level (%)
    kappa: float   # Mean reversion strength (higher = faster reversion)
    sigma: float   # Volatility (%)
    
# Market-differentiated OU parameters
OU_PARAMS = {
    'IG': OUParameters(mu=0.035, kappa=0.15, sigma=0.008),    # Investment Grade: lower vol, moderate reversion
    'HY': OUParameters(mu=0.065, kappa=0.25, sigma=0.015),   # High Yield: higher vol, faster reversion  
    'TREASURY': OUParameters(mu=0.040, kappa=0.12, sigma=0.006)  # Treasury: lowest vol, slow reversion
}

def classify_credit_quality(rating: str) -> str:
    """Classify bond into IG, HY, or TREASURY based on credit rating"""
    ig_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB']
    if rating in ig_ratings:
        return 'IG'
    else:
        return 'HY'

def simulate_ou_yield_path(y0: float, params: OUParameters, n_periods: int = 252, dt: float = 1/252, random_seed: int = 42) -> np.ndarray:
    """
    Simulate Ornstein-Uhlenbeck mean-reverting yield path
    dy = kappa * (mu - y) * dt + sigma * sqrt(dt) * dW
    
    Args:
        y0: Initial yield level
        params: OU parameters (mu, kappa, sigma)
        n_periods: Number of periods to simulate
        dt: Time step (e.g., 1/252 for daily, 1/12 for monthly)
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    
    yields = np.zeros(n_periods + 1)
    yields[0] = y0
    
    for t in range(n_periods):
        # OU process increment
        drift = params.kappa * (params.mu - yields[t]) * dt
        diffusion = params.sigma * np.sqrt(dt) * np.random.normal()
        
        yields[t + 1] = yields[t] + drift + diffusion
        
        # Ensure non-negative yields
        yields[t + 1] = max(0.0001, yields[t + 1])
    
    return yields[1:]  # Return path without initial value

def calculate_bond_monthly_returns_with_rolldown(bond: EnhancedBondSpec, 
                                               rates_df: pd.DataFrame,
                                               n_months: int = 36) -> pd.DataFrame:
    """
    Calculate monthly bond returns using OU yield simulation with proper MTM valuation
    Trading/Portfolio Context: MTM valuation changes ARE returns (capital gains/losses)
    Total Return = Coupon Income + Capital Appreciation/Depreciation from yield changes
    """
    
    # Classify bond for OU parameters
    credit_class = classify_credit_quality(bond.credit_rating)
    ou_params = OU_PARAMS[credit_class]
    
    # Determine benchmark yield based on duration
    if bond.duration <= 3:
        benchmark = 'DGS2'
    elif bond.duration <= 7:
        benchmark = 'DGS5'  
    elif bond.duration <= 20:
        benchmark = 'DGS10'
    else:
        benchmark = 'DGS30'
    
    initial_treasury = get_historical_rate(benchmark, bond.issue_date, rates_df)
    initial_yield = initial_treasury + bond.oas_spread
    
    # Simulate yield path
    yield_path = simulate_ou_yield_path(initial_yield, ou_params, n_months, dt=1/12, random_seed=42)
    
    # Calculate monthly returns with MTM valuation changes
    monthly_returns = []
    monthly_data = []
    
    base_settle_date = pd.Timestamp(bond.issue_date)
    previous_price = None
    
    for month in range(n_months):
        # Current settlement date (rolls forward each month)
        current_settle = base_settle_date + pd.DateOffset(months=month)
        
        # Skip if past maturity
        maturity_date = pd.Timestamp(bond.maturity_date)
        if current_settle >= maturity_date:
            break
            
        # Current period yield
        y_current = yield_path[month] if month < len(yield_path) else yield_path[-1]
        
        try:
            # Create bond spec for current period (with roll-down)
            current_spec = BulletBondSpec(
                face_value=bond.face_value,
                coupon_rate=bond.coupon_rate,
                payment_frequency=bond.freq,
                maturity_date=bond.maturity_date,
                settlement_date=current_settle.strftime('%Y-%m-%d')
            )
            
            # Calculate current market price and metrics
            current_clean_price, current_dirty_price = bond_price_from_ytm(current_spec, y_current)
            mod_dur_current, _, conv_current = bond_duration_convexity(current_spec, y_current)
            
            # Calculate return components
            
            # 1. Coupon Income (if coupon payment occurs this month)
            months_from_issue = (current_settle.year - base_settle_date.year) * 12 + (current_settle.month - base_settle_date.month)
            is_coupon_month = (months_from_issue % (12 // bond.freq)) == 0
            monthly_coupon_income = (bond.coupon_rate / bond.freq) if is_coupon_month else 0.0
            
            # 2. Capital Appreciation/Depreciation (MTM valuation change)
            if previous_price is not None:
                price_return = (current_clean_price - previous_price) / previous_price
            else:
                price_return = 0.0  # First month
            
            # 3. Roll-down effect (separate from yield change effect)
            # Time decay benefit as bond approaches maturity
            time_to_maturity = (maturity_date - current_settle).days / 365.25
            previous_ttm = (maturity_date - (current_settle - pd.DateOffset(months=1))).days / 365.25 if month > 0 else time_to_maturity
            rolldown_return = 0.0001 * (previous_ttm - time_to_maturity) if month > 0 else 0.0  # Small roll-down benefit
            
            # Total return = Coupon Income + MTM Price Change + Roll-down
            total_return = monthly_coupon_income + price_return + rolldown_return
            
            # Store detailed information
            monthly_data.append({
                'Month': month + 1,
                'SettleDate': current_settle,
                'Yield': y_current,
                'YieldChange': yield_path[month + 1] - y_current if month + 1 < len(yield_path) else 0,
                'ModifiedDuration': mod_dur_current,
                'Convexity': conv_current,
                'CleanPrice': current_clean_price,
                'DirtyPrice': current_dirty_price,
                'CouponIncome': monthly_coupon_income,
                'PriceReturn': price_return,
                'RolldownReturn': rolldown_return,
                'TotalReturn': total_return,
                'TimeToMaturity': time_to_maturity,
                'IsCouponMonth': is_coupon_month
            })
            
            monthly_returns.append(total_return)
            previous_price = current_clean_price  # Store for next period's price return calculation
            
        except Exception as e:
            print(f"Error calculating MTM return for month {month + 1}: {e}")
            monthly_returns.append(0.0)
    
    return pd.DataFrame(monthly_data)

def calculate_loan_monthly_returns_income_based(loan: EnhancedLoanSpec,
                                              rates_df: pd.DataFrame,
                                              n_months: int = 36) -> pd.DataFrame:
    """
    Calculate private credit loan returns based on interest and fee income only
    Principal repayments are treated as return of capital, not return
    """
    
    # Calculate current rate
    if loan.rate_type == 'floating':
        base_rate = get_current_rate(loan.base_rate, rates_df)
        current_rate = base_rate + loan.current_spread
    else:
        current_rate = loan.origination_rate
        
    loan_spec = LoanSpec(
        principal=loan.principal,
        annual_interest_rate=current_rate,
        origination_date=loan.origination_date,
        term_months=loan.maturity_months,
        interest_only_months=loan.io_months,
        amortization_style=loan.amort_style,
        upfront_fee_percent=0.01,
        exit_fee_percent=loan.prepay_penalty / 2
    )
    
    # Generate loan schedule
    schedule = loan_schedule(loan_spec)
    
    # Calculate monthly returns based on interest income only
    monthly_returns_data = []
    
    for i, (date, row) in enumerate(schedule.iterrows()):
        if i == 0:  # Skip initial funding
            continue
            
        outstanding_start = row['OutstandingStart']
        interest_payment = row['Interest']
        fee_payment = row['Fees']
        principal_payment = row['Principal']  # This is return of capital, not return
        
        # Return calculation: (Interest + Fees) / Outstanding_Start
        # Principal repayments do not contribute to return calculation
        if outstanding_start > 0:
            interest_return = interest_payment / outstanding_start
            fee_return = fee_payment / outstanding_start
            total_income_return = interest_return + fee_return
            
            # For floating rate loans, update rate for this period
            if loan.rate_type == 'floating':
                period_rate = calculate_floating_rate_for_period(loan, date, rates_df)
                period_base_rate = get_current_rate(loan.base_rate, rates_df, date.strftime('%Y-%m-%d'))
                period_spread = loan.current_spread
            else:
                period_rate = loan_spec.annual_interest_rate  # Use the loan_spec rate
                period_base_rate = loan_spec.annual_interest_rate
                period_spread = 0.0
        else:
            total_income_return = 0.0
            period_rate = current_rate
            period_base_rate = 0.0
            period_spread = 0.0
        
        monthly_returns_data.append({
            'Month': i,
            'Date': date,
            'OutstandingStart': outstanding_start,
            'InterestPayment': interest_payment,
            'FeePayment': fee_payment,
            'PrincipalPayment': principal_payment,  # Return of capital
            'InterestReturn': interest_return if outstanding_start > 0 else 0,
            'FeeReturn': fee_return if outstanding_start > 0 else 0,
            'TotalIncomeReturn': total_income_return,
            'PeriodRate': period_rate,
            'PeriodBaseRate': period_base_rate,
            'PeriodSpread': period_spread,
            'CreditRating': loan.credit_rating,
            'DefaultProb': loan.default_prob,
            'LTV': loan.ltv,
            'DSCR': loan.dscr
        })
    
    return pd.DataFrame(monthly_returns_data)

def generate_ou_based_daily_returns_for_all_bonds(bonds: list, rates_df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generate OU-based daily returns for ALL bonds using sophisticated mean-reverting yield modeling
    
    Args:
        bonds: List of all bond specifications
        rates_df: FRED rate data
        dates: Date index for return series
        
    Returns:
        DataFrame with OU-generated daily returns for all bonds
    """
    print(f"Generating OU-based returns for {len(bonds)} bonds...")
    
    bond_returns_matrix = pd.DataFrame(index=dates)
    
    for i, bond in enumerate(bonds):
        try:
            # Skip matured bonds
            maturity_date = pd.Timestamp(bond.maturity_date)
            if maturity_date <= pd.Timestamp.today():
                continue
                
            # Classify bond for OU parameters
            credit_class = classify_credit_quality(bond.credit_rating)
            ou_params = OU_PARAMS[credit_class]
            
            # Get initial yield from historical data
            if bond.duration <= 3:
                benchmark = 'DGS2'
            elif bond.duration <= 7:
                benchmark = 'DGS5'
            elif bond.duration <= 20:
                benchmark = 'DGS10'
            else:
                benchmark = 'DGS30'
            
            initial_treasury = get_historical_rate(benchmark, bond.issue_date, rates_df)
            initial_yield = initial_treasury + bond.oas_spread
            
            # Generate OU yield path for the entire date range
            n_periods = len(dates)
            # Use unique random seed for each bond based on bond ID
            bond_seed = 42 + abs(hash(bond.bond_id)) % 1000
            yield_path = simulate_ou_yield_path(
                y0=initial_yield, 
                params=ou_params, 
                n_periods=n_periods, 
                dt=1/252,
                random_seed=bond_seed
            )
            
            # Convert yield changes to bond returns using duration-convexity approximation
            daily_returns = []
            
            for j in range(len(dates)):
                if j == 0:
                    # First day - baseline
                    daily_returns.append(bond.coupon_rate / 252)  # Daily coupon accrual
                else:
                    # Calculate return from yield change
                    y_prev = yield_path[j-1] if j-1 < len(yield_path) else yield_path[-1]
                    y_curr = yield_path[j] if j < len(yield_path) else yield_path[-1]
                    delta_y = y_curr - y_prev
                    
                    # Duration-convexity return approximation
                    coupon_return = bond.coupon_rate / 252  # Daily coupon accrual
                    duration_return = -bond.duration * delta_y / 100  # Duration effect
                    convexity_return = 0.5 * bond.convexity * (delta_y ** 2) / 10000  # Convexity benefit
                    
                    total_return = coupon_return + duration_return + convexity_return
                    daily_returns.append(total_return)
            
            bond_returns_matrix[bond.bond_id] = daily_returns
            
            if (i + 1) % 20 == 0:  # Progress update every 20 bonds
                print(f"  Generated OU returns for {i+1} bonds...")
                
        except Exception as e:
            print(f"Error generating OU returns for {bond.bond_id}: {e}")
            # Fallback to simple daily return
            bond_returns_matrix[bond.bond_id] = bond.vintage_yield / 252
    
    print(f"✅ OU modeling complete for {len(bond_returns_matrix.columns)} active bonds")
    return bond_returns_matrix

def generate_treasury_factor_daily_returns_for_bonds(bonds: list, rates_df: pd.DataFrame, dates: pd.DatetimeIndex, adjusted_returns: dict) -> pd.DataFrame:
    """
    Generate treasury factor model daily returns for bonds (legacy approach)
    Uses real Treasury yield changes as systematic factors
    """
    print(f"Generating treasury factor returns for {len(bonds)} bonds...")
    
    bond_returns_matrix = pd.DataFrame(index=dates)
    
    # Use multiple real market factors for systematic risk
    treasury_2y_changes = rates_df['DGS2'].pct_change().reindex(dates, method='ffill').fillna(0)
    treasury_10y_changes = rates_df['DGS10'].pct_change().reindex(dates, method='ffill').fillna(0)
    sofr_changes = rates_df['SOFR'].pct_change().reindex(dates, method='ffill').fillna(0)
    
    for i, bond in enumerate(bonds):
        bond_id = bond.bond_id
        if bond_id not in adjusted_returns:
            continue
            
        annual_return = adjusted_returns[bond_id]
        daily_return = annual_return / 252
        
        # SCIENTIFIC BOND VOLATILITY MODEL
        duration_vol = bond.duration * 0.004  # ~40bp vol per year of duration
        
        # Credit spread volatility
        credit_vol_map = {
            'AAA': 0.02, 'AA+': 0.025, 'AA': 0.03, 'AA-': 0.035, 'A+': 0.04, 'A': 0.045, 'A-': 0.05,
            'BBB+': 0.06, 'BBB': 0.07, 'BBB-': 0.08, 'BB+': 0.12, 'BB': 0.15, 'BB-': 0.18,
            'B+': 0.22, 'B': 0.25, 'B-': 0.30, 'CCC+': 0.40, 'CCC': 0.50
        }
        credit_vol = credit_vol_map.get(bond.credit_rating, 0.10)
        
        # Liquidity volatility
        liquidity_vol = (10 - bond.liquidity_score) * 0.005
        
        # Convexity reduces volatility for large rate moves
        convexity_vol_reduction = bond.convexity * 0.0001
        
        volatility = duration_vol + credit_vol + liquidity_vol - convexity_vol_reduction
        
        # SCIENTIFIC BETA MODEL FOR BONDS
        if bond.rate_type == 'floating':
            treasury_2y_beta = 0.05
            treasury_10y_beta = 0.02
            sofr_beta = 0.1
        else:
            if bond.duration <= 3:
                treasury_2y_beta = 0.8
                treasury_10y_beta = 0.3
            elif bond.duration <= 10:
                treasury_2y_beta = 0.4
                treasury_10y_beta = 0.7
            else:
                treasury_2y_beta = 0.2
                treasury_10y_beta = 0.9
            sofr_beta = 0.1
        
        # Credit quality affects correlation
        if bond.credit_rating in ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-']:
            credit_correlation_factor = 1.0  # Investment grade
        else:
            credit_correlation_factor = 1.3  # High yield more correlated
        
        # Apply multi-factor model
        systematic_component = (treasury_2y_beta * treasury_2y_changes + 
                              treasury_10y_beta * treasury_10y_changes +
                              sofr_beta * sofr_changes) * credit_correlation_factor
        
        daily_vol = volatility / np.sqrt(252)
        
        # Create deterministic idiosyncratic component
        asset_hash = abs(hash(bond_id)) % 1000
        time_component = np.sin(np.arange(len(dates)) * 2 * np.pi / 252 + asset_hash)
        sector_component = np.cos(np.arange(len(dates)) * 2 * np.pi / (252 * 3) + asset_hash * 2)
        
        # Scale by idiosyncratic volatility
        total_systematic_vol = np.std(systematic_component) if len(systematic_component) > 1 else 0
        idiosyncratic_vol = np.sqrt(max(0, daily_vol**2 - total_systematic_vol**2))
        
        idiosyncratic_component = (time_component + sector_component) * idiosyncratic_vol * 0.5
        
        # Combine all components
        daily_returns = daily_return + systematic_component + idiosyncratic_component
        
        # Add mean reversion
        mean_reversion = 0.001
        daily_returns = pd.Series(daily_returns, index=dates)
        for j in range(1, len(daily_returns)):
            daily_returns.iloc[j] = (1 - mean_reversion) * daily_returns.iloc[j] + mean_reversion * daily_return
        
        bond_returns_matrix[bond_id] = daily_returns
    
    print(f"✅ Treasury factor modeling complete for {len(bond_returns_matrix.columns)} bonds")
    return bond_returns_matrix

def enhance_floating_bonds_with_nelson_siegel(bonds: list, rates_df: pd.DataFrame) -> Dict:
    """
    Enhance floating rate bonds using Nelson-Siegel forward curve projections
    Fixes the flat rate issue by using market-embedded rate expectations
    """
    from yield_curve_modeling import enhance_floating_bond_cashflows_with_nelson_siegel
    
    enhanced_floating_bonds = {}
    floating_bonds = [b for b in bonds if b.rate_type == 'floating']
    
    print(f"🏦 Enhancing {len(floating_bonds)} floating rate bonds with Nelson-Siegel...")
    
    for bond in floating_bonds:
        try:
            enhanced_cf = enhance_floating_bond_cashflows_with_nelson_siegel(bond, rates_df)
            if not enhanced_cf.empty:
                enhanced_floating_bonds[bond.bond_id] = enhanced_cf
                
                # Show rate evolution for first few bonds
                if len(enhanced_floating_bonds) <= 3:
                    rate_range = f"{enhanced_cf['projected_base_rate'].min():.3%} to {enhanced_cf['projected_base_rate'].max():.3%}"
                    print(f"  ✅ {bond.bond_id} ({bond.issuer}): {rate_range}")
                    
        except Exception as e:
            print(f"  ❌ Error enhancing {bond.bond_id}: {e}")
    
    print(f"✅ Enhanced {len(enhanced_floating_bonds)} floating bonds with Nelson-Siegel forward curves")
    return enhanced_floating_bonds

def export_cashflows_during_run(loans: list, bonds: list, rates_df: pd.DataFrame, adjusted_returns: dict):
    """Export individual cashflows to txt files during portfolio run"""
    import os
    
    # Create timestamped directory
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"cashflows_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Exporting Cashflows to {output_dir}/ ---")
    
    # Export loan cashflows
    loan_count = 0
    for loan in loans:  # Export ALL loans
        try:
            # Calculate current rate
            if loan.rate_type == 'floating':
                base_rate = get_current_rate(loan.base_rate, rates_df)
                current_rate = base_rate + loan.current_spread
            else:
                current_rate = loan.origination_rate
            
            # Create loan schedule
            loan_spec = LoanSpec(
                principal=loan.principal,
                annual_interest_rate=current_rate,
                origination_date=loan.origination_date,
                term_months=loan.maturity_months,
                interest_only_months=loan.io_months,
                amortization_style=loan.amort_style,
                upfront_fee_percent=0.01,
                exit_fee_percent=loan.prepay_penalty / 2,
                benchmark_rate=get_current_rate('DGS2', rates_df, loan.origination_date) if rates_df is not None else 0.03
            )
            
            schedule = loan_schedule(loan_spec)
            irr, moic = loan_IRR_and_MOIC(schedule)
            
            # Enhanced cashflow breakdown with detailed columns
            enhanced_schedule = schedule.copy()
            
            # Add period-specific floating rate calculations
            if loan.rate_type == 'floating':
                period_rates = []
                period_base_rates = []
                period_spreads = []
                
                for period_date in enhanced_schedule.index:
                    if period_date > pd.Timestamp(loan.origination_date):
                        period_rate = calculate_floating_rate_for_period(loan, period_date, rates_df)
                        
                        # Use same historical/forward logic for base rate display
                        try:
                            from forward_rate_lookup import get_global_forward_rate_lookup
                            forward_lookup = get_global_forward_rate_lookup()
                            base_rate = forward_lookup.get_forward_rate(loan.base_rate, period_date.strftime('%Y-%m-%d'), rates_df)
                        except:
                            # Fallback to old method
                            base_rate = get_current_rate(loan.base_rate, rates_df, period_date.strftime('%Y-%m-%d'))
                        
                        period_rates.append(period_rate)
                        period_base_rates.append(base_rate)
                        period_spreads.append(loan.current_spread)
                    else:
                        period_rates.append(current_rate)
                        period_base_rates.append(0)
                        period_spreads.append(0)
                
                enhanced_schedule['PeriodBaseRate'] = period_base_rates
                enhanced_schedule['PeriodSpread'] = period_spreads  
                enhanced_schedule['PeriodAllInRate'] = period_rates
            else:
                enhanced_schedule['PeriodBaseRate'] = current_rate
                enhanced_schedule['PeriodSpread'] = 0.0
                enhanced_schedule['PeriodAllInRate'] = current_rate
            
            # Detailed prepayment modeling
            prepay_speed = calculate_prepayment_speed(loan)
            enhanced_schedule['PrepaySpeed'] = prepay_speed
            enhanced_schedule['PrepayAmount'] = enhanced_schedule['OutstandingStart'] * prepay_speed / 12  # Monthly
            enhanced_schedule['PrepayPenaltyRate'] = loan.prepay_penalty
            enhanced_schedule['PrepayPenaltyAmount'] = enhanced_schedule['PrepayAmount'] * loan.prepay_penalty
            enhanced_schedule['NetPrepayment'] = enhanced_schedule['PrepayAmount'] - enhanced_schedule['PrepayPenaltyAmount']
            
            # Detailed credit default modeling
            enhanced_schedule['DefaultProb'] = loan.default_prob / 12  # Monthly default probability
            # Calculate survival probability properly
            default_prob_monthly = loan.default_prob / 12
            survival_probs = []
            cumulative_survival = 1.0
            for i in range(len(enhanced_schedule)):
                cumulative_survival *= (1 - default_prob_monthly)
                survival_probs.append(cumulative_survival)
            enhanced_schedule['SurvivalProb'] = survival_probs
            enhanced_schedule['ExpectedLossRate'] = loan.default_prob * 0.6  # 60% loss given default
            enhanced_schedule['ExpectedLossAmount'] = enhanced_schedule['OutstandingStart'] * enhanced_schedule['ExpectedLossRate'] / 12
            enhanced_schedule['CreditAdjustment'] = -enhanced_schedule['ExpectedLossAmount']
            
            # Recovery modeling
            enhanced_schedule['RecoveryRate'] = 0.4  # 40% recovery assumption
            enhanced_schedule['RecoveryAmount'] = enhanced_schedule['ExpectedLossAmount'] * (-enhanced_schedule['RecoveryRate'] / -0.6)  # Partial recovery
            
            # Final cashflow calculation
            enhanced_schedule['GrossCashflow'] = enhanced_schedule['Total']
            enhanced_schedule['PrepaymentAdjustment'] = enhanced_schedule['NetPrepayment']
            enhanced_schedule['CreditAdjustment'] = enhanced_schedule['CreditAdjustment']
            enhanced_schedule['RecoveryAdjustment'] = enhanced_schedule['RecoveryAmount']
            enhanced_schedule['NetCashflow'] = (enhanced_schedule['GrossCashflow'] + 
                                              enhanced_schedule['PrepaymentAdjustment'] +
                                              enhanced_schedule['CreditAdjustment'] + 
                                              enhanced_schedule['RecoveryAdjustment'])
            
            schedule = enhanced_schedule
            
            # Export to file
            filename = f"{loan.loan_id}_{loan.borrower.replace(' ', '_').replace('&', 'and')}_cashflows.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"LOAN CASHFLOW ANALYSIS: {loan.loan_id}\n")
                f.write(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")
                f.write(f"Borrower: {loan.borrower}\n")
                f.write(f"Sector: {loan.sector} | Geography: {loan.geography}\n")
                f.write(f"Principal: ${loan.principal:,.0f}\n")
                f.write(f"Rate Type: {loan.rate_type}\n")
                if loan.rate_type == 'floating':
                    f.write(f"Base Rate: {loan.base_rate} ({base_rate:.2%})\n")
                    f.write(f"Spread: {loan.current_spread:.2%}\n")
                    f.write(f"All-in Rate: {current_rate:.2%}\n")
                else:
                    f.write(f"Fixed Rate: {current_rate:.2%}\n")
                f.write(f"Credit Rating: {loan.credit_rating}\n")
                f.write(f"Default Probability: {loan.default_prob:.2%}\n")
                f.write(f"LTV: {loan.ltv:.1%} | DSCR: {loan.dscr:.2f}x\n")
                f.write(f"Prepayment Penalty: {loan.prepay_penalty:.2%}\n")
                f.write(f"Seniority: {loan.seniority}\n")
                f.write(f"Collateral: {loan.collateral_type}\n")
                f.write(f"\nPORTFOLIO METRICS:\n")
                f.write(f"IRR (base): {irr:.2%}\n")
                f.write(f"IRR (adjusted): {adjusted_returns.get(loan.loan_id, irr):.2%}\n")
                f.write(f"MOIC: {moic:.3f}x\n")
                f.write("\n" + "="*80 + "\n")
                f.write("DETAILED CASHFLOW SCHEDULE\n")
                f.write("="*120 + "\n")
                
                # Write comprehensive header
                f.write(f"{'Date':>12} | {'Outstanding':>12} | {'BaseRate':>9} | {'Spread':>8} | {'AllIn':>8} | ")
                f.write(f"{'Interest':>10} | {'Principal':>10} | {'PrepayAmt':>10} | {'PrepayPen':>10} | ")
                f.write(f"{'DefaultLoss':>10} | {'Recovery':>10} | {'NetCF':>10}\n")
                f.write("-" * 120 + "\n")
                
                # Write detailed cashflow data
                for idx, row in schedule.iterrows():
                    f.write(f"{idx.strftime('%Y-%m-%d'):>12} | ")
                    f.write(f"${row['OutstandingStart']:>11,.0f} | ")
                    f.write(f"{row.get('PeriodBaseRate', current_rate):>8.3f} | ")
                    f.write(f"{row.get('PeriodSpread', 0):>7.3f} | ")
                    f.write(f"{row.get('PeriodAllInRate', current_rate):>7.3f} | ")
                    f.write(f"${row['Interest']:>9,.0f} | ")
                    f.write(f"${row['Principal']:>9,.0f} | ")
                    f.write(f"${row.get('PrepayAmount', 0):>9,.0f} | ")
                    f.write(f"${row.get('PrepayPenaltyAmount', 0):>9,.0f} | ")
                    f.write(f"${abs(row.get('CreditAdjustment', 0)):>9,.0f} | ")
                    f.write(f"${row.get('RecoveryAmount', 0):>9,.0f} | ")
                    f.write(f"${row['NetCashflow']:>9,.0f}\n")
                
                f.write("\n" + "="*120 + "\n")
                f.write("COLUMN DEFINITIONS:\n")
                f.write("="*120 + "\n")
                f.write("Outstanding: Principal balance at period start\n")
                f.write("BaseRate: Historical base rate (SOFR/Treasury) for floating loans\n")
                f.write("Spread: Credit spread over base rate\n") 
                f.write("AllIn: Total interest rate (BaseRate + Spread)\n")
                f.write("Interest: Interest payment for period\n")
                f.write("Principal: Scheduled principal payment\n")
                f.write("PrepayAmt: Expected prepayment amount based on borrower characteristics\n")
                f.write("PrepayPen: Penalty fee on prepayments\n")
                f.write("DefaultLoss: Expected credit loss for the period\n")
                f.write("Recovery: Expected recovery value in case of default\n")
                f.write("NetCF: Final net cashflow after all adjustments\n")
            
            loan_count += 1
            
        except Exception as e:
            print(f"Error exporting {loan.loan_id}: {e}")
    
    # Export bond cashflows
    bond_count = 0
    for bond in bonds:  # Export ALL bonds
        try:
            current_ytm = adjusted_returns.get(bond.bond_id, bond.vintage_yield)
            
            # Generate coupon dates (simplified)
            maturity = pd.Timestamp(bond.maturity_date)
            settle = pd.Timestamp.today()
            
            if maturity <= settle:
                continue  # Skip matured bonds
                
            # Generate payment dates
            months_between = 12 // bond.freq
            payment_dates = []
            current_date = maturity
            
            while current_date > settle:
                payment_dates.append(current_date)
                current_date = current_date - pd.DateOffset(months=months_between)
                
            payment_dates = sorted([d for d in payment_dates if d > settle])
            
            if not payment_dates:
                continue
            
            # Create cashflows
            coupon_amount = bond.face_value * bond.coupon_rate / bond.freq
            
            # Credit adjustments
            credit_default_map = {
                'AAA': 0.0001, 'AA+': 0.0002, 'AA': 0.0003, 'AA-': 0.0005,
                'A+': 0.0008, 'A': 0.0012, 'A-': 0.0018,
                'BBB+': 0.0025, 'BBB': 0.0035, 'BBB-': 0.0050,
                'BB+': 0.0080, 'BB': 0.0120, 'BB-': 0.0180,
                'B+': 0.0250, 'B': 0.0350, 'B-': 0.0500,
                'CCC+': 0.0800, 'CCC': 0.1200, 'CCC-': 0.1800
            }
            
            default_prob = credit_default_map.get(bond.credit_rating, 0.02)
            recovery_rate = 0.4
            
            # Export to file
            filename = f"{bond.bond_id}_{bond.issuer.replace(' ', '_').replace('.', '').replace('&', 'and')}_cashflows.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"BOND CASHFLOW ANALYSIS: {bond.bond_id}\n")
                f.write(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")
                f.write(f"Issuer: {bond.issuer}\n")
                f.write(f"Sector: {bond.sector} | Geography: {bond.geography}\n")
                f.write(f"Face Value: ${bond.face_value:,.0f}\n")
                f.write(f"Coupon Rate: {bond.coupon_rate:.2%}\n")
                f.write(f"Rate Type: {bond.rate_type}\n")
                if bond.rate_type == 'floating':
                    f.write(f"Base Rate: {bond.base_rate}\n")
                    f.write(f"Spread: {bond.spread:.2%}\n")
                f.write(f"Payment Frequency: {bond.freq}x per year\n")
                f.write(f"Credit Rating: {bond.credit_rating}\n")
                f.write(f"Duration: {bond.duration:.2f} | Convexity: {bond.convexity:.2f}\n")
                f.write(f"Liquidity Score: {bond.liquidity_score:.1f}/10\n")
                f.write(f"Callable: {bond.callable} | Seniority: {bond.seniority}\n")
                f.write(f"\nPORTFOLIO METRICS:\n")
                f.write(f"Current YTM: {current_ytm:.2%}\n")
                f.write(f"Default Probability: {default_prob:.3%}\n")
                f.write(f"Expected Recovery: {recovery_rate:.1%}\n")
                f.write("\n" + "="*80 + "\n")
                f.write("DETAILED BOND CASHFLOW SCHEDULE\n")
                f.write("="*120 + "\n")
                
                # Write comprehensive header for bonds
                f.write(f"{'Date':>12} | {'CouponRate':>10} | {'BaseRate':>9} | {'Spread':>8} | ")
                f.write(f"{'Coupon':>10} | {'Principal':>10} | {'DefaultLoss':>10} | {'Recovery':>10} | {'NetCF':>10}\n")
                f.write("-" * 120 + "\n")
                
                # Write detailed bond cashflow data
                for i, date in enumerate(payment_dates):
                    # Calculate period-specific rates for floating rate bonds
                    if bond.rate_type == 'floating':
                        # Use forward rate lookup if available, otherwise current rate
                        try:
                            from forward_rate_lookup import get_global_forward_rate_lookup
                            forward_lookup = get_global_forward_rate_lookup()
                            
                            if forward_lookup.forward_rates_dir:
                                # Use historical for past dates, forward for future dates
                                period_base_rate = forward_lookup.get_forward_rate(
                                    bond.base_rate, 
                                    date.strftime('%Y-%m-%d'),
                                    rates_df  # Pass FRED data for historical lookups
                                )
                            else:
                                # Fallback to current rate
                                period_base_rate = get_current_rate(bond.base_rate, rates_df, date.strftime('%Y-%m-%d'))
                        except:
                            # Fallback if forward rate system not available
                            period_base_rate = get_current_rate(bond.base_rate, rates_df, date.strftime('%Y-%m-%d'))
                        
                        period_coupon_rate = period_base_rate + bond.spread
                        period_coupon = bond.face_value * period_coupon_rate / bond.freq
                    else:
                        period_base_rate = get_historical_rate('DGS10', bond.issue_date, rates_df)
                        period_coupon_rate = bond.coupon_rate
                        period_coupon = coupon_amount
                    
                    if i == len(payment_dates) - 1:  # Final payment
                        coupon = period_coupon
                        principal = bond.face_value
                        total = coupon + principal
                    else:  # Coupon payments only
                        coupon = period_coupon
                        principal = 0
                        total = coupon
                    
                    # Enhanced credit modeling
                    periods_from_now = i + 1
                    period_default_prob = default_prob * periods_from_now / len(payment_dates)
                    survival_prob = (1 - default_prob / len(payment_dates)) ** periods_from_now
                    
                    expected_loss = total * period_default_prob
                    recovery_value = expected_loss * recovery_rate
                    net_credit_adjustment = recovery_value - expected_loss
                    
                    adjusted_total = total * survival_prob + recovery_value
                    
                    f.write(f"{date.strftime('%Y-%m-%d'):>12} | ")
                    f.write(f"{period_coupon_rate:>9.3%} | ")
                    f.write(f"{period_base_rate:>8.3%} | ")
                    f.write(f"{bond.spread if bond.rate_type=='floating' else 0:>7.3%} | ")
                    f.write(f"${coupon:>9,.0f} | ")
                    f.write(f"${principal:>9,.0f} | ")
                    f.write(f"${expected_loss:>9,.0f} | ")
                    f.write(f"${recovery_value:>9,.0f} | ")
                    f.write(f"${adjusted_total:>9,.0f}\n")
                
                f.write("\n" + "="*120 + "\n")
                f.write("COLUMN DEFINITIONS:\n")
                f.write("="*120 + "\n")
                f.write("CouponRate: Period-specific coupon rate (fixed or floating)\n")
                f.write("BaseRate: Historical base rate from FRED data\n")
                f.write("Spread: Credit spread over base rate (floating bonds only)\n")
                f.write("Coupon: Coupon payment for the period\n")
                f.write("Principal: Principal repayment (final period only)\n")
                f.write("DefaultLoss: Expected credit loss for the period\n")
                f.write("Recovery: Expected recovery value in case of default\n")
                f.write("NetCF: Final net cashflow after credit adjustments\n")
            
            bond_count += 1
            
        except Exception as e:
            print(f"Error exporting {bond.bond_id}: {e}")
    
    # Create summary file
    summary_path = os.path.join(output_dir, "EXPORT_SUMMARY.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CASHFLOW EXPORT SUMMARY\n")
        f.write(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
        f.write(f"Exported Loan Cashflows: {loan_count}\n")
        f.write(f"Exported Bond Cashflows: {bond_count}\n")
        f.write(f"Total Files: {loan_count + bond_count + 1}\n")
        f.write(f"\nDirectory: {os.path.abspath(output_dir)}\n")
        f.write("\nAll cashflows include:\n")
        f.write("- Real-time floating rate calculations\n")
        f.write("- Credit default probability adjustments\n")
        f.write("- Prepayment penalty impacts (loans)\n")
        f.write("- Expected loss provisions\n")
        f.write("- Recovery value estimates\n")
    
    print(f"Exported {loan_count} loan and {bond_count} bond cashflow files to {output_dir}/")
    return output_dir

def create_enhanced_bond_portfolio(use_ou_modeling: bool = True, 
                                  generate_forward_rates: bool = True,
                                  data_source: str = 'synthetic'):
    """
    Create bond portfolio using enriched specifications and real market data
    
    Args:
        use_ou_modeling: If True, use OU process for all bonds. If False, use treasury factor model.
        generate_forward_rates: If True, generate Nelson-Siegel forward rate projections.
        data_source: 'synthetic' for synthetic_data files, 'deals_data' for deals_data files
    """
    
    print("=== Enhanced Bond Portfolio with Real Market Data ===")
    print(f"Bond modeling approach: {'OU Mean-Reverting Process' if use_ou_modeling else 'Treasury Factor Model'}")
    print(f"Forward rate projections: {'Nelson-Siegel' if generate_forward_rates else 'Current rates only'}")
    
    # Load data
    rates_df = load_fred_rates()
    loans = load_loans(data_source)
    bonds = load_bonds(data_source)
    
    # Get actual risk-free rate from FRED data
    risk_free_rate = get_current_risk_free_rate(rates_df)
    
    if not loans:
        print("Failed to load loan data")
        return
    
    if not bonds and data_source != 'deals_data':
        print("Failed to load bond data")
        return
    
    print(f"\\nLoaded {len(loans)} loans and {len(bonds)} bonds")
    print(f"Rate data available: {rates_df is not None}")
    
    # Generate forward rate projections using Nelson-Siegel
    forward_rates_dir = None
    if generate_forward_rates:
        print("\\n🏦 GENERATING NELSON-SIEGEL FORWARD RATE PROJECTIONS")
        print("="*60)
        try:
            from forward_rate_projections import export_all_forward_rate_projections
            
            forward_rates_dir = export_all_forward_rate_projections(
                rates_df=rates_df,
                projection_horizon_years=10.0
            )
            
            print(f"✅ Forward rate projections exported to: {forward_rates_dir}")
            
        except Exception as e:
            print(f"❌ Error generating forward rates: {e}")
            print("Continuing with current rate approach...")
            forward_rates_dir = None
    
    # Find the best available cashflow directory dynamically
    cashflow_dirs = [d for d in os.listdir('.') if d.startswith('cashflows_')]
    cashflow_dir = None
    
    if cashflow_dirs:
        # Look for a directory with meaningful IRR data
        for dir_candidate in sorted(cashflow_dirs, reverse=True):
            test_file_pattern = os.path.join(dir_candidate, '*_cashflows.txt')
            test_files = glob.glob(test_file_pattern)
            
            if test_files:
                # Test if this directory has good IRR data
                try:
                    with open(test_files[0], 'r') as f:
                        content = f.read()
                        base_irr_match = re.search(r'IRR \(base\): ([\d.]+)%', content)
                        if base_irr_match:
                            irr_val = float(base_irr_match.group(1))
                            if irr_val > 4.0:  # Look for meaningful IRRs (>4%)
                                cashflow_dir = dir_candidate
                                print(f"\\nFound valid cashflow directory: {cashflow_dir}")
                                break
                except:
                    continue
        
        # Fallback to latest if no good data found
        if not cashflow_dir:
            cashflow_dir = max(cashflow_dirs)
            print(f"\\nUsing latest cashflow directory: {cashflow_dir}")
    else:
        print("\\nNo cashflow directory found, using vintage yield estimates")
    
    # Calculate sophisticated returns and risk metrics
    print("\\nCalculating returns from IRR/MOIC analysis...")
    returns_dict = calculate_returns_from_cashflows(loans, bonds, cashflow_dir)
    
    print("\\nCalculating sophisticated volatility estimates...")
    volatilities = calculate_private_loan_volatilities(returns_dict, data_source)
    
    print("\\nGenerating correlation matrix...")
    correlation_matrix = create_correlation_matrix(returns_dict, data_source)
    
    # Extract returns for portfolio optimization  
    returns = {}
    for asset_id, metrics in returns_dict.items():
        returns[asset_id] = metrics['annual_return']
    
    print("Calculating bond returns with current market rates...")
    for bond in bonds:
        try:
            current_ytm = calculate_bond_current_ytm(bond, rates_df)
            returns[bond.bond_id] = current_ytm
            
        except Exception as e:
            print(f"Error calculating return for {bond.bond_id}: {e}")
            # Use FRED data for structured fallback based on bond characteristics
            if bond.rate_type == 'floating':
                # For floating rate bonds, use current FRED rate + spread
                current_base_rate = get_current_rate(bond.base_rate, rates_df)
                returns[bond.bond_id] = current_base_rate + bond.spread
            else:
                # For fixed rate bonds, use historical vs current FRED rates
                base_treasury = get_current_rate('DGS10', rates_df)  # Current 10Y from FRED
                vintage_treasury = get_historical_rate('DGS10', bond.issue_date, rates_df)  # Historical from FRED
                duration_adj = -(base_treasury - vintage_treasury) * bond.duration / 100
                credit_adj = bond.oas_spread * (10 - bond.liquidity_score) / 10
                returns[bond.bond_id] = bond.vintage_yield + duration_adj + credit_adj
    
    # Apply credit and prepayment adjustments
    print("Applying prepayment and credit adjustments...")
    adjusted_returns = apply_prepayment_and_default_adjustments(returns, loans, bonds)
    
    # Enhance floating rate bonds with Nelson-Siegel forward curves
    print("Enhancing floating rate bonds with Nelson-Siegel...")
    enhanced_floating_bonds = enhance_floating_bonds_with_nelson_siegel(bonds, rates_df)
    
    # Export individual cashflows with timestamp
    print("Exporting individual cashflows...")
    export_dir = export_cashflows_during_run(loans, bonds, rates_df, adjusted_returns)
    
    # Create daily returns matrix for portfolio optimization
    print("Creating daily returns matrix...")
    start_date = '2022-01-01'
    end_date = dt.date.today().strftime('%Y-%m-%d')
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate loan returns (always use scientific factor model for loans)
    print("Generating loan returns using scientific factor model...")
    loan_returns_matrix = pd.DataFrame(index=dates)
    
    # Use Treasury factors for loan returns (not OU - loans don't have yield curves)
    treasury_10y_changes = rates_df['DGS10'].pct_change().reindex(dates, method='ffill').fillna(0)
    sofr_changes = rates_df['SOFR'].pct_change().reindex(dates, method='ffill').fillna(0)
    
    for loan in loans:
        loan_id = loan.loan_id
        if loan_id not in adjusted_returns:
            continue
            
        annual_return = adjusted_returns[loan_id]
        daily_return = annual_return / 252
        
        # SCIENTIFIC LOAN VOLATILITY MODEL
        credit_vol_map = {
            'A': 0.08, 'A-': 0.09, 'BBB+': 0.10, 'BBB': 0.12, 'BBB-': 0.14,
            'BB+': 0.16, 'BB': 0.18, 'BB-': 0.20, 'B+': 0.22, 'B': 0.25, 'B-': 0.28
        }
        base_vol = credit_vol_map.get(loan.credit_rating, 0.20)
        ltv_vol = loan.ltv * 0.15
        dscr_vol = max(0, (1.5 - loan.dscr) * 0.08)
        
        sector_vol_map = {
            'Technology': 0.12, 'Healthcare': 0.08, 'Financial Services': 0.15,
            'Energy': 0.25, 'Real Estate': 0.18, 'Consumer Discretionary': 0.14,
            'Consumer Staples': 0.07, 'Industrials': 0.11, 'Materials': 0.16,
            'Transportation': 0.20, 'Utilities': 0.09, 'Telecommunications': 0.10
        }
        sector_vol = sector_vol_map.get(loan.sector, 0.12)
        volatility = 0.4 * base_vol + 0.3 * sector_vol + 0.2 * ltv_vol + 0.1 * dscr_vol
        
        # Loan beta model
        if loan.rate_type == 'floating':
            sofr_beta = 0.1 + loan.default_prob * 0.3
            treasury_beta = 0.05
        else:
            sofr_beta = 0.3 + loan.default_prob * 0.5
            treasury_beta = 0.2 + loan.default_prob * 0.4
        
        systematic_component = sofr_beta * sofr_changes + treasury_beta * treasury_10y_changes
        
        daily_vol = volatility / np.sqrt(252)
        asset_hash = abs(hash(loan_id)) % 1000
        time_component = np.sin(np.arange(len(dates)) * 2 * np.pi / 252 + asset_hash)
        sector_component = np.cos(np.arange(len(dates)) * 2 * np.pi / (252 * 3) + asset_hash * 2)
        
        total_systematic_vol = np.std(systematic_component) if len(systematic_component) > 1 else 0
        idiosyncratic_vol = np.sqrt(max(0, daily_vol**2 - total_systematic_vol**2))
        idiosyncratic_component = (time_component + sector_component) * idiosyncratic_vol * 0.5
        
        daily_returns = daily_return + systematic_component + idiosyncratic_component
        
        # Add mean reversion for loans
        mean_reversion = 0.002
        daily_returns = pd.Series(daily_returns, index=dates)
        for j in range(1, len(daily_returns)):
            daily_returns.iloc[j] = (1 - mean_reversion) * daily_returns.iloc[j] + mean_reversion * daily_return
        
        loan_returns_matrix[loan_id] = daily_returns
    
    # Generate bond returns using selected approach
    if use_ou_modeling:
        print("Generating bond returns using OU mean-reverting process...")
        bond_returns_matrix = generate_ou_based_daily_returns_for_all_bonds(bonds, rates_df, dates)
    else:
        print("Generating bond returns using treasury factor model...")  
        bond_returns_matrix = generate_treasury_factor_daily_returns_for_bonds(bonds, rates_df, dates, adjusted_returns)
    
    # Combine loan and bond returns
    returns_matrix = pd.concat([loan_returns_matrix, bond_returns_matrix], axis=1)
    
    # Advanced OU modeling for sample bonds and loans
    print("\\nRunning advanced OU modeling for sample assets...")
    try:
        # Run OU simulation for first 3 bonds (if any exist)
        if bonds:
            sample_bonds = bonds[:3]
            for bond in sample_bonds:
                try:
                    bond_ou_results = calculate_bond_monthly_returns_with_rolldown(bond, rates_df, n_months=12)
                    print(f"OU simulation complete for {bond.bond_id}")
                except Exception as e:
                    print(f"Error in OU simulation for {bond.bond_id}: {e}")
        
        # Run income-based simulation for first 3 loans  
        sample_loans = loans[:3]
        for loan in sample_loans:
            try:
                loan_income_results = calculate_loan_monthly_returns_income_based(loan, rates_df, n_months=12)
                print(f"Income simulation complete for {loan.loan_id}")
            except Exception as e:
                print(f"Error in income simulation for {loan.loan_id}: {e}")
                
    except Exception as e:
        print(f"Error in advanced modeling: {e}")
    
    # For deals_data, use path simulation instead of portfolio optimization
    if data_source == 'deals_data':
        print("\\n" + "="*60)
        print("MONTE CARLO PATH SIMULATION FOR INDIVIDUAL LOANS")
        print("="*60)
        print("Running stochastic simulations for each loan (10,000 scenarios each)...")
        
        # Run proper fixed income simulation
        path_results = simulate_loan_paths(loans, returns_dict, volatilities, rates_df, num_simulations=10000, time_horizon_years=5)
        
        # Create path simulation plots (individual loan histograms)
        plot_filename = create_path_simulation_plots(path_results)
        
        # Generate efficient frontier from Monte Carlo portfolio combinations
        print("\\n" + "="*60)
        print("GENERATING PORTFOLIO EFFICIENT FRONTIER FROM MONTE CARLO PATHS")
        print("="*60)
        
        portfolios, max_sharpe, min_vol = create_portfolio_efficient_frontier_from_paths(path_results, returns_dict, risk_free_rate, num_portfolios=5000)
        
        # Calculate portfolio-level statistics for summary
        all_mean_irrs = [path_results[loan_id]['stats']['mean_irr'] for loan_id in path_results.keys()]
        all_irr_vols = [path_results[loan_id]['stats']['irr_vol'] for loan_id in path_results.keys()]
        all_default_rates = [path_results[loan_id]['stats']['default_rate'] for loan_id in path_results.keys()]
        
        portfolio_mean_irr = np.mean(all_mean_irrs)
        portfolio_irr_vol = np.mean(all_irr_vols)
        portfolio_default_rate = np.mean(all_default_rates)
        
    else:
        # For synthetic data, use traditional portfolio optimization
        print("\\nGenerating returns matrix for portfolio optimization...")
        returns_matrix = create_sophisticated_returns_matrix(returns_dict, volatilities, correlation_matrix, num_periods=252)
        print(f"✅ Created returns matrix: {returns_matrix.shape[0]} periods x {returns_matrix.shape[1]} assets")
        
        # Portfolio optimization
        print("\\nOptimizing portfolio...")
        portfolios = simulate_random_portfolios_from_returns(
            returns_matrix, num_portfolios=10000, risk_free_rate=risk_free_rate, allow_short=False, random_seed=123
        )
        
        max_sharpe, min_vol = best_portfolios(portfolios)
    
    # Results
    print("\\n" + "="*60)
    if data_source == 'deals_data':
        print("PRIVATE LOAN PATH SIMULATION RESULTS")
    else:
        print("ENHANCED BOND PORTFOLIO RESULTS")
    print("="*60)
    
    if data_source == 'deals_data':
        print("\\n=== PORTFOLIO-LEVEL PATH SIMULATION STATISTICS ===")
        print(f"Average Loan Mean IRR: {portfolio_mean_irr:.2%}")
        print(f"Average Loan IRR Volatility: {portfolio_irr_vol:.2%}")
        print(f"Average Default Rate: {portfolio_default_rate:.1%}")
        
        print("\\n=== BEST PERFORMING LOAN ===")
        print(f"Expected IRR: {max_sharpe['Return']:.2%}")
        print(f"IRR Volatility: {max_sharpe['Volatility']:.2%}")
        print(f"Risk-Adjusted Score: {max_sharpe['Sharpe']:.2f}")
        
        print("\\n=== WORST PERFORMING LOAN ===") 
        print(f"Expected IRR: {min_vol['Return']:.2%}")
        print(f"IRR Volatility: {min_vol['Volatility']:.2%}")
        print(f"Risk-Adjusted Score: {min_vol['Sharpe']:.2f}")
        
    else:
        print("\\n=== Max-Sharpe Portfolio (Enhanced) ===")
        print(f"Expected Return: {max_sharpe['Return']:.2%}")
        print(f"Volatility: {max_sharpe['Volatility']:.2%}")
        print(f"Sharpe Ratio: {max_sharpe['Sharpe']:.2f}")
        
        print("\\n=== Min-Vol Portfolio (Enhanced) ===")
        print(f"Expected Return: {min_vol['Return']:.2%}")
        print(f"Volatility: {min_vol['Volatility']:.2%}")
        print(f"Sharpe Ratio: {min_vol['Sharpe']:.2f}")
    
    # Show top holdings in each portfolio
    print("\\n=== Top 10 Holdings - Max Sharpe ===")
    weights = {k: v for k, v in max_sharpe.items() if k not in ['Return', 'Volatility', 'Sharpe']}
    top_holdings = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
    for asset, weight in top_holdings:
        asset_type = "Loan" if asset.startswith('LOAN_') else "Bond"
        print(f"{asset:12s}: {weight:6.2%} ({asset_type})")
    
    # Plot results based on analysis type
    if data_source == 'deals_data':
        print("\\n✅ Individual loan path simulation plots generated")
        print(f"✅ Now generating portfolio efficient frontier plot...")
        
        # Plot efficient frontier using Monte Carlo portfolio data
        title = f"Enhanced Loan Portfolio - Efficient Frontier\n({len(loans)} Private Loans from Monte Carlo Analysis)"
        plot_efficient_frontier(
            portfolios, max_sharpe, min_vol, risk_free_rate=risk_free_rate,
            title=title
        )
    else:
        # Traditional efficient frontier plot for synthetic data
        title = f"Enhanced Bond Portfolio - Efficient Frontier\n({len(loans)} Loans + {len(bonds)} Corporate Bonds with Real Market Data)"
        plot_efficient_frontier(
            portfolios, max_sharpe, min_vol, risk_free_rate=risk_free_rate,
            title=title
        )
    
    # Sample analytics
    print("\\n=== Sample Asset Analytics ===")
    sample_loan = loans[0]
    if bonds:  # Only access bonds if they exist
        sample_bond = bonds[0]
    
    print(f"\\nSample Loan: {sample_loan.loan_id}")
    print(f"  Borrower: {sample_loan.borrower}")
    print(f"  Sector: {sample_loan.sector}")
    print(f"  Rate Type: {sample_loan.rate_type}")
    if sample_loan.rate_type == 'floating':
        print(f"  Base Rate: {sample_loan.base_rate}")
        print(f"  Current Spread: {sample_loan.current_spread:.2%}")
    print(f"  Current Return: {adjusted_returns[sample_loan.loan_id]:.2%}")
    print(f"  Credit Rating: {sample_loan.credit_rating}")
    print(f"  Default Probability: {sample_loan.default_prob:.1%}")
    
    if bonds:  # Only show bond analytics if bonds exist
        print(f"\\nSample Bond: {sample_bond.bond_id}")
        print(f"  Issuer: {sample_bond.issuer}")
        print(f"  Sector: {sample_bond.sector}")
        print(f"  Rate Type: {sample_bond.rate_type}")
        print(f"  Current YTM: {adjusted_returns[sample_bond.bond_id]:.2%}")
        print(f"  Credit Rating: {sample_bond.credit_rating}")
        print(f"  Duration: {sample_bond.duration:.1f}")
        print(f"  Liquidity Score: {sample_bond.liquidity_score:.1f}/10")
    else:
        print("\\nNo bonds in portfolio (loans-only analysis)")
    
    return portfolios, max_sharpe, min_vol, loans, bonds, adjusted_returns

if __name__ == "__main__":
    create_enhanced_bond_portfolio()