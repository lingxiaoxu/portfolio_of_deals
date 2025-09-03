#!/usr/bin/env python3
"""
Fixed Income Portfolio Analysis System - Main Entry Point

This script orchestrates comprehensive fixed income portfolio analysis:
- Private Credit Portfolio: 50+ private loans with advanced modeling
- Corporate Bond Portfolio: 100+ corporate bonds with OU mean-reverting processes
- Scientific Risk Modeling: FRED data integration, Nelson-Siegel yield curves
- Enhanced Analytics: MTM returns for bonds, income-based returns for loans

Usage: python run.py

Dependencies:
- enriched_bond_portfolio.py: Fixed income analysis with advanced OU modeling  
- bond_utilities.py: Mathematical functions for bond and loan calculations
- External data: synthetic_data/synthetic_loans.csv, synthetic_data/synthetic_bonds.csv, fred_rates.csv
"""

import datetime as dt
import logging
import sys
import os

def setup_logging():
    """Setup comprehensive logging with timestamp"""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"synthetic_analysis_{timestamp}.log"
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    log_filepath = os.path.join("logs", log_filename)
    
    # Configure logging (INFO level reduces matplotlib font debugging)
    logging.basicConfig(
        level=logging.INFO,  # Changed from DEBUG to reduce matplotlib noise
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Portfolio analysis logging started - {timestamp}")
    logger.info(f"Log file: {log_filepath}")
    
    return logger, log_filepath

def main():
    """Main function to run comprehensive portfolio analysis"""
    
    # Setup logging
    logger, log_filepath = setup_logging()
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE PORTFOLIO ANALYSIS SYSTEM")
    logger.info("="*80)
    logger.info("✓ Fixed Income Portfolio: FRED rate data integration")
    logger.info("✓ Bond Portfolio: 50 loans + 100 bonds with FRED rate data")
    logger.info("✓ Scientific Risk Modeling: No random fallbacks")
    logger.info("✓ Enhanced Cashflow Export: Detailed breakdowns with timestamps")
    logger.info("="*80)
    
    # Set analysis parameters
    start_date = "2021-01-01"
    end_date = dt.date.today().strftime("%Y-%m-%d")
    rf_annual = 0.035  # 3.5% risk-free rate
    
    try:
        # 1. Run Fixed Income Portfolio Analysis  
        logger.info("\n" + "="*60)
        logger.info("1. FIXED INCOME PORTFOLIO ANALYSIS") 
        logger.info("="*60)
        
        from enriched_bond_portfolio import create_enhanced_bond_portfolio
        logger.info("Imported bond portfolio module successfully")
        
        logger.info("Starting bond portfolio analysis...")
        # Use OU modeling for all bonds (set to False to use treasury factor model)
        use_ou_modeling = True
        
        # Check if deals_data exists, otherwise use synthetic data
        import os
        if os.path.exists('deals_data/deal_start_structured.csv') and os.path.exists('deals_data/portfolio.csv'):
            logger.info("Found deals_data files - using real deal data")
            data_source = 'deals_data'
        else:
            logger.info("Using synthetic data files")
            data_source = 'synthetic'
            
        bond_portfolios, bond_max_sharpe, bond_min_vol, loans, bonds, adjusted_returns = create_enhanced_bond_portfolio(
            use_ou_modeling=use_ou_modeling, 
            data_source=data_source
        )
        logger.info("Bond portfolio analysis completed successfully")
        
        logger.info(f"\n✅ Bond Portfolio Analysis Complete")
        logger.info(f"   - Analyzed {len(loans)} private loans + {len(bonds)} corporate bonds")
        logger.info(f"   - Max-Sharpe: {bond_max_sharpe['Return']:.2%} return, {bond_max_sharpe['Sharpe']:.2f} Sharpe")
        logger.info(f"   - Min-Vol: {bond_min_vol['Return']:.2%} return, {bond_min_vol['Sharpe']:.2f} Sharpe")
        logger.info(f"   - Cashflows exported with detailed breakdowns")
        
        # 2. Fixed Income Portfolio Summary
        print("\n" + "="*60)
        print("2. FIXED INCOME PORTFOLIO SUMMARY")
        print("="*60)
        
        print("\nFIXED INCOME PORTFOLIO:")
        data_source_msg = "deals_data files" if data_source == 'deals_data' else "synthetic_data files"
        print(f"  Assets: {len(loans) + len(bonds)} (loaded from {data_source_msg})")
        print(f"  Max-Sharpe: {bond_max_sharpe['Return']:.2%} return, {bond_max_sharpe['Volatility']:.2%} vol")
        print(f"  Min-Vol: {bond_min_vol['Return']:.2%} return, {bond_min_vol['Volatility']:.2%} vol")
        
        # Risk-return characteristics
        bond_sharpe_ratio = bond_max_sharpe['Sharpe']
        
        print(f"\nRISK-ADJUSTED PERFORMANCE:")
        print(f"  Fixed Income Sharpe Ratio: {bond_sharpe_ratio:.2f}")
        print(f"  Volatility: {bond_max_sharpe['Volatility']:.2%}")
        print(f"  Return: {bond_max_sharpe['Return']:.2%}")
        
        print("\n" + "="*60)
        print("ANALYSIS FEATURES ACTIVE:")
        print("="*60)
        print("✅ Real FRED interest rate data (SOFR, 2Y, 10Y Treasury)")
        print("✅ Historical rate lookups for vintage vs current comparison")
        print("✅ Floating rate resets based on actual reset frequencies")
        print("✅ Scientific volatility modeling (credit, duration, sector)")
        print("✅ Convexity adjustments for bond pricing")
        print("✅ Multi-factor prepayment speed calculations")
        print("✅ Detailed cashflow breakdowns with separate risk columns")
        print("✅ All 150 bond/loan cashflows exported with timestamps")
        print("✅ Dynamic bond/loan loading from external configuration files")
        
        print(f"\n🎯 PORTFOLIO ANALYSIS COMPLETE")
        print(f"   Total assets analyzed: {len(loans) + len(bonds)}")
        print(f"   Analysis period: {start_date} to {end_date}")
        print(f"   Risk-free rate: {rf_annual:.2%}")
        
        # Log final completion
        logger.info("Portfolio analysis completed successfully!")
        
        return {
            'bond_results': {
                'portfolios': bond_portfolios,
                'max_sharpe': bond_max_sharpe,
                'min_vol': bond_min_vol,
                'loans': loans,
                'bonds': bonds,
                'adjusted_returns': adjusted_returns
            },
            'log_file': log_filepath
        }
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR in portfolio analysis: {e}")
        logger.error("Full traceback:")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\n🎯 Analysis complete! Log file saved to: {result['log_file']}")
    else:
        print("\n❌ Analysis failed - check log file for errors")