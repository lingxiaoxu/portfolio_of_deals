#!/usr/bin/env python3
"""
Centralized Prepayment Risk Module for Private Credit Analysis

This module provides consistent prepayment modeling for both run_deals.py and run_synthetic.py
Includes sophisticated multi-factor prepayment speed calculations and cashflow adjustments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PrepaymentFactors:
    """Factors affecting prepayment behavior"""
    ltv: float
    dscr: float  
    credit_rating: str
    rate_type: str  # 'fixed' or 'floating'
    seniority: str  # 'Senior' or 'Subordinated' 
    sector: str
    geography: str
    current_rate_environment: float  # Current interest rates vs origination

class PrepaymentRiskEngine:
    """Centralized engine for prepayment risk modeling"""
    
    def __init__(self):
        # Base prepayment parameters
        self.base_annual_rate = 0.05  # 5% annual base prepayment rate
        
        # Adjustment factors
        self.ltv_adjustments = {
            'high': -0.02,    # >80% LTV reduces prepay ability
            'medium': 0.0,    # 60-80% LTV neutral
            'low': 0.015      # <60% LTV increases prepay ability
        }
        
        self.dscr_adjustments = {
            'strong': 0.01,   # >1.5x DSCR increases prepay ability  
            'medium': 0.0,    # 1.2-1.5x DSCR neutral
            'weak': -0.015    # <1.2x DSCR reduces prepay ability
        }
        
        self.rating_adjustments = {
            'AAA': 0.02, 'AA+': 0.018, 'AA': 0.015, 'AA-': 0.012, 
            'A+': 0.01, 'A': 0.008, 'A-': 0.005, 
            'BBB+': 0.003, 'BBB': 0.0, 'BBB-': -0.002,
            'BB+': -0.005, 'BB': -0.008, 'BB-': -0.01,
            'B+': -0.015, 'B': -0.02, 'B-': -0.025,
            'CCC+': -0.03, 'CCC': -0.035, 'CCC-': -0.04
        }
        
        self.sector_adjustments = {
            'Technology': 0.015,           # High cash, growth companies
            'Healthcare': 0.01,           # Stable cash flows
            'Consumer Staples': 0.005,    # Stable but lower margins
            'Industrials': 0.0,           # Baseline
            'Materials': -0.005,          # Cyclical, capital intensive
            'Energy (Midstream)': -0.01,  # Capital intensive, regulated
            'Transportation & Logistics': -0.015,  # Asset heavy, cyclical
            'Retail (Apparel)': -0.02,   # Volatile, seasonal
            'Retail (General)': -0.015,  # Competitive, margin pressure  
            'Semicap Supply Chain': -0.01 # Cyclical, capital needs
        }
        
        self.rate_type_adjustments = {
            'floating': 0.01,  # Floating rate loans have refinancing incentive
            'fixed': 0.0       # Fixed rate loans less likely to refinance
        }
        
        self.seniority_adjustments = {
            'Senior': 0.005,      # Senior debt easier to refinance
            'Subordinated': -0.01 # Subordinated debt harder to refinance
        }
    
    def calculate_prepayment_speed(self, loan_factors: PrepaymentFactors) -> float:
        """
        Calculate dynamic prepayment speed based on loan characteristics
        
        Returns: Annual prepayment rate (CPR - Conditional Prepayment Rate)
        """
        # Start with base rate
        prepay_speed = self.base_annual_rate
        
        # LTV adjustment
        if loan_factors.ltv > 0.8:
            ltv_adj = self.ltv_adjustments['high']
        elif loan_factors.ltv < 0.6:
            ltv_adj = self.ltv_adjustments['low']
        else:
            ltv_adj = self.ltv_adjustments['medium']
            
        # DSCR adjustment
        if loan_factors.dscr > 1.5:
            dscr_adj = self.dscr_adjustments['strong']
        elif loan_factors.dscr < 1.2:
            dscr_adj = self.dscr_adjustments['weak']
        else:
            dscr_adj = self.dscr_adjustments['medium']
            
        # Rating adjustment
        rating_adj = self.rating_adjustments.get(loan_factors.credit_rating, -0.01)
        
        # Sector adjustment
        sector_adj = self.sector_adjustments.get(loan_factors.sector, 0.0)
        
        # Rate type adjustment
        rate_type_adj = self.rate_type_adjustments.get(loan_factors.rate_type, 0.0)
        
        # Seniority adjustment
        seniority_adj = self.seniority_adjustments.get(loan_factors.seniority, 0.0)
        
        # Combine all adjustments
        total_prepay_speed = (prepay_speed + ltv_adj + dscr_adj + rating_adj + 
                            sector_adj + rate_type_adj + seniority_adj)
        
        # Apply realistic bounds (0.1% to 25% annually)
        final_speed = max(0.001, min(0.25, total_prepay_speed))
        
        return final_speed
    
    def calculate_prepayment_amount(self, outstanding_principal: float, prepay_speed: float, 
                                  period_type: str = 'annual') -> float:
        """
        Calculate prepayment amount for a given period
        
        Args:
            outstanding_principal: Current outstanding balance
            prepay_speed: Annual prepayment rate (CPR)
            period_type: 'annual' or 'monthly'
            
        Returns: Prepayment amount for the period
        """
        if period_type == 'monthly':
            # Convert annual CPR to SMM (Single Monthly Mortality)
            smm = 1 - (1 - prepay_speed) ** (1/12)
            return outstanding_principal * smm
        else:
            # Annual prepayment
            return outstanding_principal * prepay_speed
    
    def calculate_prepayment_penalty(self, prepay_amount: float, prepay_penalty_rate: float) -> float:
        """Calculate prepayment penalty amount"""
        return prepay_amount * prepay_penalty_rate
    
    def apply_prepayment_to_cashflows(self, schedule: pd.DataFrame, loan_factors: PrepaymentFactors,
                                    prepay_penalty_rate: float) -> pd.DataFrame:
        """
        Apply prepayment modeling to a loan schedule
        
        Args:
            schedule: Loan cashflow schedule DataFrame
            loan_factors: PrepaymentFactors object with loan characteristics  
            prepay_penalty_rate: Prepayment penalty rate
            
        Returns: Enhanced schedule with prepayment columns
        """
        enhanced_schedule = schedule.copy()
        
        # Calculate prepayment speed
        annual_prepay_speed = self.calculate_prepayment_speed(loan_factors)
        
        # Add prepayment columns
        enhanced_schedule['PrepaySpeed'] = annual_prepay_speed
        
        # Calculate monthly prepayment amounts
        enhanced_schedule['PrepayAmount'] = enhanced_schedule.apply(
            lambda row: self.calculate_prepayment_amount(
                row['OutstandingStart'], annual_prepay_speed, 'monthly'
            ), axis=1
        )
        
        # Calculate penalties
        enhanced_schedule['PrepayPenalty'] = enhanced_schedule.apply(
            lambda row: self.calculate_prepayment_penalty(
                row['PrepayAmount'], prepay_penalty_rate
            ), axis=1
        )
        
        # Net prepayment (amount - penalty)
        enhanced_schedule['NetPrepayment'] = (enhanced_schedule['PrepayAmount'] - 
                                            enhanced_schedule['PrepayPenalty'])
        
        return enhanced_schedule
    
    def simulate_prepayment_event(self, outstanding_principal: float, loan_factors: PrepaymentFactors,
                                 prepay_penalty_rate: float, time_period: float = 1.0) -> Dict:
        """
        Simulate a single prepayment event for Monte Carlo analysis
        
        Args:
            outstanding_principal: Current outstanding balance
            loan_factors: Loan characteristics
            prepay_penalty_rate: Penalty rate for prepayment
            time_period: Time period (1.0 = annual, 0.25 = quarterly)
            
        Returns: Dict with prepayment simulation results
        """
        # Calculate prepayment probability for this period
        annual_prepay_speed = self.calculate_prepayment_speed(loan_factors)
        period_prepay_prob = 1 - (1 - annual_prepay_speed) ** time_period
        
        # Determine if prepayment occurs
        prepayment_occurs = np.random.random() < period_prepay_prob
        
        if prepayment_occurs:
            # Full prepayment of outstanding principal
            prepay_amount = outstanding_principal
            penalty_amount = self.calculate_prepayment_penalty(prepay_amount, prepay_penalty_rate)
            net_prepay_cashflow = prepay_amount + penalty_amount
            
            return {
                'prepayment_occurred': True,
                'prepay_amount': prepay_amount,
                'penalty_amount': penalty_amount,
                'net_cashflow': net_prepay_cashflow,
                'remaining_principal': 0.0
            }
        else:
            return {
                'prepayment_occurred': False,
                'prepay_amount': 0.0,
                'penalty_amount': 0.0,
                'net_cashflow': 0.0,
                'remaining_principal': outstanding_principal
            }

def get_prepayment_factors_from_loan(loan) -> PrepaymentFactors:
    """
    Extract prepayment factors from loan object (works with both Enhanced loan specs)
    """
    return PrepaymentFactors(
        ltv=loan.ltv,
        dscr=loan.dscr,
        credit_rating=loan.credit_rating,
        rate_type=loan.rate_type,
        seniority=loan.seniority,
        sector=loan.sector,
        geography=loan.geography,
        current_rate_environment=0.0433  # Will be updated dynamically
    )

# Global prepayment engine instance
prepayment_engine = PrepaymentRiskEngine()