"""
training_orchestrator.py — Nydra v0.6.0
============================================
The Unified Engine for ML Training Data Auditing.

This module orchestrates:
1. DataLeakageDetector : Detects contamination between train/test splits.
2. BiasDetector        : Detects fairness issues and representation bias.
3. LabelQualityAnalyzer: Detects label noise, ambiguity, and drift.

Usage:
------
orchestrator = TrainingOrchestrator()
results = orchestrator.run_audit(train_df, test_df, target_col="label", sensitive_cols=["gender", "race"])
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union

from src.vision_nlp.data_leakage  import LeakageOrchestrator as DataLeakageDetector
from src.vision_nlp.bias_detector  import BiasDetector
from src.vision_nlp.label_quality import LabelQualityAnalyzer

logger = logging.getLogger("nydra.training_orchestrator")

class TrainingOrchestrator:
    """
    Orchestrates Machine Learning training data audits.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def run_audit(
        self, 
        train_df: pd.DataFrame, 
        test_df: Optional[pd.DataFrame] = None, 
        target_col: str = "target",
        sensitive_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Runs a comprehensive training data audit.
        """
        results = {"status": "success", "summary": {}}

        # 1. Label Quality Audit
        try:
            if self.verbose: logger.info("Running Label Quality Audit...")
            # Ensure target_col exists
            if target_col not in train_df.columns:
                raise ValueError(f"Target column '{target_col}' not found in training data.")
            
            lq_analyzer = LabelQualityAnalyzer(verbose=self.verbose)
            feature_cols = [c for c in train_df.columns if c != target_col]
            lq_report = lq_analyzer.analyze(train_df, label_col=target_col, feature_cols=feature_cols)
            
            results["label_quality"] = lq_report
            results["summary"]["label_quality_score"] = getattr(lq_report, 'label_quality_score', 0)
            results["summary"]["label_errors_detected"] = getattr(lq_report, 'n_label_errors', 0)
        except Exception as e:
            logger.error(f"Label quality audit failed: {e}")
            results["label_quality_error"] = str(e)
            results["summary"]["label_quality_score"] = 0

        # 2. Data Leakage Audit
        try:
            if self.verbose: logger.info("Running Data Leakage Audit...")
            leakage_audit = DataLeakageDetector(
                df=train_df,
                train_df=train_df,
                test_df=test_df if test_df is not None else train_df,
                target_col=target_col
            )
            leakage_report = leakage_audit.run()
            results["leakage"] = leakage_report
            
            # Leakage report is a dict
            score_dict = leakage_report.get("score", {})
            results["summary"]["leakage_score"] = score_dict.get("leakage_risk_score", 100)
            results["summary"]["leakage_risk"] = score_dict.get("label", "Unknown")
        except Exception as e:
            logger.error(f"Leakage audit failed: {e}")
            results["leakage_error"] = str(e)
            results["summary"]["leakage_score"] = 0
            results["summary"]["leakage_risk"] = "Error"

        # 3. Bias & Fairness Audit
        if sensitive_cols:
            try:
                # Filter sensitive_cols to those that actually exist
                existing_sens = [c for c in sensitive_cols if c in train_df.columns]
                if not existing_sens:
                    results["bias_info"] = f"None of the sensitive columns {sensitive_cols} found in data."
                else:
                    if self.verbose: logger.info(f"Running Bias Audit on {existing_sens}...")
                    bias_audit = BiasDetector(
                        df=train_df,
                        sensitive_cols=existing_sens,
                        target_col=target_col,
                        train_df=train_df,
                        test_df=test_df if test_df is not None else train_df
                    )
                    bias_report = bias_audit.detect()
                    results["bias"] = bias_report
                    
                    # Bias report is a dict
                    score_dict = bias_report.get("bias_score", {})
                    results["summary"]["bias_score"] = score_dict.get("overall_score", 100)
                    results["summary"]["bias_verdict"] = score_dict.get("verdict", "Unknown")
            except Exception as e:
                logger.error(f"Bias audit failed: {e}")
                results["bias_error"] = str(e)
                results["summary"]["bias_score"] = 0
                results["summary"]["bias_verdict"] = "Error"
        else:
            results["bias_info"] = "No sensitive columns provided for bias audit."

        return results

def run_training_audit(
    train_df: pd.DataFrame, 
    test_df: Optional[pd.DataFrame] = None, 
    target_col: str = "target",
    sensitive_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Convenience function to run a training audit."""
    return TrainingOrchestrator().run_audit(train_df, test_df, target_col, sensitive_cols)

