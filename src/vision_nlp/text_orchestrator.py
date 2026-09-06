"""
text_orchestrator.py — Nydra v0.6.0
=========================================
The Unified Engine for Text Intelligence & Quality Auditing.

This module orchestrates:
1. TextAnalyzer : Advanced text intelligence (sentiment, keywords, style, etc.).
2. TextQuality  : Deep quality audit (PII, noise, duplicates, coherence, etc.).
3. Reporting    : Generates unified text analysis reports.

Usage:
------
orchestrator = TextOrchestrator()
results = orchestrator.analyze_text("Your long text here...")
report_paths = orchestrator.generate_reports(results, output_dir="reports")
"""

import os
import logging
import traceback
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.vision_nlp.text_analyzer import TextAnalyzer
from src.vision_nlp.text_quality import SmartTextQualityChecker

logger = logging.getLogger("nydra.text_orchestrator")

class TextOrchestrator:
    """
    Orchestrates text analysis and quality auditing.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.analyzer = None
        self.quality_checker = SmartTextQualityChecker()

    def analyze_text(self, text: str, name: str = "document") -> Dict[str, Any]:
        """
        Runs both intelligence analysis and quality audit on a single text.
        """
        if not text or not isinstance(text, str):
            return {"status": "error", "message": "Invalid text input."}

        try:
            if self.verbose: logger.info(f"Analyzing text: {name}...")
            
            # 1. Text Intelligence
            ta = TextAnalyzer(text)
            intelligence = ta.analyze()
            
            # 2. Text Quality
            quality = self.quality_checker.check(text, verbose=self.verbose)
            
            return {
                "status": "success",
                "name": name,
                "intelligence": intelligence,
                "quality": quality,
                "text_length": len(text)
            }
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {"status": "error", "message": str(e)}

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """
        Runs analysis on a collection of texts in a DataFrame.
        """
        if text_column not in df.columns:
            return {"status": "error", "message": f"Column '{text_column}' not found."}
            
        texts = df[text_column].dropna().astype(str).tolist()
        if not texts:
            return {"status": "error", "message": "No text found in column."}

        try:
            if self.verbose: logger.info(f"Analyzing corpus from column '{text_column}' ({len(texts)} rows)...")
            
            # For now, we use the checker's corpus mode for quality
            quality_corpus = self.quality_checker.check_corpus(texts)
            
            # Aggregate some intelligence (e.g. sentiment)
            # This is a placeholder for more advanced corpus-level intelligence
            return {
                "status": "success",
                "count": len(texts),
                "quality_summary": quality_corpus
            }
        except Exception as e:
            logger.error(f"Corpus analysis failed: {e}")
            return {"status": "error", "message": str(e)}

def run_text_audit(text_or_df: Union[str, pd.DataFrame], text_column: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to run a text audit."""
    orchestrator = TextOrchestrator()
    if isinstance(text_or_df, str):
        return orchestrator.analyze_text(text_or_df)
    else:
        return orchestrator.analyze_dataframe(text_or_df, text_column or "text")

