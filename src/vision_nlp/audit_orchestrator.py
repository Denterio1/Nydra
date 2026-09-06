"""
audit_orchestrator.py — Nydra v0.6.0
==========================================
The Unified Engine for Image Dataset Auditing.

This module orchestrates the full pipeline:
1. ImageLoader  : Scans and loads images from directory or uploads.
2. ImageQuality : Analyzes individual image quality (brightness, blur, etc.).
3. ImageAnalyzer: Performs dataset-level analysis (bias, diversity, outliers).
4. ImageCleaner : (Optional) Automatically repairs common quality issues.
5. ImageReport  : Generates professional multi-format reports.

Usage:
------
orchestrator = AuditOrchestrator(target_dir="data/images")
results = orchestrator.run_full_audit(clean=True)
report_paths = orchestrator.generate_reports(output_dir="reports")
"""

import os
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.vision_nlp.image_loader import ImageLoader
from src.vision_nlp.image_quality import analyze_dataset_quality
from src.vision_nlp.image_analyzer import ImageAnalyzer
from src.vision_nlp.image_cleaner import ImageCleaner
from src.vision_nlp.image_report import generate_report

logger = logging.getLogger("nydra.audit_orchestrator")

class AuditOrchestrator:
    """
    Orchestrates the image auditing process, providing a clean API for
    CLI and Web interfaces.
    """

    def __init__(self, target_dir: Optional[str] = None, verbose: bool = True):
        self.target_dir = target_dir
        self.verbose = verbose
        
        self.loader_df = None
        self.quality_df = None
        self.analyzer_df = None
        self.cleaner_df = None
        
        self.loader_report = None
        self.quality_report = None
        
        self.results = {}

    def run_full_audit(self, clean: bool = False, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs the complete audit pipeline.
        
        Parameters:
        -----------
        clean : bool
            Whether to run the ImageCleaner to attempt automated fixes.
        output_dir : Optional[str]
            Directory to save cleaned images if clean=True.
            
        Returns:
        --------
        Dict containing DataFrames and reports from each stage.
        """
        if not self.target_dir:
            raise ValueError("target_dir must be provided to run an audit.")

        try:
            # 1. Load & Scan
            if self.verbose: logger.info(f"Step 1/5: Scanning & Loading images from {self.target_dir}...")
            loader = ImageLoader(self.target_dir, verbose=self.verbose)
            self.loader_df, self.loader_report = loader.load()
            
            if self.loader_df is None or self.loader_df.empty:
                logger.warning("No valid images found in the target directory.")
                return {"status": "empty", "message": "No valid images found."}

            # 2. Quality Analysis
            if self.verbose: logger.info("Step 2/5: Analyzing individual image quality...")
            self.quality_df, self.quality_report = analyze_dataset_quality(self.loader_df, verbose=self.verbose)

            # 3. Dataset-level Analysis
            if self.verbose: logger.info("Step 3/5: Analyzing dataset structure, diversity, and bias...")
            analyzer = ImageAnalyzer(
                loader_df=self.loader_df,
                quality_df=self.quality_df,
                image_dir=self.target_dir,
                path_column="file_path",
                label_column="label"
            )
            self.analyzer_report = analyzer.analyze_all(verbose=self.verbose)
            # Some parts of the code might expect a DataFrame for the analyzer stage
            self.analyzer_df = analyzer.get_summary_dataframe()

            # 4. (Optional) Cleaning
            if clean:
                if self.verbose: logger.info("Step 4/5: Running automated cleaning pipeline...")
                if not output_dir:
                    output_dir = os.path.join(os.path.dirname(self.target_dir), f"{os.path.basename(self.target_dir)}_cleaned")
                
                cleaner = ImageCleaner(input_dir=self.target_dir, output_dir=output_dir)
                # Combine loader and quality info for the cleaner
                full_info_df = self.loader_df.merge(self.quality_df, on="file_path", how="left")
                self.cleaner_df = cleaner.clean(df=full_info_df)
            else:
                if self.verbose: logger.info("Step 4/5: Skipping automated cleaning.")

            self.results = {
                "loader_df": self.loader_df,
                "quality_df": self.quality_df,
                "analyzer_df": self.analyzer_df,
                "analyzer_report": self.analyzer_report,
                "cleaner_df": self.cleaner_df,
                "loader_report": self.loader_report,
                "quality_report": self.quality_report,
                "status": "success"
            }
            
            return self.results

        except Exception as e:
            logger.error(f"Audit failed: {e}")
            logger.debug(traceback.format_exc())
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

    def generate_reports(self, output_dir: str = "reports") -> Dict[str, str]:
        """
        Generates professional reports based on the audit results.
        """
        if not self.results or self.results.get("status") != "success":
            raise ValueError("Audit must be run successfully before generating reports.")

        # Robustness: If attributes are missing (e.g. results were injected), restore them
        if self.loader_df is None: self.loader_df = self.results.get("loader_df")
        if self.quality_df is None: self.quality_df = self.results.get("quality_df")
        if self.analyzer_df is None: self.analyzer_df = self.results.get("analyzer_df")
        if self.cleaner_df is None: self.cleaner_df = self.results.get("cleaner_df")

        if self.verbose: logger.info(f"Step 5/5: Generating professional reports in {output_dir}...")
        
        report_paths = generate_report(
            self.loader_df,
            quality_df=self.quality_df,
            analyzer_df=self.analyzer_df,
            cleaner_df=self.cleaner_df,
            output_dir=output_dir
        )
        
        return report_paths

def run_image_audit(path: str, clean: bool = False, output_dir: str = "reports") -> Dict[str, Any]:
    """Convenience function to run the full audit."""
    orchestrator = AuditOrchestrator(target_dir=path)
    results = orchestrator.run_full_audit(clean=clean)
    if results["status"] == "success":
        report_paths = orchestrator.generate_reports(output_dir=output_dir)
        results["report_paths"] = report_paths
    return results

