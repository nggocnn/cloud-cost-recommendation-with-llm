#!/usr/bin/env python3
"""
Enhanced evidence tracing for LLM cost recommendations.

This script analyzes the recommendations and traces them back to the actual input data,
identifying where evidence is missing or estimated vs. based on real data.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class RecommendationEvidenceTracer:
    """Trace recommendations back to input data sources"""
    
    def __init__(self, analysis_file: str, cli_data_dir: str, transformed_data_dir: str):
        self.analysis_file = Path(analysis_file)
        self.cli_data_dir = Path(cli_data_dir)
        self.transformed_data_dir = Path(transformed_data_dir)
        
        # Load analysis results
        with open(self.analysis_file, 'r') as f:
            self.analysis = json.load(f)
            
        # Load transformed data for comparison
        self.inventory_data = self._load_inventory_data()
        self.billing_data = self._load_billing_data()
        self.metrics_data = self._load_metrics_data()

    def trace_all_recommendations(self):
        """Trace evidence for all recommendations"""
        print("🔍 Evidence Tracing Report")
        print("=" * 80)
        print(f"Analysis File: {self.analysis_file}")
        print(f"Generated: {self.analysis['generated_at']}")
        print(f"Total Recommendations: {self.analysis['total_recommendations']}")
        print()
        
        for i, rec in enumerate(self.analysis['recommendations'], 1):
            print(f"\n{'='*60}")
            print(f"RECOMMENDATION #{i}: {rec['resource_id']}")
            print(f"{'='*60}")
            self._trace_recommendation_evidence(rec)
            
    def _trace_recommendation_evidence(self, recommendation: Dict):
        """Trace evidence for a single recommendation"""
        resource_id = recommendation['resource_id']
        service = recommendation['service']
        
        print(f"Resource ID: {resource_id}")
        print(f"Service: {service}")
        print(f"Recommendation Type: {recommendation['recommendation_type']}")
        print(f"Estimated Savings: ${recommendation['estimated_monthly_savings']}/month")
        print(f"Confidence Score: {recommendation['confidence_score']}")
        print()
        
        # Find source data
        inventory_item = self._find_inventory_item(resource_id)
        billing_items = self._find_billing_items(resource_id)
        metrics_item = self._find_metrics_item(resource_id)
        cli_files = self._find_cli_files(resource_id, service)
        
        print("📊 DATA SOURCES ANALYSIS:")
        print("-" * 40)
        
        # Inventory Evidence
        print("1. INVENTORY DATA:")
        if inventory_item:
            print(f"   ✅ Found in transformed inventory")
            print(f"   📁 Source: {inventory_item}")
            if cli_files['inventory_source']:
                print(f"   🔗 Original CLI file: {cli_files['inventory_source']}")
        else:
            print(f"   ❌ NOT found in inventory data")
        print()
        
        # Billing Evidence  
        print("2. BILLING DATA:")
        if billing_items:
            print(f"   ✅ Found {len(billing_items)} billing record(s)")
            for item in billing_items:
                print(f"   💰 Cost: ${item.get('lineItem/UnblendedCost', 'N/A')}")
                print(f"   📈 Usage: {item.get('lineItem/UsageAmount', 'N/A')} {item.get('lineItem/UsageUnit', '')}")
            print(f"   ⚠️  Note: Billing costs are ESTIMATED (not from real AWS bill)")
        else:
            print(f"   ❌ NO billing data found")
        print()
        
        # Metrics Evidence
        print("3. PERFORMANCE METRICS:")
        if metrics_item:
            print(f"   ✅ Found metrics data")
            self._print_metrics_details(metrics_item)
        else:
            print(f"   ❌ NO metrics data found")
            if cli_files['metrics_files']:
                print(f"   📁 Available CLI metrics files: {cli_files['metrics_files']}")
                self._analyze_cli_metrics(cli_files['metrics_files'])
        print()
        
        # Evidence Quality Analysis
        print("4. EVIDENCE QUALITY ASSESSMENT:")
        self._assess_evidence_quality(recommendation, inventory_item, billing_items, metrics_item, cli_files)
        print()
        
        # LLM Claims vs Reality
        print("5. LLM CLAIMS vs ACTUAL DATA:")
        self._compare_llm_claims_to_data(recommendation, inventory_item, billing_items, metrics_item)

    def _find_inventory_item(self, resource_id: str) -> Optional[Dict]:
        """Find inventory item for resource"""
        for item in self.inventory_data:
            if item.get('resource_id') == resource_id:
                return item
        return None
        
    def _find_billing_items(self, resource_id: str) -> List[Dict]:
        """Find billing items for resource"""
        if not self.billing_data:
            return []
        return [item for item in self.billing_data if item.get('lineItem/ResourceId') == resource_id]
        
    def _find_metrics_item(self, resource_id: str) -> Optional[Dict]:
        """Find metrics item for resource"""
        if not self.metrics_data:
            return None
        for item in self.metrics_data:
            if item.get('resource_id') == resource_id:
                return item
        return None
        
    def _find_cli_files(self, resource_id: str, service: str) -> Dict[str, Any]:
        """Find original CLI files that might contain this resource"""
        cli_files = {
            'inventory_source': None,
            'metrics_files': [],
            'all_related_files': []
        }
        
        service_lower = service.lower()
        
        # Find all JSON files that might relate to this service/resource
        for json_file in self.cli_data_dir.glob("*.json"):
            filename = json_file.name.lower()
            
            # Skip Zone.Identifier files
            if "zone.identifier" in filename:
                continue
                
            # Check if file relates to the service
            if any(s in filename for s in [service_lower.split('.')[-1], resource_id.lower()]):
                cli_files['all_related_files'].append(str(json_file))
                
                # Specific categorization
                if any(keyword in filename for keyword in ['instance', 'volume', 'function', 'distribution']):
                    cli_files['inventory_source'] = str(json_file)
                elif 'metrics' in filename:
                    cli_files['metrics_files'].append(str(json_file))
                    
        return cli_files
        
    def _analyze_cli_metrics(self, metrics_files: List[str]):
        """Analyze CLI metrics files to see what data is actually available"""
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    
                datapoints = data.get('Datapoints', [])
                label = data.get('Label', 'Unknown')
                
                if datapoints:
                    print(f"   📊 {Path(metrics_file).name}: {len(datapoints)} datapoints for {label}")
                    # Show sample data
                    if len(datapoints) > 0:
                        sample = datapoints[0]
                        print(f"      Sample: {sample}")
                else:
                    print(f"   ⚠️  {Path(metrics_file).name}: NO datapoints (empty metrics)")
                    
            except Exception as e:
                print(f"   ❌ Error reading {metrics_file}: {e}")
                
    def _print_metrics_details(self, metrics_item: Dict):
        """Print details of metrics data"""
        metrics_keys = [k for k in metrics_item.keys() if k not in ['resource_id', 'timestamp', 'period_days']]
        
        for key in metrics_keys:
            value = metrics_item.get(key)
            if value is not None:
                print(f"   📊 {key}: {value}")
                
    def _assess_evidence_quality(self, recommendation: Dict, inventory_item: Optional[Dict], 
                                billing_items: List[Dict], metrics_item: Optional[Dict],
                                cli_files: Dict[str, Any]):
        """Assess the quality of evidence supporting the recommendation"""
        
        evidence_score = 0
        max_score = 4
        issues = []
        
        # Inventory data (required)
        if inventory_item:
            evidence_score += 1
            print(f"   ✅ Inventory data: AVAILABLE")
        else:
            print(f"   ❌ Inventory data: MISSING")
            issues.append("No inventory data found")
            
        # Billing data (important for cost calculations)
        if billing_items:
            evidence_score += 1
            print(f"   ⚠️  Billing data: ESTIMATED (not real AWS bills)")
        else:
            print(f"   ❌ Billing data: MISSING")
            issues.append("No billing data")
            
        # Metrics data (crucial for usage-based recommendations)
        if metrics_item:
            evidence_score += 1
            print(f"   ✅ Metrics data: AVAILABLE")
        else:
            print(f"   ❌ Metrics data: MISSING")
            issues.append("No performance metrics")
            
        # Original CLI data
        if cli_files['all_related_files']:
            evidence_score += 1
            print(f"   ✅ CLI source data: AVAILABLE")
        else:
            print(f"   ❌ CLI source data: NOT FOUND")
            
        # Overall assessment
        print(f"\n   📋 Evidence Quality Score: {evidence_score}/{max_score}")
        
        if evidence_score <= 1:
            print(f"   🔴 POOR - Recommendation based on minimal data")
        elif evidence_score <= 2:
            print(f"   🟡 FAIR - Some data missing, recommendations may be inaccurate")
        elif evidence_score <= 3:
            print(f"   🟠 GOOD - Most data available, but some gaps")
        else:
            print(f"   🟢 EXCELLENT - Comprehensive data available")
            
        if issues:
            print(f"   ⚠️  Issues: {', '.join(issues)}")
            
    def _compare_llm_claims_to_data(self, recommendation: Dict, inventory_item: Optional[Dict],
                                   billing_items: List[Dict], metrics_item: Optional[Dict]):
        """Compare LLM claims in evidence to actual available data"""
        
        evidence = recommendation.get('evidence', {})
        
        print("   LLM Evidence Claims vs Reality:")
        print("   " + "-" * 35)
        
        # Metrics analysis claim
        metrics_claim = evidence.get('metrics_analysis', '')
        if metrics_claim:
            print(f"   🤖 LLM Claim: {metrics_claim}")
            if metrics_item:
                print(f"   ✅ Reality: Metrics data available to support claim")
            else:
                print(f"   ❌ Reality: NO metrics data - claim is FABRICATED/ESTIMATED")
        
        # Cost breakdown claim
        cost_claim = evidence.get('cost_breakdown', '')
        if cost_claim:
            print(f"   🤖 LLM Claim: {cost_claim}")
            if billing_items:
                actual_cost = billing_items[0].get('lineItem/UnblendedCost')
                print(f"   ⚠️  Reality: Cost ${actual_cost} is ESTIMATED, not from real AWS bill")
            else:
                print(f"   ❌ Reality: NO billing data - cost claim is ESTIMATED")
                
        # Performance impact claim
        perf_claim = evidence.get('performance_impact', '')
        if perf_claim:
            print(f"   🤖 LLM Claim: {perf_claim}")
            if metrics_item:
                print(f"   ✅ Reality: Has performance data to assess impact")
            else:
                print(f"   ❌ Reality: NO performance data - impact assessment is SPECULATIVE")

    def _load_inventory_data(self) -> List[Dict]:
        """Load inventory data"""
        inventory_file = self.transformed_data_dir / "inventory" / "cli_inventory.json"
        if inventory_file.exists():
            with open(inventory_file, 'r') as f:
                return json.load(f)
        return []
        
    def _load_billing_data(self) -> List[Dict]:
        """Load billing data"""
        billing_file = self.transformed_data_dir / "billing" / "cli_billing.csv"
        if billing_file.exists():
            df = pd.read_csv(billing_file)
            return df.to_dict('records')
        return []
        
    def _load_metrics_data(self) -> List[Dict]:
        """Load metrics data"""
        metrics_file = self.transformed_data_dir / "metrics" / "cli_metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            return df.to_dict('records')
        return []

    def generate_improved_data_collection_guide(self):
        """Generate guide for collecting better data"""
        print("\n" + "=" * 80)
        print("📋 IMPROVED DATA COLLECTION GUIDE")
        print("=" * 80)
        print()
        print("To get more accurate recommendations with proper evidence:")
        print()
        
        print("1. 🔍 COLLECT REAL CLOUDWATCH METRICS:")
        print("   For CloudFront distributions:")
        print("   aws cloudwatch get-metric-statistics \\")
        print("     --namespace AWS/CloudFront \\")
        print("     --metric-name Requests \\")
        print("     --dimensions Name=DistributionId,Value=E38746PWHXCI05 \\")
        print("     --start-time 2025-08-01T00:00:00Z \\")
        print("     --end-time 2025-09-01T00:00:00Z \\")
        print("     --period 86400 \\")
        print("     --statistics Sum,Average")
        print()
        
        print("2. 💰 GET REAL BILLING DATA:")
        print("   - Enable AWS Cost and Usage Reports")
        print("   - Download actual billing CSV from AWS")
        print("   - Use real costs instead of estimates")
        print()
        
        print("3. 🌍 COLLECT GEOGRAPHIC METRICS:")
        print("   aws cloudfront get-distribution-config \\")
        print("     --id E38746PWHXCI05")
        print("   aws logs start-query \\")
        print("     --log-group-name /aws/cloudfront/E38746PWHXCI05 \\")
        print("     --query-string 'fields @timestamp, c-ip | stats count() by c-ip'")


def main():
    parser = argparse.ArgumentParser(description="Trace LLM recommendation evidence")
    parser.add_argument("--analysis-file", default="cli_analysis.json",
                       help="LLM analysis JSON file")
    parser.add_argument("--cli-dir", default="data/cli",
                       help="Original CLI data directory")
    parser.add_argument("--data-dir", default="data",
                       help="Transformed data directory")
    
    args = parser.parse_args()
    
    tracer = RecommendationEvidenceTracer(
        args.analysis_file, 
        args.cli_dir, 
        args.data_dir
    )
    
    tracer.trace_all_recommendations()
    tracer.generate_improved_data_collection_guide()


if __name__ == "__main__":
    main()
