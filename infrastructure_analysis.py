"""
Comparative Analysis Module for IT Infrastructure Models
Evaluates different infrastructure approaches for CBA upgrades in Nigerian banks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class InfrastructureComparativeAnalysis:
    """
    Comparative analysis of IT infrastructure models for Core Banking upgrades
    Focuses on: On-Premise, Hybrid Cloud, Private Cloud, Multi-Cloud
    """
    
    def __init__(self):
        self.models = ['On-Premise', 'Hybrid Cloud', 'Private Cloud', 'Multi-Cloud']
        self.criteria = [
            'Upgrade Success Rate',
            'Average Downtime',
            'Cost Efficiency',
            'Scalability',
            'Regulatory Compliance',
            'Disaster Recovery',
            'Performance',
            'Vendor Lock-in Risk',
            'Power Dependency',
            'Implementation Complexity'
        ]
        
    def generate_comparison_matrix(self):
        """
        Generate a comparison matrix of infrastructure models
        Scores are 1-10 (10 being best)
        """
        
        # Based on literature review and industry best practices
        comparison_data = {
            'Criteria': self.criteria,
            'On-Premise': [6.5, 5.0, 4.0, 5.0, 8.0, 6.0, 7.0, 9.0, 3.0, 6.0],
            'Hybrid Cloud': [8.5, 8.0, 7.5, 8.5, 8.5, 8.5, 8.0, 7.0, 8.0, 7.0],
            'Private Cloud': [7.5, 7.5, 6.0, 7.5, 8.0, 7.5, 7.5, 6.0, 7.0, 6.5],
            'Multi-Cloud': [7.0, 7.0, 5.5, 9.0, 7.0, 9.0, 8.5, 4.0, 8.5, 5.0]
        }
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def calculate_weighted_scores(self, weights=None):
        """
        Calculate weighted scores for each infrastructure model
        
        Parameters:
        -----------
        weights : dict, optional
            Custom weights for each criterion (default: equal weights)
        """
        
        df = self.generate_comparison_matrix()
        
        if weights is None:
            # Default weights (Nigerian banking context)
            weights = {
                'Upgrade Success Rate': 0.15,
                'Average Downtime': 0.12,
                'Cost Efficiency': 0.10,
                'Scalability': 0.08,
                'Regulatory Compliance': 0.15,  # High priority in Nigeria
                'Disaster Recovery': 0.12,
                'Performance': 0.10,
                'Vendor Lock-in Risk': 0.06,
                'Power Dependency': 0.08,  # Important for Nigeria
                'Implementation Complexity': 0.04
            }
        
        # Calculate weighted scores
        weighted_scores = {}
        for model in self.models:
            score = sum(df[df['Criteria'] == criterion][model].values[0] * weights[criterion] 
                       for criterion in self.criteria)
            weighted_scores[model] = round(score, 2)
        
        return weighted_scores, weights
    
    def visualize_comparison(self, save_path='infrastructure_comparison.png'):
        """Generate comprehensive visualization of infrastructure comparison"""
        
        df = self.generate_comparison_matrix()
        weighted_scores, weights = self.calculate_weighted_scores()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparative Analysis: IT Infrastructure Models for CBA Upgrades in Nigerian Banks', 
                     fontsize=16, fontweight='bold')
        
        # 1. Heatmap comparison
        ax1 = axes[0, 0]
        heatmap_data = df.set_index('Criteria')[self.models]
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   vmin=0, vmax=10, ax=ax1, cbar_kws={'label': 'Score (1-10)'})
        ax1.set_title('Infrastructure Model Comparison Matrix', fontweight='bold')
        ax1.set_xlabel('')
        
        # 2. Overall weighted scores
        ax2 = axes[0, 1]
        models = list(weighted_scores.keys())
        scores = list(weighted_scores.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        bars = ax2.barh(models, scores, color=colors)
        ax2.set_xlabel('Weighted Score', fontweight='bold')
        ax2.set_title('Overall Infrastructure Model Ranking', fontweight='bold')
        ax2.set_xlim(0, 10)
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax2.text(score + 0.1, i, f'{score}', va='center', fontweight='bold')
        
        # Highlight best option
        best_idx = scores.index(max(scores))
        bars[best_idx].set_edgecolor('darkgreen')
        bars[best_idx].set_linewidth(3)
        
        # 3. Radar chart for top 2 models
        ax3 = axes[1, 0]
        
        # Select top 2 models
        sorted_models = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        top_2_models = [m[0] for m in sorted_models[:2]]
        
        angles = np.linspace(0, 2 * np.pi, len(self.criteria), endpoint=False).tolist()
        angles += angles[:1]
        
        ax3 = plt.subplot(223, projection='polar')
        
        for model in top_2_models:
            values = df[model].tolist()
            values += values[:1]
            ax3.plot(angles, values, 'o-', linewidth=2, label=model)
            ax3.fill(angles, values, alpha=0.15)
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(self.criteria, size=8)
        ax3.set_ylim(0, 10)
        ax3.set_title('Top 2 Infrastructure Models - Detailed Comparison', 
                     fontweight='bold', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax3.grid(True)
        
        # 4. Criteria weights
        ax4 = axes[1, 1]
        criteria_short = [c[:20] + '...' if len(c) > 20 else c for c in self.criteria]
        weight_values = [weights[c] for c in self.criteria]
        
        bars = ax4.barh(criteria_short, weight_values, color='steelblue')
        ax4.set_xlabel('Weight', fontweight='bold')
        ax4.set_title('Criteria Importance Weights (Nigerian Context)', fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (bar, weight) in enumerate(zip(bars, weight_values)):
            ax4.text(weight + 0.005, i, f'{weight*100:.0f}%', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Infrastructure comparison visualization saved to: {save_path}")
        
        return fig
    
    def generate_recommendation_report(self):
        """Generate detailed recommendation report"""
        
        df = self.generate_comparison_matrix()
        weighted_scores, weights = self.calculate_weighted_scores()
        
        print("\n" + "="*80)
        print("COMPARATIVE ANALYSIS: IT INFRASTRUCTURE MODELS FOR CBA UPGRADES")
        print("Context: Nigerian Banking Sector")
        print("="*80)
        
        print("\n[1] COMPARISON MATRIX")
        print("-" * 80)
        print(df.to_string(index=False))
        
        print("\n\n[2] WEIGHTED SCORES (10-point scale)")
        print("-" * 80)
        sorted_scores = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, score) in enumerate(sorted_scores, 1):
            print(f"{rank}. {model:20s}: {score:.2f}/10")
        
        print("\n\n[3] DETAILED ANALYSIS")
        print("-" * 80)
        
        best_model = sorted_scores[0][0]
        best_score = sorted_scores[0][1]
        
        print(f"\n✓ RECOMMENDED MODEL: {best_model}")
        print(f"  Overall Score: {best_score}/10")
        print(f"\n  Strengths:")
        
        # Identify top strengths
        model_scores = df[best_model].tolist()
        top_criteria_idx = np.argsort(model_scores)[-3:][::-1]
        for idx in top_criteria_idx:
            criterion = self.criteria[idx]
            score = model_scores[idx]
            print(f"    • {criterion}: {score}/10")
        
        print(f"\n  Considerations:")
        # Identify areas for improvement
        bottom_criteria_idx = np.argsort(model_scores)[:2]
        for idx in bottom_criteria_idx:
            criterion = self.criteria[idx]
            score = model_scores[idx]
            print(f"    • {criterion}: {score}/10 (needs attention)")
        
        print("\n\n[4] DEPLOYMENT STRATEGY RECOMMENDATIONS")
        print("-" * 80)
        
        deployment_recommendations = {
            'Hybrid Cloud': {
                'preferred_strategies': ['Blue-Green', 'Canary'],
                'rationale': 'Flexibility to test in cloud before production deployment',
                'implementation': 'Use cloud for testing, on-premise for stable production',
                'risk_mitigation': 'Gradual migration with easy rollback capabilities'
            },
            'On-Premise': {
                'preferred_strategies': ['Blue-Green', 'Phased'],
                'rationale': 'Need parallel environments for safe testing',
                'implementation': 'Duplicate infrastructure for blue-green deployment',
                'risk_mitigation': 'Ensure sufficient hardware capacity for parallel systems'
            },
            'Private Cloud': {
                'preferred_strategies': ['Canary', 'Rolling'],
                'rationale': 'Cloud agility enables progressive rollout',
                'implementation': 'Deploy to subset of instances first',
                'risk_mitigation': 'Monitor metrics before full deployment'
            },
            'Multi-Cloud': {
                'preferred_strategies': ['Blue-Green', 'Canary'],
                'rationale': 'Leverage multiple cloud providers for redundancy',
                'implementation': 'Deploy across clouds with traffic management',
                'risk_mitigation': 'Complex but provides maximum resilience'
            }
        }
        
        rec = deployment_recommendations[best_model]
        print(f"\nFor {best_model}:")
        print(f"  Preferred Deployment Strategies: {', '.join(rec['preferred_strategies'])}")
        print(f"  Rationale: {rec['rationale']}")
        print(f"  Implementation: {rec['implementation']}")
        print(f"  Risk Mitigation: {rec['risk_mitigation']}")
        
        print("\n\n[5] NIGERIAN CONTEXT CONSIDERATIONS")
        print("-" * 80)
        print("""
  Key Factors for Nigerian Banks:
  
  1. Power Infrastructure:
     - Hybrid/Cloud models reduce dependency on local power supply
     - Critical for business continuity
  
  2. CBN Regulatory Compliance:
     - Data localization requirements favor private/hybrid models
     - Must ensure Nigerian data residency
  
  3. Cost Considerations:
     - Initial investment vs. operational costs
     - Hybrid models offer balanced approach
  
  4. Skills Availability:
     - Consider local expertise in cloud vs. on-premise
     - Training requirements for new models
  
  5. Disaster Recovery:
     - Cloud-based models provide better DR capabilities
     - Critical for Nigerian banking sector resilience
        """)
        
        print("\n" + "="*80)
        print("END OF COMPARATIVE ANALYSIS REPORT")
        print("="*80)
        
        return df, weighted_scores, deployment_recommendations
    
    def export_analysis(self, output_file='infrastructure_analysis.csv'):
        """Export comparison data to CSV"""
        
        df = self.generate_comparison_matrix()
        weighted_scores, _ = self.calculate_weighted_scores()
        
        # Add weighted scores as a row
        weighted_row = pd.DataFrame([{
            'Criteria': 'WEIGHTED TOTAL SCORE',
            **weighted_scores
        }])
        
        df_export = pd.concat([df, weighted_row], ignore_index=True)
        df_export.to_csv(output_file, index=False)
        
        print(f"\nComparison data exported to: {output_file}")
        
        return df_export


if __name__ == "__main__":
    # Run comparative analysis
    analyzer = InfrastructureComparativeAnalysis()
    
    print("="*80)
    print("RUNNING COMPARATIVE ANALYSIS OF IT INFRASTRUCTURE MODELS")
    print("="*80)
    
    # Generate and display report
    df, scores, recommendations = analyzer.generate_recommendation_report()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    analyzer.visualize_comparison()
    
    # Export data
    analyzer.export_analysis()
    
    print("\n✓ Comparative analysis complete!")
