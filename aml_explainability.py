"""
AML Model Explainability with SHAP
Purpose: Provide interpretable explanations for AML predictions (regulatory requirement)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


class AMLExplainer:
    """SHAP-based explainability for AML models"""

    def __init__(self, model, X_train, feature_names):
        """
        Initialize explainer

        Args:
            model: Trained XGBoost/LightGBM/RF model
            X_train: Training data for background samples
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names

        # Create SHAP explainer
        print("Creating SHAP explainer (this may take a moment)...")

        # Use TreeExplainer for tree-based models
        self.explainer = shap.TreeExplainer(model)

        # Calculate SHAP values for training set (sample for speed)
        sample_size = min(1000, len(X_train))
        self.X_sample = X_train.sample(sample_size, random_state=42)
        self.shap_values = self.explainer.shap_values(self.X_sample)

        print("SHAP explainer ready!")

    def plot_summary(self):
        """Create SHAP summary plot showing feature importance"""
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_sample,
                          feature_names=self.feature_names,
                          show=False)
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("SHAP summary plot saved!")

    def explain_prediction(self, X_instance, instance_id="Sample"):
        """
        Explain a single prediction

        Args:
            X_instance: Single row DataFrame
            instance_id: Identifier for the instance
        """
        shap_values_instance = self.explainer.shap_values(X_instance)

        # Waterfall plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_instance[0],
                base_values=self.explainer.expected_value,
                data=X_instance.iloc[0],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'/mnt/user-data/outputs/shap_waterfall_{instance_id}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Get top contributing features
        feature_contributions = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_values_instance[0],
            'feature_value': X_instance.iloc[0].values
        })
        feature_contributions['abs_shap'] = np.abs(feature_contributions['shap_value'])
        feature_contributions = feature_contributions.sort_values('abs_shap', ascending=False)

        return feature_contributions.head(10)

    def plot_dependence(self, feature_name):
        """Plot SHAP dependence for a specific feature"""
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'/mnt/user-data/outputs/shap_dependence_{feature_name}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def generate_explanation_report(self, X_test, y_test, predictions, top_n=5):
        """Generate explanation report for top false positive reductions"""

        # Find cases where model correctly identified false positives
        correct_fps = (y_test == 0) & (predictions == 0)
        fp_indices = y_test[correct_fps].index[:top_n]

        report = []
        report.append("=" * 80)
        report.append("SHAP EXPLANATION REPORT - False Positive Examples")
        report.append("=" * 80)

        for i, idx in enumerate(fp_indices, 1):
            X_instance = X_test.loc[[idx]]
            contributions = self.explain_prediction(X_instance, f"FP_{i}")

            report.append(f"\nExample {i}: Transaction correctly identified as FALSE POSITIVE")
            report.append("-" * 40)
            report.append("Top Contributing Features:")
            report.append(contributions[['feature', 'feature_value', 'shap_value']].to_string(index=False))
            report.append("")

        return "\n".join(report)


def add_shap_analysis(model, X_train, X_test, y_test, predictions, feature_names):
    """Add SHAP analysis to the main pipeline"""

    print("\n" + "=" * 80)
    print("GENERATING SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 80)

    # Create explainer
    explainer = AMLExplainer(model, X_train, feature_names)

    # Generate summary plot
    explainer.plot_summary()

    # Generate explanation report
    report = explainer.generate_explanation_report(X_test, y_test, predictions)

    with open('/mnt/user-data/outputs/shap_explanation_report.txt', 'w') as f:
        f.write(report)

    print(report)

    # Plot dependence for top features
    top_features = ['transaction_amount', 'alert_rate', 'amount_to_avg_ratio']
    for feature in top_features:
        if feature in feature_names:
            explainer.plot_dependence(feature)

    print("\nSHAP analysis complete!")
    print("Generated files:")
    print("  - shap_summary.png")
    print("  - shap_explanation_report.txt")
    print("  - shap_waterfall_*.png")
    print("  - shap_dependence_*.png")

    return explainer