import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict


class RuleVisualizer:
    def __init__(self, rules_file):
        """
        Initialize visualizer with rules data
        """
        self.rules_df = pd.read_csv(rules_file)
        self.output_dir = "outputs"
        self._setup_output_dir()

    def _setup_output_dir(self):
        """Create output directory if it doesn't exist"""
        import os

        os.makedirs(self.output_dir, exist_ok=True)

    def plot_support_confidence_scatter(self):
        """
        Create scatter plot of support vs confidence, with lift as color intensity
        """
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            self.rules_df["support"],
            self.rules_df["confidence"],
            c=self.rules_df["lift"],
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter, label="Lift")
        plt.xlabel("Support")
        plt.ylabel("Confidence")
        plt.title("Support vs Confidence (color = Lift)")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/support_confidence_scatter.png")
        plt.close()

    def plot_top_rules_by_lift(self, n=20):
        """
        Plot top N rules by lift
        """
        top_rules = self.rules_df.nlargest(n, "lift")

        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(top_rules)), top_rules["lift"])
        plt.yticks(
            range(len(top_rules)),
            [
                f"{' → '.join([str(a), str(c)])}"
                for a, c in zip(top_rules["antecedent"], top_rules["consequent"])
            ],
            fontsize=8,
        )
        plt.xlabel("Lift")
        plt.title(f"Top {n} Rules by Lift")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/top_rules_lift.png")
        plt.close()

    def create_network_graph(self, min_lift=1.5):
        """
        Create network graph of rules with minimum lift
        """
        G = nx.DiGraph()

        # Filter rules by minimum lift
        filtered_rules = self.rules_df[self.rules_df["lift"] >= min_lift]

        # Create edges
        for _, rule in filtered_rules.iterrows():
            antecedents = (
                eval(rule["antecedent"])
                if isinstance(rule["antecedent"], str)
                else rule["antecedent"]
            )
            consequents = (
                eval(rule["consequent"])
                if isinstance(rule["consequent"], str)
                else rule["consequent"]
            )

            for a in antecedents:
                for c in consequents:
                    G.add_edge(
                        a,
                        c,
                        weight=rule["lift"],
                        support=rule["support"],
                        confidence=rule["confidence"],
                    )

        # Create network visualization
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

        # Draw edges with varying thickness based on lift
        edges = G.edges()
        weights = [G[u][v]["weight"] for u, v in edges]
        nx.draw_networkx_edges(
            G,
            pos,
            width=[w / max(weights) * 3 for w in weights],
            edge_color="gray",
            arrows=True,
            arrowsize=20,
        )

        plt.title("Network of Association Rules")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/rules_network.png", bbox_inches="tight")
        plt.close()

    def generate_pattern_frequency_analysis(self):
        """
        Analyze frequency of patterns in rules
        """
        # Count pattern occurrences in antecedents and consequents
        pattern_counts = defaultdict(int)

        for _, rule in self.rules_df.iterrows():
            antecedents = (
                eval(rule["antecedent"])
                if isinstance(rule["antecedent"], str)
                else rule["antecedent"]
            )
            consequents = (
                eval(rule["consequent"])
                if isinstance(rule["consequent"], str)
                else rule["consequent"]
            )

            for pattern in antecedents + consequents:
                pattern_counts[pattern] += 1

        # Create and save frequency plot
        plt.figure(figsize=(12, 6))
        patterns = list(pattern_counts.keys())
        counts = list(pattern_counts.values())

        plt.bar(range(len(patterns)), counts)
        plt.xticks(range(len(patterns)), patterns, rotation=45, ha="right")
        plt.xlabel("Pattern")
        plt.ylabel("Frequency")
        plt.title("Pattern Frequency in Rules")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pattern_frequency.png")
        plt.close()

        return pattern_counts

    def generate_summary_report(self):
        """
        Generate a summary report of the analysis
        """
        pattern_counts = self.generate_pattern_frequency_analysis()

        report = [
            "Association Rules Analysis Summary",
            "=================================",
            f"\nTotal number of rules: {len(self.rules_df)}",
            f"Average lift: {self.rules_df['lift'].mean():.2f}",
            f"Average confidence: {self.rules_df['confidence'].mean():.2f}",
            f"Average support: {self.rules_df['support'].mean():.2f}",
            "\nTop 5 patterns by frequency:",
        ]

        # Add top 5 patterns
        sorted_patterns = sorted(
            pattern_counts.items(), key=lambda x: x[1], reverse=True
        )
        for pattern, count in sorted_patterns[:5]:
            report.append(f"- {pattern}: {count} occurrences")

        # Add top 5 rules by lift
        report.extend(["\nTop 5 rules by lift:", "-------------------"])

        top_rules = self.rules_df.nlargest(5, "lift")
        for _, rule in top_rules.iterrows():
            report.append(
                f"- {rule['antecedent']} → {rule['consequent']}"
                f" (lift: {rule['lift']:.2f}, conf: {rule['confidence']:.2f}, support: {rule['support']:.2f})"
            )

        # Save report
        with open(f"{self.output_dir}/analysis_summary.txt", "w") as f:
            f.write("\n".join(report))


def main():
    input_file = "outputs/BINANCE_BTCUSDT_D1_rules.csv"

    print("Starting visualization and analysis...")
    visualizer = RuleVisualizer(input_file)

    print("Generating support-confidence scatter plot...")
    visualizer.plot_support_confidence_scatter()

    print("Generating top rules by lift plot...")
    visualizer.plot_top_rules_by_lift()

    print("Generating network graph of rules...")
    visualizer.create_network_graph()

    print("Generating pattern frequency analysis...")
    visualizer.generate_pattern_frequency_analysis()

    print("Generating summary report...")
    visualizer.generate_summary_report()

    print("\nVisualization complete! Check the 'outputs' directory for results.")


if __name__ == "__main__":
    main()
