import pandas as pd
from collections import defaultdict
from itertools import combinations


class AprioriAnalyzer:
    def __init__(self, min_support=0.01, min_confidence=0.5):
        """
        Initialize with minimum support and confidence thresholds
        min_support: minimum support threshold (0-1)
        min_confidence: minimum confidence threshold (0-1)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.transactions = []
        self.item_counts = defaultdict(int)
        self.n_transactions = 0
        self.frequent_itemsets = {}
        self.rules = []

    def fit(self, transactions):
        """
        Fit the Apriori algorithm to the transaction data
        """
        self.transactions = transactions
        self.n_transactions = len(transactions)

        # Generate L1 (frequent 1-itemsets)
        self._generate_L1()

        # Generate Lk (frequent k-itemsets)
        k = 2
        while True:
            frequent_k = self._generate_Lk(k)
            if not frequent_k:
                break
            self.frequent_itemsets[k] = frequent_k
            k += 1

        # Generate association rules
        self._generate_rules()

        return self

    def _generate_L1(self):
        """
        Generate frequent 1-itemsets
        """
        # Count occurrences of each item
        for transaction in self.transactions:
            for item in transaction:
                self.item_counts[item] += 1

        # Filter items by minimum support
        l1 = {
            frozenset([item]): count / self.n_transactions
            for item, count in self.item_counts.items()
            if count / self.n_transactions >= self.min_support
        }

        if l1:
            self.frequent_itemsets[1] = l1

    def _generate_Lk(self, k):
        """
        Generate frequent k-itemsets
        """
        if k - 1 not in self.frequent_itemsets:
            return {}

        # Generate candidate k-itemsets
        prev_frequent = self.frequent_itemsets[k - 1]
        candidates = {}

        # Generate candidates from previous frequent itemsets
        prev_items = list(prev_frequent.keys())
        for i in range(len(prev_items)):
            for j in range(i + 1, len(prev_items)):
                items1 = set(prev_items[i])
                items2 = set(prev_items[j])
                union = items1.union(items2)
                if len(union) == k:
                    candidates[frozenset(union)] = 0

        # Count occurrences of candidates
        for transaction in self.transactions:
            transaction_set = set(transaction)
            for candidate in candidates:
                if candidate.issubset(transaction_set):
                    candidates[candidate] += 1

        # Filter by minimum support
        frequent_k = {
            itemset: count / self.n_transactions
            for itemset, count in candidates.items()
            if count / self.n_transactions >= self.min_support
        }

        return frequent_k

    def _generate_rules(self):
        """
        Generate association rules from frequent itemsets
        """
        self.rules = []

        # Generate rules from each frequent itemset of size >= 2
        for k in range(2, len(self.frequent_itemsets) + 1):
            for itemset, support in self.frequent_itemsets[k].items():
                # Generate all possible antecedent combinations
                for i in range(1, k):
                    for antecedent in combinations(itemset, i):
                        antecedent = frozenset(antecedent)
                        consequent = frozenset(itemset - antecedent)

                        # Calculate confidence
                        if antecedent in self.frequent_itemsets[len(antecedent)]:
                            conf = (
                                support
                                / self.frequent_itemsets[len(antecedent)][antecedent]
                            )

                            if conf >= self.min_confidence:
                                lift = self._calculate_lift(
                                    antecedent, consequent, support
                                )
                                self.rules.append(
                                    {
                                        "antecedent": list(antecedent),
                                        "consequent": list(consequent),
                                        "support": support,
                                        "confidence": conf,
                                        "lift": lift,
                                    }
                                )

    def _calculate_lift(self, antecedent, consequent, rule_support):
        """
        Calculate lift for a rule
        """
        antecedent_support = self.frequent_itemsets[len(antecedent)][antecedent]
        consequent_support = self.frequent_itemsets[len(consequent)][consequent]
        return rule_support / (antecedent_support * consequent_support)

    def get_rules(self, sort_by="lift", ascending=False):
        """
        Get sorted association rules
        """
        rules_df = pd.DataFrame(self.rules)
        return rules_df.sort_values(by=sort_by, ascending=ascending)


def main():
    # Read the patterns file
    input_file = "outputs/BINANCE_BTCUSDT_D1_with_patterns.csv"
    output_file = "outputs/BINANCE_BTCUSDT_D1_rules.csv"

    print(f"Reading patterns from {input_file}")
    df = pd.read_csv(input_file)

    # Convert patterns string to list
    transactions = df["patterns"].apply(eval).tolist()

    # Initialize and run Apriori
    print("\nRunning Apriori algorithm...")
    apriori = AprioriAnalyzer(min_support=0.01, min_confidence=0.5)
    apriori.fit(transactions)

    # Get and save rules
    print("Generating association rules...")
    rules_df = apriori.get_rules()

    # Save rules to CSV
    print(f"Saving rules to {output_file}")
    rules_df.to_csv(output_file, index=False)

    # Print summary
    print("\nMining results:")
    print(f"Total number of rules found: {len(rules_df)}")
    print("\nTop 5 rules by lift:")
    print(rules_df.head().to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()
