#!/usr/bin/env python3
"""
KUBE-AI Accuracy Visualization
Generates accuracy plots from evaluation results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_evaluation(results_path='../results/evaluation_results.json'):
    """Generate accuracy plots from evaluation JSON."""
    if not os.path.exists(results_path):
        print(f"No results found at {results_path}. Run evaluate_model.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    accuracy = results['overall_accuracy']
    per_class = results['per_class']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('KUBE-AI Model Accuracy Evaluation', fontsize=16, fontweight='bold')

    # 1. Overall accuracy pie
    ax1.pie([accuracy, 1 - accuracy], labels=['Correct', 'Incorrect'],
            colors=['#2ECC71', '#E74C3C'], autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Overall Accuracy: {accuracy:.1%}')

    # 2. Per-class F1
    names = list(per_class.keys())
    f1s = [per_class[n]['f1'] for n in names]
    bars = ax2.bar(names, f1s, color='#3498DB')
    ax2.set_title('F1-Score by Animal')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    for b, v in zip(bars, f1s):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)

    # 3. Precision vs Recall
    precs = [per_class[n]['precision'] for n in names]
    recs = [per_class[n]['recall'] for n in names]
    x = np.arange(len(names))
    ax3.bar(x - 0.2, precs, 0.4, label='Precision', color='#E67E22')
    ax3.bar(x + 0.2, recs, 0.4, label='Recall', color='#9B59B6')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45)
    ax3.set_title('Precision & Recall')
    ax3.set_ylim(0, 1)
    ax3.legend()

    # 4. Summary text
    ax4.axis('off')
    txt = f"KUBE-AI Accuracy Report\n\nOverall Accuracy: {accuracy:.1%}\nTotal Samples: {results['total_samples']}\n\n"
    for n, m in per_class.items():
        txt += f"{n:12s}  P={m['precision']:.2f}  R={m['recall']:.2f}  F1={m['f1']:.2f}  n={m['support']}\n"
    ax4.text(0.05, 0.95, txt, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    os.makedirs('../visualizations', exist_ok=True)
    out = '../visualizations/accuracy_evaluation.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


if __name__ == '__main__':
    plot_evaluation()
