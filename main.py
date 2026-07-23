import os
import pandas as pd
from pipeline import DynamicVariantPipeline

def main():
    panels = {
        'MASTER': 'YARISMA_TRAIN_MASTER.csv',
        'KANSER': 'YARISMA_TRAIN_KANSER.csv',
        'PAH': 'YARISMA_TRAIN_PAH.csv',
        'CFTR': 'YARISMA_TRAIN_CFTR.csv'
    }

    all_results = []
    
    for panel_name, file_name in panels.items():
        if not os.path.exists(file_name):
            print(f"Warning: {file_name} not found. Skipping {panel_name} panel.")
            continue
            
        try:
            pipeline = DynamicVariantPipeline(file_name, panel_name)
            results = pipeline.execute()
            if results:
                if isinstance(results, list):
                    all_results.extend(results)
                else:
                    all_results.append(results)
        except Exception as e:
            print(f"An error occurred while processing {panel_name} panel: {e}")
            import traceback
            traceback.print_exc()
    
    # Adım 9.2 — Panel Arası Konsolide Metrik Tablosu
    if all_results:
        print(f"\n{'='*80}")
        print("CONSOLIDATED METRICS TABLE (Adım 9.2)")
        print(f"{'='*80}")
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        
        os.makedirs("outputs", exist_ok=True)
        results_df.to_csv("outputs/consolidated_metrics.csv", index=False)
        print(f"\nConsolidated table saved to outputs/consolidated_metrics.csv")

if __name__ == "__main__":
    main()
