import json
import glob
import os
import numpy as np
from datetime import datetime

def get_color_and_label(probs):
    """Determine the winning class, color, and label based on node probabilities."""
    winner = int(np.argmax(probs))
    if winner == 0:
        return "#ff4757", "Bearish" # Red
    elif winner == 1:
        return "#747d8c", "Choppy"  # Gray
    else:
        return "#2ed573", "Bullish" # Green

def main():
    print("Generating Hive Mind Dashboard...")
    
    # 1. Load Master Ensemble Data
    try:
        with open("current_regime.json", "r") as f:
            master_data = json.load(f)
    except FileNotFoundError:
        print("Error: current_regime.json not found.")
        return

    # 2. Load Individual Node Data
    json_files = glob.glob("all_nodes/**/*.json", recursive=True)
    if not json_files:
        # Fallback if running locally directly in the output folder
        json_files = glob.glob("output/*.json")
        
    models = ["gru", "xgboost", "lightgbm", "catboost", "random_forest"]
    lookbacks = [5, 14, 21, 63]
    
    # Organize data into a grid [model][lookback]
    matrix = {m: {l: None for l in lookbacks} for m in models}
    
    for jf in json_files:
        with open(jf, "r") as f:
            data = json.load(f)
            m = data.get("model")
            l = data.get("lookback")
            if m in matrix and l in matrix[m]:
                matrix[m][l] = data

    # 3. Master Dashboard Metrics
    master_regime = master_data.get("master_actionable_regime", "Unknown")
    master_conf = master_data.get("master_confidence", 0) * 100
    update_time = master_data.get("updated_at", "Unknown")
    
    master_color = "#747d8c"
    if "Bearish" in master_regime: master_color = "#ff4757"
    if "Bull" in master_regime: master_color = "#2ed573"

    warning_banner = ""
    if master_conf < 45:
        warning_banner = f"""
        <div style="background-color: #ffa502; color: #000; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; text-align: center;">
            ⚠️ WARNING: Low Master Confidence ({master_conf:.1f}%). High node divergence detected. Enact Risk-Off protocols.
        </div>
        """

    # 4. Generate HTML Grid
    grid_html = ""
    for m in models:
        grid_html += f"<tr><td style='font-weight: bold; text-transform: uppercase;'>{m.replace('_', ' ')}</td>"
        for l in lookbacks:
            node = matrix[m][l]
            if node:
                color, label = get_color_and_label(node["today_probs"])
                acc = node.get("val_accuracy", 0) * 100
                grid_html += f"""
                <td style='background-color: {color}; color: #fff; text-align: center; padding: 15px; border-radius: 5px; border: 2px solid #2f3542;'>
                    <div style='font-size: 1.1em; font-weight: bold;'>{label}</div>
                    <div style='font-size: 0.8em; margin-top: 5px;'>Val Acc: {acc:.1f}%</div>
                </td>
                """
            else:
                grid_html += "<td style='background-color: #2f3542; color: #fff; text-align: center;'>OFFLINE</td>"
        grid_html += "</tr>"

    # 5. Assemble HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nifty 50 Hive Mind Matrix</title>
        <style>
            body {{ background-color: #1e272e; color: #d2dae2; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; }}
            .header {{ text-align: center; padding-bottom: 20px; border-bottom: 1px solid #485460; margin-bottom: 20px; }}
            .master-card {{ background-color: #2f3542; padding: 20px; border-radius: 10px; margin-bottom: 30px; text-align: center; border-left: 8px solid {master_color}; }}
            table {{ width: 100%; border-collapse: separate; border-spacing: 8px; margin-top: 20px; }}
            th {{ background-color: #485460; padding: 12px; color: #fff; border-radius: 5px; }}
            td {{ padding: 10px; }}
            .footer {{ text-align: center; margin-top: 40px; font-size: 0.9em; color: #808e9b; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 20-Node Hive Mind Matrix</h1>
                <p>Nifty 50 Algorithmic Regime Classifier | Last Updated: {update_time}</p>
            </div>
            
            {warning_banner}

            <div class="master-card">
                <h2>Master Ensemble Consensus</h2>
                <h1 style="color: {master_color}; margin: 10px 0;">{master_regime.replace('_', ' ')}</h1>
                <h3>Confidence: {master_conf:.1f}%</h3>
            </div>

            <h3 style="text-align: center; margin-top: 40px;">Node Visualization Grid</h3>
            <table>
                <tr>
                    <th>Model Family</th>
                    <th>5-Day (Micro)</th>
                    <th>14-Day (Short)</th>
                    <th>21-Day (Medium)</th>
                    <th>63-Day (Macro)</th>
                </tr>
                {grid_html}
            </table>

            <div class="footer">
                Automated by GitHub Actions | Deep Learning + XGBoost + LightGBM + CatBoost + Random Forest
            </div>
        </div>
    </body>
    </html>
    """

    with open("index.html", "w") as f:
        f.write(html_content)
    
    print("Dashboard generated successfully: index.html")

if __name__ == "__main__":
    main()
