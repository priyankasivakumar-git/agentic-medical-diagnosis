import os
import json
import glob

os.makedirs("outputs/paper_assets", exist_ok=True)

logs_dir = "outputs/logs"
log_files = sorted(glob.glob(os.path.join(logs_dir, "*.json")), key=os.path.getmtime, reverse=True)

if not log_files:
    print("No log files found in outputs/logs/. Run main.py or the Streamlit app first.")
    exit()

chosen_log = log_files[0]   # newest log by default; change index to pick a different one
print(f"Formatting: {chosen_log}")

with open(chosen_log) as f:
    data = json.load(f)


def truncate_long_values(obj, max_items=6):
    """Recursively shortens any long list/array so the appendix stays readable."""
    if isinstance(obj, dict):
        return {k: truncate_long_values(v, max_items) for k, v in obj.items()}
    elif isinstance(obj, list):
        if len(obj) > max_items:
            shown = obj[:max_items]
            return shown + [f"... ({len(obj) - max_items} more values truncated) ..."]
        return [truncate_long_values(v, max_items) for v in obj]
    else:
        return obj


clean_data = truncate_long_values(data)

output_path = "outputs/paper_assets/appendix_execution_log.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("# Appendix: Sample Multi-Agent Execution Log\n\n")
    f.write(f"*Source file:* {os.path.basename(chosen_log)}  \n")
    f.write(f"*Timestamp:* {clean_data.get('timestamp', 'N/A')}\n\n")
    f.write("This log records the full sequence of tool calls made by the Master Agent "
            "during one complete diagnostic run, demonstrating the system's audit trail "
            "and orchestration order. Long numeric feature arrays have been truncated for "
            "readability.\n\n")
    f.write("json\n")
    f.write(json.dumps(clean_data, indent=2, default=str))
    f.write("\n\n")

print(f"Saved formatted appendix log to: {output_path}")
print(f"You can copy the contents of this .md file directly into your paper's appendix section, "
      f"or convert it to PDF/screenshot as needed.")