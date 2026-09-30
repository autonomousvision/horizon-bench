import re
import json


def extract_json(llm_output):
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def json_to_markdown(json_obj):
    # make every key a title
    md_lines = []
    for key, value in json_obj.items():
        md_lines.append(f"### {key}\n")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                md_lines.append(f"- **{sub_key}**: {sub_value}\n")
        elif isinstance(value, list):
            for item in value:
                md_lines.append(f"- {item}\n")
        else:
            md_lines.append(f"{value}\n")
    return "\n".join(md_lines)