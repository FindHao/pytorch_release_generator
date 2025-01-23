import json
import requests
import re
import argparse
import os
import time
from typing import List, Dict, Set
from datetime import datetime

# Configure GitHub and Ollama related information
# Recommended to manage sensitive information using environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)
if GITHUB_TOKEN is None:
    raise ValueError("GITHUB_TOKEN is not set")
REPO_OWNER = "pytorch"
REPO_NAME = "pytorch"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:14b"  # Replace with your model name
BATCH_SIZE = 5  # Number of PRs to process per batch
LOG_FILE = "ollama_responses.log"  # Log file path
UNPROCESSED_PR_FILE = "unprocessed_prs.txt"  # Output file for unprocessed PRs

# GitHub API request headers
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}


def read_pr_list(file_path: str) -> List[Dict[str, str]]:
    """
    Read PR list file, extract PR number, original title and optional tags.
    File format example:
    [Flex Attention][AOTI] Paged Attention (#137164)
     Make requires_stride_order more unbacked-symint-aware (#137201)
    Paged Attention without tags (#137165)
    """
    pr_entries = []
    # This regular expression will match titles with one or more tags
    pr_pattern_with_tags = re.compile(r"^(?:\[(.*?)\])+.*\(#(\d+)\)")
    # This regular expression will match titles without tags
    pr_pattern_without_tags = re.compile(r"^(?!\[(.*?)\]).*\(#(\d+)\)")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            match = pr_pattern_with_tags.match(line)
            if match:
                # Extract all tags
                tags = re.findall(r"\[(.*?)\]", line)
                pr_number = match.group(2)
                pr_entries.append({
                    "number": pr_number,
                    "original_title": line,
                    "tags": tags
                })
            else:
                match = pr_pattern_without_tags.match(line)
                if match:
                    pr_number = match.group(2)
                    pr_entries.append({
                        "number": pr_number,
                        "original_title": line,
                        "tags": []  # No tags
                    })
                else:
                    print(
                        f"Line '{line}' does not match expected format. Skipping...")
    return pr_entries


def fetch_pr_details(pr_number: str) -> Dict:
    """
    Get PR details, including title and description.
    """
    pr_api_url = f"https://api.github.com/repos/{
        REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}"
    try:
        response = requests.get(pr_api_url, headers=HEADERS)
        handle_rate_limit(response)
        if response.status_code == 200:
            pr_data = response.json()
            pr_title = pr_data.get("title", "N/A")
            pr_body = pr_data.get("body", "")
            return {
                "number": pr_number,
                "title": pr_title,
                "body": pr_body
            }
        else:
            print(f"Failed to fetch PR #{
                  pr_number} details. HTTP Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred while fetching PR #{
              pr_number} details: {str(e)}")
        return None


def fetch_pr_comments(pr_number: str) -> List[Dict]:
    """
    Get PR comments, excluding robot comments.
    """
    comments_api_url = f"https://api.github.com/repos/{
        REPO_OWNER}/{REPO_NAME}/issues/{pr_number}/comments"
    try:
        response = requests.get(comments_api_url, headers=HEADERS)
        handle_rate_limit(response)
        if response.status_code == 200:
            comments = response.json()
            user_comments = []
            for comment in comments:
                commenter = comment.get("user", {}).get("login", "N/A")
                if commenter not in ["github-actions[bot]", "pytorch-bot[bot]", "pytorchmergebot"]:
                    comment_body = comment.get("body", "")
                    user_comments.append({
                        "user": commenter,
                        "body": comment_body
                    })
            return user_comments
        else:
            print(f"Failed to fetch comments for PR #{
                  pr_number}. HTTP Status Code: {response.status_code}")
            return []
    except Exception as e:
        print(f"An error occurred while fetching comments for PR #{
              pr_number}: {str(e)}")
        return []


def handle_rate_limit(response: requests.Response):
    """
    Check and handle GitHub API rate limit.
    """
    if 'X-RateLimit-Remaining' in response.headers:
        remaining = int(response.headers['X-RateLimit-Remaining'])
        if remaining < 10:
            reset_time = int(response.headers.get(
                'X-RateLimit-Reset', time.time() + 60))
            sleep_time = max(reset_time - int(time.time()),
                             0) + 5  # Add 5 seconds buffer
            reset_time_str = time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime(reset_time))
            print(f"Approaching rate limit. Sleeping for {
                  sleep_time} seconds until {reset_time_str}...")
            time.sleep(sleep_time)


def prepare_batches(pr_data_list: List[Dict], batch_size: int) -> List[List[Dict]]:
    """
    Split PR data list into multiple batches.
    """
    return [pr_data_list[i:i + batch_size] for i in range(0, len(pr_data_list), batch_size)]


def prepare_prompt(batch: List[Dict]) -> str:
    """
    Prepare prompt to send to Ollama model, including classification instructions, classification definitions and PR data.
    """
    category_definitions = """
### Category Definitions:

- **BC breaking**: All commits that are BC-breaking. These are the most important commits. If any pre-sorted commit is actually BC-breaking, do move it to this section. Each commit should contain a paragraph explaining the rationale behind the change as well as an example for how to update user code BC-Guidelines.
- **Deprecations**: All commits introducing deprecation. Each commit should include a small example explaining what should be done to update user code.
- **New_features**: All commits introducing a new feature (new functions, new submodule, new supported platform etc).
- **Improvements**: All commits providing improvements to existing features should be here (new backend for a function, new argument, better numerical stability).
- **Bug Fixes**: All commits that fix bugs and behaviors that do not match the documentation.
- **Performance**: All commits that are added mainly for performance (we separate this from improvements above to make it easier for users to look for it).
- **Documentation**: All commits that add/update documentation.
- **Developers**: All commits that are not end-user facing but still impact people that compile from source, develop into PyTorch, extend PyTorch, etc.
"""

    example_response = """
### Example Output:

## Improvements:
- [Improvements] Adds broadcast support for key-value batch dimensions in FlexAttention to enhance flexibility and performance (#135505).
- [Improvements] Flip custom_op_default_layout_constraint in Inductor to optimize tensor layout for improved computation efficiency (#135239).

## Bug Fixes:
- [Bug Fixes] Fixes an edge case in remove_split_with_size_one to enhance stability (#135962).

## New_features:
- [New_features] Introduces a new backend for faster computation in Triton kernels (#135530).

## Deprecations:
- [Deprecations] Deprecates the old stride order configuration in favor of the new method (#136367).

## BC breaking:
- [BC breaking] Changes the layout constraint which requires users to update their code as follows: ...

## Performance:
- [Performance] Optimizes the kernel to reduce computation time by 20% (#135239).

## Documentation:
- [Documentation] Updates the documentation to include new layout constraints (#135581).

## Developers:
- [Developers] Refactors the cache management system to improve extensibility (#138239).
"""

    instructions = (
        "You are a release notes generator for the PyTorch repository. "
        "Your task is to categorize a list of Pull Requests (PRs) into the following categories based on the definitions provided below:\n\n"
        "{category_definitions}\n\n"
        "Each PR should be summarized in one sentence and placed under the appropriate category. "
        "Use the format '- [Category] one sentence summary of the PR (#PR_Number)'. "
        "Ensure that the output is in valid Markdown format.\n\n"
        "### Example Output:\n{example_response}\n\n"
        "Here is the list of PRs:\n"
    ).format(category_definitions=category_definitions, example_response=example_response)

    pr_entries = []
    for pr in batch:
        entry = f"- {pr['original_title']}\n"
        pr_entries.append(entry)

    prompt = instructions + "".join(pr_entries)
    return prompt


def send_to_ollama(prompt: str) -> str:
    """
    Send prompt to local Ollama model and get complete response.
    Handle stream response, gradually collect all response parts until done is True.
    """
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "options": {
            "max_length": 2000,  # Adjust as needed
            "stream": True  # Enable stream response
        }
    }
    try:
        with requests.post(OLLAMA_URL, json=data, stream=True) as response:
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_line = json.loads(line.decode('utf-8'))
                            if 'response' in json_line:
                                full_response += json_line['response']
                            if json_line.get('done', False):
                                break
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON line: {e}")
                return full_response.strip()
            else:
                print(f"Error from Ollama API: {response.status_code}")
                print(response.text)
                return ""
    except Exception as e:
        print(f"An error occurred while communicating with Ollama: {str(e)}")
        return ""


def parse_ollama_response(response_text: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Parse Ollama model response, categorize PRs.
    Each category contains PR summary and number.
    """
    categories = {
        "bc_breaking": [],
        "deprecations": [],
        "new_features": [],
        "improvements": [],
        "bug_fixes": [],
        "performance": [],
        "documentation": [],
        "developers": []
    }

    category_mapping = {
        "BC breaking": "bc_breaking",
        "Deprecations": "deprecations",
        "New_features": "new_features",
        "Improvements": "improvements",
        "Bug Fixes": "bug_fixes",
        "Performance": "performance",
        "Documentation": "documentation",
        "Developers": "developers"
    }

    # Regular expression matching: - [Category][Tags] Summary (#PRNumber)
    pr_pattern = re.compile(r"- \[(.*?)\](?:\[(.*?)\])* (.*?) \(#(\d+)\)")

    for line in response_text.splitlines():
        line = line.strip()
        match = pr_pattern.match(line)
        if match:
            category_tag = match.group(1)
            tags = match.group(2) if match.group(2) else ""
            summary = match.group(3)
            pr_number = match.group(4)
            category_key = category_mapping.get(category_tag, None)
            if category_key:
                # Combine tags
                combined_tags = ""
                if tags:
                    combined_tags = f"[{tags}]"
                if category_tag:
                    combined_tags = f"[{category_tag}]{combined_tags}"
                # Store summary and PR number
                pr_entry = {
                    "summary": f"{combined_tags} {summary}",
                    "pr_number": pr_number
                }
                categories[category_key].append(pr_entry)
            else:
                print(f"Unknown category tag '{
                      category_tag}' in PR '{line}'. Skipping...")
        else:
            if line.startswith("## "):
                # Optional: Handle unexpected category title
                continue
            elif line.startswith("- "):
                print(
                    f"PR '{line}' does not match the expected format. Skipping...")
            else:
                # Ignore other lines
                continue

    return categories


def aggregate_markdown(existing: Dict[str, List[Dict[str, str]]], new: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Add new categorized PRs to existing categories.
    """
    for key in existing.keys():
        existing[key].extend(new.get(key, []))
    return existing


def generate_markdown(categories: Dict[str, List[Dict[str, str]]], include_urls: bool = False) -> str:
    """
    Generate Markdown content based on categorization results.
    If include_urls is True, replace PR numbers with URLs.
    """
    markdown = ""

    for category in ["bc_breaking", "deprecations", "new_features", "improvements", "bug_fixes", "performance", "documentation", "developers"]:
        entries = categories.get(category, [])
        if entries:
            # Convert category key to title
            category_title = {
                "bc_breaking": "BC breaking",
                "deprecations": "Deprecations",
                "new_features": "New_features",
                "improvements": "Improvements",
                "bug_fixes": "Bug Fixes",
                "performance": "Performance",
                "documentation": "Documentation",
                "developers": "Developers"
            }.get(category, category.capitalize())

            markdown += f"## {category_title}:\n"
            for pr in entries:
                summary = pr["summary"]
                pr_number = pr["pr_number"]
                if include_urls:
                    pr_url = f"https://github.com/{REPO_OWNER}/{
                        REPO_NAME}/pull/{pr_number}"
                    markdown += f"- {summary} [#{pr_number}]({pr_url}).\n"
                else:
                    markdown += f"- {summary} (#{pr_number}).\n"
            markdown += "\n"

    return markdown.strip()


def save_output(markdown: str, file_path: str):
    """
    Save Markdown content to file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    print(f"Release notes saved to {file_path}")


def log_ollama_response(response_text: str):
    """
    Log Ollama response to log file, append content and add timestamp.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"\n### {timestamp}\n{response_text}\n"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    print(f"Ollama response logged to {LOG_FILE}")


def extract_pr_numbers_from_release(release_file: str) -> Set[str]:
    """
    Extract all PR numbers from release.md file.
    """
    pr_numbers = set()
    pr_number_pattern = re.compile(r"\(#(\d+)\)")

    try:
        with open(release_file, 'r', encoding='utf-8') as f:
            for line in f:
                matches = pr_number_pattern.findall(line)
                for match in matches:
                    pr_numbers.add(match)
    except FileNotFoundError:
        print(f"Release file '{release_file}' not found.")

    return pr_numbers


def summarize_processing(input_prs: List[Dict[str, str]], release_pr_numbers: Set[str], output_unprocessed_file: str):
    """
    Compare input PR list and PR numbers in release.md, output summary and write unprocessed PR info to file.
    """
    input_pr_numbers = set(pr["number"] for pr in input_prs)
    processed_pr_numbers = release_pr_numbers
    unprocessed_pr_numbers = input_pr_numbers - processed_pr_numbers

    total_prs = len(input_pr_numbers)
    processed = len(processed_pr_numbers & input_pr_numbers)
    unprocessed = len(unprocessed_pr_numbers)

    print("\n### Processing Summary:")
    print(f"Total PRs in input: {total_prs}")
    print(f"PRs processed (included in release.md): {processed}")
    print(f"PRs not processed (not included in release.md): {unprocessed}")

    if unprocessed > 0:
        # Get unprocessed PR details
        unprocessed_prs = [
            pr for pr in input_prs if pr["number"] in unprocessed_pr_numbers]
        with open(output_unprocessed_file, 'w', encoding='utf-8') as f:
            for pr in unprocessed_prs:
                f.write(f"{pr['original_title']}\n")
        print(f"Unprocessed PRs have been written to '{
              output_unprocessed_file}'")
    else:
        print("All PRs have been processed.")


def main(input_file: str, output_file_md: str, output_file_url_md: str, output_unprocessed_file: str):
    # Read PR list
    pr_entries = read_pr_list(input_file)
    print(f"Found {len(pr_entries)} PRs in the input list.")

    if not pr_entries:
        print("No PRs found in the input file.")
        return

    # Initialize category dictionary
    categories = {
        "bc_breaking": [],
        "deprecations": [],
        "new_features": [],
        "improvements": [],
        "bug_fixes": [],
        "performance": [],
        "documentation": [],
        "developers": []
    }

    # Process in batches
    batches = prepare_batches(pr_entries, BATCH_SIZE)
    print(f"Processing {len(batches)} batches of up to {BATCH_SIZE} PRs each.")

    for i, batch_pr_entries in enumerate(batches, start=1):
        print(f"\nProcessing batch {i}/{len(batches)}...")

        # Get current batch PR details
        pr_data_list = []
        for pr_entry in batch_pr_entries:
            pr_number = pr_entry["number"]
            pr_details = fetch_pr_details(pr_number)
            if not pr_details:
                continue
            comments = fetch_pr_comments(pr_number)
            pr_details["comments"] = comments
            pr_details["original_title"] = pr_entry["original_title"]
            pr_details["tags"] = pr_entry["tags"]
            pr_data_list.append(pr_details)

        if not pr_data_list:
            print(f"No PR data fetched for batch {i}. Skipping...")
            continue

        # Prepare prompt
        prompt = prepare_prompt(pr_data_list)

        # Send to Ollama
        ollama_response = send_to_ollama(prompt)
        if not ollama_response:
            print(f"No response from Ollama for batch {i}. Skipping...")
            continue

        # Log Ollama response to log file
        log_ollama_response(ollama_response)

        # Parse response
        categorized = parse_ollama_response(ollama_response)

        # Add tags to categorized results
        for category, prs in categorized.items():
            for pr in prs:
                pr_number = pr["pr_number"]
                # Find corresponding PR details
                pr_detail = next(
                    (p for p in pr_data_list if p["number"] == pr_number), None)
                if pr_detail:
                    tags = pr_detail["tags"]
                    if tags:
                        # Format tags as [tag1][tag2]...
                        formatted_tags = "".join([f"[{tag}]" for tag in tags])
                        pr["summary"] = f"{formatted_tags} {pr['summary']}"
                else:
                    print(f"Tags not found for PR #{pr_number}")

        # Aggregate categorized results
        categories = aggregate_markdown(categories, categorized)

        # Generate release.md and release_url.md
        markdown_output = generate_markdown(categories, include_urls=False)
        markdown_url_output = generate_markdown(categories, include_urls=True)

        # Save to file
        save_output(markdown_output, output_file_md)
        save_output(markdown_url_output, output_file_url_md)

        print(f"Batch {i} processed and results written to files.")

        # Prevent too fast requests, adjust as needed
        time.sleep(1)

    print("\nAll batches processed.")

    # Extract PR numbers from release.md
    release_pr_numbers = extract_pr_numbers_from_release(output_file_md)
    print(f"Extracted {len(release_pr_numbers)
                       } PR numbers from '{output_file_md}'.")

    # Compare and summarize
    summarize_processing(pr_entries, release_pr_numbers,
                         output_unprocessed_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate categorized release notes from GitHub PRs using Ollama.")
    parser.add_argument("-i", "--input", required=True,
                        help="Path to the input file containing the list of PRs.")
    parser.add_argument("-m", "--output_md", required=False, default="release.md",
                        help="Path to the output Markdown file for release notes.")
    parser.add_argument("-u", "--output_url_md", required=False, default="release_url.md",
                        help="Path to the output Markdown file with PR URLs for release notes.")
    parser.add_argument("-o", "--output_unprocessed", required=False,
                        default="unprocessed_prs.txt", help="Path to the output file for unprocessed PRs.")
    args = parser.parse_args()

    main(args.input, args.output_md, args.output_url_md, args.output_unprocessed)
