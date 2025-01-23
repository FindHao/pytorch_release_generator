import json
import requests
import re
import argparse
import os
import time
from typing import List, Dict, Set
from datetime import datetime

# Configuration for GitHub and Ollama
# It's recommended to manage sensitive information using environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)
if GITHUB_TOKEN is None:
    raise ValueError("GITHUB_TOKEN is not set")
REPO_OWNER = "pytorch"
REPO_NAME = "pytorch"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:14b"  # Updated to your new model name
BATCH_SIZE = 5  # Number of PRs to process per batch

# GitHub API request headers
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}


def read_pr_list(file_path: str) -> List[Dict[str, str]]:
    """
    Reads the PR list file, extracting PR number, original title, and optional tags.
    Example file format:
    [Flex Attention][AOTI] Paged Attention (#137164)
    [inductor][AOTI] Make requires_stride_order more unbacked-symint-aware (#137201)
    Paged Attention without tags (#137165)
    """
    pr_entries = []
    # Regex to match titles with one or more tags
    pr_pattern_with_tags = re.compile(r"^(?:\[(.*?)\])+.*\(#(\d+)\)")
    # Regex to match titles without tags
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
    Fetches detailed information for a PR, including title and description.
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
    Fetches comments for a PR, excluding bot comments.
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
    Checks and handles GitHub API rate limiting.
    """
    if 'X-RateLimit-Remaining' in response.headers:
        remaining = int(response.headers['X-RateLimit-Remaining'])
        if remaining < 10:
            reset_time = int(response.headers.get(
                'X-RateLimit-Reset', time.time() + 60))
            sleep_time = max(reset_time - int(time.time()),
                             0) + 5  # Add a 5-second buffer
            reset_time_str = time.strftime(
                '%Y-%m-%d %H:%M:%S', time.localtime(reset_time))
            print(f"Approaching rate limit. Sleeping for {
                  sleep_time} seconds until {reset_time_str}...")
            time.sleep(sleep_time)


def prepare_batches(pr_data_list: List[Dict], batch_size: int) -> List[List[Dict]]:
    """
    Splits the PR data list into multiple batches.
    """
    return [pr_data_list[i:i + batch_size] for i in range(0, len(pr_data_list), batch_size)]


def prepare_prompt(batch: List[Dict]) -> str:
    """
    Prepares the prompt to send to the Ollama model, including categorization instructions, definitions, and PR data.
    """
    category_definitions = """
### Category Definitions:

- **BC breaking**: All commits that are BC-breaking. These are the most important commits. If any pre-sorted commit is actually BC-breaking, move it to this section. Each commit should contain a paragraph explaining the rationale behind the change as well as an example for how to update user code BC-Guidelines.
- **Deprecations**: All commits introducing deprecation. Each commit should include a small example explaining what should be done to update user code.
- **New_features**: All commits introducing a new feature (new functions, new submodule, new supported platform, etc.).
- **Improvements**: All commits providing improvements to existing features should be here (new backend for a function, new argument, better numerical stability).
- **Bug Fixes**: All commits that fix bugs and behaviors that do not match the documentation.
- **Performance**: All commits that are added mainly for performance (we separate this from improvements above to make it easier for users to look for it).
- **Documentation**: All commits that add/update documentation.
- **Developers**: All commits that are not end-user facing but still impact people that compile from source, develop into PyTorch, extend PyTorch, etc.
"""

    example_response = """
### Example Output:

## Improvements:
- [inductor][AOTI] Adds broadcast support for key-value batch dimensions in FlexAttention to enhance flexibility and performance (#135505).
- [inductor][AOTI] Flip custom_op_default_layout_constraint in Inductor to optimize tensor layout for improved computation efficiency (#135239).

## Bug Fixes:
- [inductor][AOTI] Fixes an edge case in remove_split_with_size_one to enhance stability (#135962).

## New_features:
- [inductor][AOTI] Introduces a new backend for faster computation in Triton kernels (#135530).

## Deprecations:
- [inductor][AOTI] Deprecates the old stride order configuration in favor of the new method (#136367).

## BC breaking:
- [inductor][AOTI] Changes the layout constraint which requires users to update their code as follows: ...

## Performance:
- [inductor][AOTI] Optimizes the kernel to reduce computation time by 20% (#135239).

## Documentation:
- [inductor][AOTI] Updates the documentation to include new layout constraints (#135581).

## Developers:
- [inductor][AOTI] Refactors the cache management system to improve extensibility (#138239).
"""

    instructions = (
        "You are a release notes generator for the PyTorch repository. "
        "Your task is to categorize a list of Pull Requests (PRs) into the following categories based on the definitions provided below:\n\n"
        "{category_definitions}\n\n"
        "Each PR should be summarized in one sentence and placed under the appropriate category. "
        "Use the format '- [Tags] one sentence summary of the PR (#PR_Number)'. "
        "Ensure that the output is in valid Markdown format and that the PR number is placed at the end of each entry.\n\n"
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
    Sends the prompt to the local Ollama model and retrieves the complete response.
    Handles streaming responses by collecting all response parts until done is True.
    """
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "options": {
            "max_length": 2000,  # Adjust as needed
            "stream": True  # Enable streaming responses
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


def extract_tags_from_summary(summary: str) -> Set[str]:
    """
    Extracts tags from the PR summary.
    """
    tag_pattern = re.compile(r"\[(.*?)\]")
    return set(tag.lower() for tag in tag_pattern.findall(summary))


def parse_ollama_response(response_text: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Parses the response from the Ollama model, categorizing PRs accordingly.
    Returns a dictionary where each key is a category and the value is a list of PR summaries and numbers.
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

    # Use Markdown structure to parse categories and corresponding PRs
    current_category = None
    # Updated regex to make [Tags] optional
    pr_pattern = re.compile(r"-(?:\[(.*?)\])? (.*?) \(#(\d+)\)")

    for line in response_text.splitlines():
        line = line.strip()
        if line.startswith("## "):
            # Get the category name
            category_title = line[3:].rstrip(":").strip()
            current_category = category_mapping.get(category_title, None)
            if not current_category:
                print(f"Unknown category '{category_title}'. Skipping...")
        elif line.startswith("- "):
            match = pr_pattern.match(line)
            if match and current_category:
                tags = match.group(1) if match.group(1) else ""
                summary = match.group(2)
                pr_number = match.group(3)
                pr_entry = {
                    "summary": f"[{tags}] {summary}" if tags else f"{summary}",
                    "pr_number": pr_number
                }
                categories[current_category].append(pr_entry)
            else:
                print(f"PR '{
                      line}' does not match the expected format or no current category. Skipping...")
        else:
            continue

    return categories


def aggregate_markdown(existing: Dict[str, List[Dict[str, str]]], new: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Adds newly categorized PRs to the existing categories.
    """
    for key in existing.keys():
        existing[key].extend(new.get(key, []))
    return existing


def generate_markdown(categories: Dict[str, List[Dict[str, str]]], include_urls: bool = False) -> str:
    """
    Generates Markdown content based on the categorized PRs.
    If include_urls is True, replaces PR numbers with URLs.
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
    Saves the Markdown content to a file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    print(f"Release notes saved to {file_path}")


def log_ollama_response(response_text: str):
    """
    Logs the Ollama response to a log file with a timestamp.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"\n### {timestamp}\n{response_text}\n"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    print(f"Ollama response logged to {LOG_FILE}")


def extract_pr_numbers_from_release(release_file: str) -> Set[str]:
    """
    Extracts all PR numbers from the release.md file.
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
    Compares the input PR list with the PR numbers in the release notes,
    outputs a summary, and writes unprocessed PRs to a separate file.
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
        # Get detailed information of unprocessed PRs
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
    # Read the PR list
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

    # Batch processing
    batches = prepare_batches(pr_entries, BATCH_SIZE)
    print(f"Processing {len(batches)} batches of up to {BATCH_SIZE} PRs each.")

    for i, batch_pr_entries in enumerate(batches, start=1):
        print(f"\nProcessing batch {i}/{len(batches)}...")

        # Fetch details for the current batch
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

        # Prepare the prompt
        prompt = prepare_prompt(pr_data_list)

        # Send to Ollama
        ollama_response = send_to_ollama(prompt)
        if not ollama_response:
            print(f"No response from Ollama for batch {i}. Skipping...")
            continue

        # Log the Ollama response
        log_ollama_response(ollama_response)

        # Parse the response
        categorized = parse_ollama_response(ollama_response)

        # Add original tags to the categorized results without duplicating
        for category, prs in categorized.items():
            for pr in prs:
                pr_number = pr["pr_number"]
                # Find the corresponding PR details
                pr_detail = next(
                    (p for p in pr_data_list if p["number"] == pr_number), None)
                if pr_detail:
                    original_tags = set(tag.lower()
                                        for tag in pr_detail["tags"])
                    if original_tags:
                        # Extract existing tags from the summary
                        existing_tags = extract_tags_from_summary(
                            pr["summary"])
                        # Determine which original tags are missing
                        missing_tags = original_tags - existing_tags
                        if missing_tags:
                            # Format missing tags as [tag1][tag2]...
                            formatted_missing_tags = "".join(
                                [f"[{tag}]" for tag in missing_tags])
                            pr["summary"] = f"{
                                formatted_missing_tags} {pr['summary']}"
                else:
                    print(f"Tags not found for PR #{pr_number}")

        # Aggregate categorized results
        categories = aggregate_markdown(categories, categorized)

        # Generate release.md and release_url.md
        markdown_output = generate_markdown(categories, include_urls=False)
        markdown_url_output = generate_markdown(categories, include_urls=True)

        # Save to files
        save_output(markdown_output, output_file_md)
        save_output(markdown_url_output, output_file_url_md)

        print(f"Batch {i} processed and results written to files.")

        # Prevent too rapid requests; adjust as needed
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
