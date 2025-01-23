# Generate Release Notes from GitHub Pull Requests

## Overview

`generate_release_notes.py` is a Python script designed to automate the generation of categorized release notes for the PyTorch repository. It processes a list of GitHub Pull Requests (PRs), categorizes them into predefined sections, and produces formatted Markdown files suitable for release documentation. Additionally, it logs the AI model's responses and provides a summary of processed and unprocessed PRs.

## Features

- **Tag Handling**: Recognizes and preserves optional tags in PR titles (e.g., `[inductor][AOTI]`).
- **GitHub Integration**: Fetches detailed information and comments for each PR using the GitHub API.
- **AI-Powered Categorization**: Utilizes the Ollama model to categorize PRs into sections like Improvements, Bug Fixes, etc.
- **Markdown Generation**: Produces well-structured `release.md` and `release_url.md` files with or without PR URLs.
- **Logging**: Records AI model responses with timestamps for debugging and tracking.
- **Processing Summary**: Compares processed PRs against the input list and outputs a summary along with unprocessed PRs.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Example](#example)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Further Enhancements](#further-enhancements)
- [License](#license)

## Requirements

- Python 3.7 or higher
- `requests` library
- Ollama installed and running locally
- GitHub Personal Access Token

## Installation

1. **Clone the Repository**

   ```
   git clone https://github.com/FindHao/pytorch_release_notes_generator
   cd pytorch_release_notes_generator
   ```

2. **Install Dependencies**

   Ensure you have `pip` installed. Then, install the required Python packages:

   ```
   pip install requests
   ```

## Configuration

### 1. GitHub Personal Access Token

To interact with the GitHub API, you need a Personal Access Token (PAT) with appropriate permissions.

- **Generate a PAT**:

  - Go to [GitHub Settings](https://github.com/settings/tokens).
  - Click on **"Generate new token"**.
  - Select the necessary scopes (e.g., `repo`).
  - Generate and copy the token.

- **Set the Token as an Environment Variable**:

  It's recommended to store the token securely using environment variables.

  ```
  export GITHUB_TOKEN=ghp_your_personal_access_token
  ```

  **Note**: Replace `ghp_your_personal_access_token` with your actual token. Avoid hardcoding the token in scripts, especially in shared or version-controlled environments.

### 2. Ollama Model Setup

Ensure that the Ollama model you intend to use is running locally and accessible via the specified URL. Adjust the `MODEL_NAME` and `OLLAMA_URL` in the script if necessary.
```
ollama run deepseek-r1:14b
```

## Usage

The script can be executed via the command line with various options to specify input and output files.

### Command-Line Arguments

- `-i`, `--input`: **(Required)** Path to the input file containing the list of PRs.
- `-m`, `--output_md`: **(Optional)** Path to the output Markdown file for release notes. Defaults to `release.md`.
- `-u`, `--output_url_md`: **(Optional)** Path to the output Markdown file with PR URLs for release notes. Defaults to `release_url.md`.
- `-o`, `--output_unprocessed`: **(Optional)** Path to the output file for unprocessed PRs. Defaults to `unprocessed_prs.txt`.

### Running the Script

```bash
python generate_release_notes.py -i pr_list.txt -m release.md -u release_url.md -o unprocessed_prs.txt
```

**Example**:

```bash
python generate_release_notes.py -i pr_list.txt
```

If optional arguments are not specified, the script will generate `release.md`, `release_url.md`, and `unprocessed_prs.txt` in the current directory.

## Example

### Preparing the PR List

Create a text file named `pr_list.txt` with the following content:

```bash
[inductor][AOTI] Make requires_stride_order more unbacked-symint-aware (#137201)
[Flex Attention][AOTI] Paged Attention (#137164)
Paged Attention without tags (#137165)
```

### Running the Script

```bash
python generate_release_notes.py -i pr_list.txt -m release.md -u release_url.md -o unprocessed_prs.txt
```

### Expected Output

- **`release.md`**:

  ```markdown
  ## Improvements:
  - [inductor][AOTI] Adds broadcast support for key-value batch dimensions in FlexAttention to enhance flexibility and performance (#137164).
  - [inductor][AOTI] Makes requires_stride_order more unbacked-symint-aware to enhance functionality (#137201).
  
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
  ```

- **`release_url.md`**:

  ```markdown

  ## Improvements:
  - [inductor][AOTI] Adds broadcast support for key-value batch dimensions in FlexAttention to enhance flexibility and performance [#137164](https://github.com/pytorch/pytorch/pull/137164).
  - [inductor][AOTI] Makes requires_stride_order more unbacked-symint-aware to enhance functionality [#137201](https://github.com/pytorch/pytorch/pull/137201).
  
  ## Bug Fixes:
  - [inductor][AOTI] Fixes an edge case in remove_split_with_size_one to enhance stability [#135962](https://github.com/pytorch/pytorch/pull/135962).
  
  ## New_features:
  - [inductor][AOTI] Introduces a new backend for faster computation in Triton kernels [#135530](https://github.com/pytorch/pytorch/pull/135530).
  
  ## Deprecations:
  - [inductor][AOTI] Deprecates the old stride order configuration in favor of the new method [#136367](https://github.com/pytorch/pytorch/pull/136367).
  
  ## BC breaking:
  - [inductor][AOTI] Changes the layout constraint which requires users to update their code as follows: ...
  
  ## Performance:
  - [inductor][AOTI] Optimizes the kernel to reduce computation time by 20% [#135239](https://github.com/pytorch/pytorch/pull/135239).
  
  ## Documentation:
  - [inductor][AOTI] Updates the documentation to include new layout constraints [#135581](https://github.com/pytorch/pytorch/pull/135581).
  
  ## Developers:
  - [inductor][AOTI] Refactors the cache management system to improve extensibility [#138239](https://github.com/pytorch/pytorch/pull/138239).
  ```

- **`unprocessed_prs.txt`**:

  ```
  Paged Attention without tags (#137165)
  ```

- **`ollama_responses.log`**:

  ```markdown
  ### 2025-01-22 10:15:30
  ## Improvements:
  - [Improvements][inductor][AOTI] Adds broadcast support for key-value batch dimensions in FlexAttention to enhance flexibility and performance (#137164).
  - [Improvements][inductor][AOTI] Makes requires_stride_order more unbacked-symint-aware to enhance functionality (#137201).
  
  ## Bug Fixes:
  - [Bug Fixes][inductor][AOTI] Fixes an edge case in remove_split_with_size_one to enhance stability (#135962).
  
  ## New_features:
  - [New_features][inductor][AOTI] Introduces a new backend for faster computation in Triton kernels (#135530).
  
  ## Deprecations:
  - [Deprecations][inductor][AOTI] Deprecates the old stride order configuration in favor of the new method (#136367).
  
  ## BC breaking:
  - [BC breaking][inductor][AOTI] Changes the layout constraint which requires users to update their code as follows: ...
  
  ## Performance:
  - [Performance][inductor][AOTI] Optimizes the kernel to reduce computation time by 20% (#135239).
  
  ## Documentation:
  - [Documentation][inductor][AOTI] Updates the documentation to include new layout constraints (#135581).
  
  ## Developers:
  - [Developers][inductor][AOTI] Refactors the cache management system to improve extensibility (#138239).
  ```

### Processing Summary

After running the script, the console will display a summary similar to:

```bash
Found 3 PRs in the input list.
Processing 1 batches of up to 5 PRs each.

Processing batch 1/1...
Batch 1 processed and results written to files.

All batches processed.
Extracted 2 PR numbers from 'release.md'.

### Processing Summary:
Total PRs in input: 3
PRs processed (included in release.md): 2
PRs not processed (not included in release.md): 1
Unprocessed PRs have been written to 'unprocessed_prs.txt'
```

## Output Files

- **`release.md`**: Contains categorized release notes with PR summaries and numbers.
- **`release_url.md`**: Similar to `release.md` but includes hyperlinks to the respective PRs.
- **`unprocessed_prs.txt`**: Lists PRs from the input that were not included in the release notes.
- **`ollama_responses.log`**: Logs the AI model's responses with timestamps for each batch processed.

## Troubleshooting

- **GitHub API Rate Limits**:

  If you encounter rate limit issues, the script will automatically pause and wait until the rate limit resets. Ensure your PAT has the necessary scopes and consider increasing the wait time if needed.

- **Ollama Model Issues**:

  - Ensure the Ollama model is running and accessible at the specified `OLLAMA_URL`.
  - Verify that the `MODEL_NAME` matches the name of your deployed model.
  - Check `ollama_responses.log` for detailed error messages.

- **Invalid PR List Format**:

  The script expects each line in the input file to follow one of the two formats:

  ```bash
  [Tag1][Tag2] PR Title (#PRNumber)
  PR Title without tags (#PRNumber)
  ```

  Ensure your `pr_list.txt` adheres to this format. Lines that do not match will be skipped with a warning.

## Further Enhancements

- **Batch Size Optimization**: Adjust the `BATCH_SIZE` based on the performance and capabilities of your Ollama model.
- **Enhanced Error Handling**: Implement retries for failed API requests or model responses.
- **Parallel Processing**: Utilize multi-threading or asynchronous requests to speed up PR detail fetching, keeping in mind GitHub's rate limits.
- **Automated Testing**: Develop test cases to validate script functionality under various scenarios.
- **Dynamic Logging**: Implement more granular logging, such as logging per PR or per category.

## License

MIT License