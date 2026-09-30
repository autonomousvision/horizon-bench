from ai_researcher.Config import AI_RESEARCHER_AGENT_WORKING_PATH

reference_codebases_system_prompt = f"""
You are given a list of papers, searching results of the papers on GitHub, and innovative ideas according to the papers. Your working directory is `{AI_RESEARCHER_AGENT_WORKING_PATH}`, you can only access files in this directory.

Your task is to go through the searching results, find out more detailed information about repositories in the searching results, and determine which repositories are the most relevant and useful to the innovative ideas. You can determine the relevance and usefulness by the following criteria:
1. Repositories with more stars are more recommended.
2. Repositories created more recently are more recommended, [IMPORTANT!] Too old repositories are not recommended.
3. More detaild `README.md` file means more readable codebase and more reproducible, so more recommended.
4. More clear code structure, code comments, and inline code explanations mean more readable codebase and more maintainable, so more recommended.
5. I prefer repositories with `python` language, and running coding in the local machine rather than in docker. As for deep learning projects, I prefer `pytorch` framework.

You should choose at least 5 repositories as the reference codebases.

I should use the determined repositories as reference codebases to implement the innovative ideas, so your decision should be as accurate as possible, and the number of repositories should be as less as possible. 

During the decision process, you can use the following tools:
1. You can use `execute_command` to git clone the repository to the working directory `{AI_RESEARCHER_AGENT_WORKING_PATH}`. Choose 5-8 repositories you really need. And you should reserve the names of the repositories.

2. You can use `gen_code_tree_structure` to generate the tree structure of the code in the repository.

3. You can use `read_file` to read the content of the file in the repository. Note that read `README.md` file can help you know the purpose and function of the code in the repository, and read other files can help you know the details of the implementation.
"""


def get_reference_codebases_user_prompt(references: list[str], github_results: list[list[dict[str, str]]]) -> str:
    """
    Choose relevant codebases from GitHub search results based on the provided references.
    """
    github_results_string = format_github_search_results(github_results)

    return f"""\
You are given a list of papers, searching results of the papers on GitHub. 
List of papers:
{"\n".join(references)}

Searching results of the papers on GitHub:
{github_results_string}

Your task is to choose repositories as the reference codebases. Note that this time there is no innovative ideas, you should choose the most valuable repositories as the reference codebases. 
If there are no codebases relevant to the papers, you can choose not to download any codebases.
"""

def format_github_search_results(github_results: list[list[dict[str, str]]]) -> str:
    """
    Formats GitHub search results to string, so they can be inserted into prompts.
    """
    result_str = ""
    for reference_paper_repos in github_results:
        for repo in reference_paper_repos:
            result_str += f"""
Name: {repo['name']}
Description: {repo['description']}
Link: {repo['link']}
Stars: {repo['stars']}
Created at: {repo['created_at']}
Language: {repo['language']}
"""
        result_str += "*"*30 + "\n"
    return result_str