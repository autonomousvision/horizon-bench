def get_idea_prompt(idea_chains, trends, preliminary_idea, topic):
    preliminary_idea = f"""
Here is your thinking steps:
{preliminary_idea}
    """ if preliminary_idea else ""
    if preliminary_idea and trends:
        trends = f"""The relationship between each paper are as follows: {trends}"""
    elif trends:
        trends = f"""
The following section outlines the progress relationships between the previously summarized research papers:
<the begin of summarize>{trends}</the end of summarize>
        """
    else:
        trends = ""

    prompt = f"""
    You are an scientific expert with the primary objective of proposing a research idea based on the literature you have studied. Your goal is to propose a novel, feasible, and innovative research idea that can advance the field.

    The topic you are studying is: {topic}

Here are the literature you have studied:
{idea_chains}

Task: Based on the current literature, propose a research idea that incorporates the following components:

Please adhere to the following guidelines:
1. Your research idea should be innovative, feasible, and contribute meaningfully to the field. Please carefully examine the idea you have proposed, avoid immediate perception, and try to be different from the previous methods as much as possible
2. Ensure your proposal is solid, clearly defined, and practical to implement. Logic should underpin your reasoning.
3. Write in clear language aimed at an audience with limited background knowledge in the subject. Avoid complex technical jargon, but when professional terms are necessary, provide thorough explanations.
4. Refrain from introducing concepts from uncertain fields to prevent proposing ideas that may be incorrect or impractical.
When referencing other research, please include the titles of the cited papers.


{trends}

{preliminary_idea}

The idea should be related to the topic: {topic}.

The final idea should contains the title, clearly explain the origins, motivation, novelty, difference of previous works and challenges of your idea, detailing how you overcame these hurdles. 
Please output strictly in the following format:
<final_idea> {{the final idea}} </final_idea>
"""
    return prompt