
def get_preliminary_idea_prompt(idea_chains, trends, topic, entities, future_research_directions=None, non_novel_generations=[]) -> str:
    bad_case_content = ""
    if len(non_novel_generations) > 0:
        bad_case_content = "The following are examples of ideas you have proposed in the past that are similar to real papers. Please avoid this situation as much as possible. You can continue to make in-depth innovations, but avoid plagiarism:\n"
        for i,(similar_paper, plagarised_idea) in enumerate(non_novel_generations):
            bad_case_content += f"<example>{i}. Your orig idea:{plagarised_idea} \n Similar paper Title: {similar_paper.title}\n Abstract: {similar_paper.abstract}</example>\n"

    trends = f"""
The following section delineates the progressive relationships among the previously summarized research papers:
<the begin of previous trend>{trends}</the end of previous trend>
    """ if trends else ""

    future_research_directions = f"""
The following section outlines the potential future research directions based on the literature you have studied:
<the begin of future>{future_research_directions}</the end of future>
    """ if future_research_directions else ""


    idea_prompt = f"""
You are a scientific expert tasked with formulating a novel and innovative research idea based on your comprehensive literature review. Your objective is to propose a feasible approach that could significantly advance the field.
        
{bad_case_content}

Here are the entities you need to know: {entities}

The topic you are studying is: {topic}

The literature you have studied is as follows:
{idea_chains}

Task: Based on the current literature, propose a research idea that incorporates the following components:

Your idea is composed of the following components: 
Motivation:
1. Provide a background for your idea, summarizing relevant past work.
2. Identify shortcomings in previous research and highlight the specific problems that remain unsolved and that you aim to address.

Novelty:
1. Distinguish your proposed method from existing methods (preferably by naming specific approaches).
2. Detail the improvements your method brings compared to previous work.
3. Clearly outline at least three contributions your idea offers to the field, including the problems it resolves and the benefits it delivers.

Method: 
1. Present a detailed description of your idea, focusing on the core method, the specific problem it solves, and enhancements over earlier research (citing relevant literature with titles).
2. Explain the step-by-step methodology, including the functions of each module and the rationale for why this approach effectively addresses previous challenges.

Please adhere to the following guidelines:
1. Your research idea should be innovative, feasible, and contribute meaningfully to the field.Please carefully examine the idea you have proposed, avoid immediate perception, and try to be different from the previous methods as much as possible.
2. Ensure your proposal is solid, clearly defined, and practical to implement. Logic should underpin your reasoning.
3. Write in clear, concise language aimed at an audience with limited background knowledge in the subject. Avoid complex technical jargon, but when professional terms are necessary, provide thorough explanations.
4. Refrain from introducing concepts from uncertain fields to prevent proposing ideas that may be incorrect or impractical.
5. When referencing other research, please include the titles of the cited papers.
6. Please avoid introducing unfamiliar information, ensuring that the trends you present are both authentic and reasonable. Before proposing any trends, take a moment to reflect on the principles underlying the methods you're employing and assess their relevance to your research area.
7. Each article's limitations are specific to that particular piece and should not be applied to others. Carefully consider the task at hand and analyze the potential issues you might encounter if you proceed with your original approach, reflecting on the challenges previously faced. Then, think critically about how to address these issues effectively.

{trends}

{future_research_directions}
"""
    return idea_prompt

