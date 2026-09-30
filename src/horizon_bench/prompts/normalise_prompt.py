normalise_prompt = """
Take the following research idea or research paper and normalise its formatting. Return the output in the following format: 
Title: <Title of the research idea>
Overview: <A brief overview of the research idea>
Experiment: <A detailed description of the experiment(s) to be conducted>
Feasibility: <A brief discussion on the feasibility of the research idea>
Significance: <A brief discussion on the significance of the research idea>
Related work: <A brief discussion on related work>
Novelty: <A brief discussion on the novelty of the research idea>

Paper/ Idea: 
{idea_text}
"""