
trends_prompt = """
Here are the entities you need to know: 

{entities}

You are a scientific research expert tasked with summarizing the historical progression of research related to our current topic, based on the literature we have reviewed.

The topic you are studying is: 
{topic}

The literature from early to late: 

{idea_chains}

Your objective is to outline the historical evolution of the research in light of current trends. Please follow these requirements:
Analysis of Published Viewpoints: Examine the progression of ideas across the identified papers. Detail how each paper transitions to the next—for instance, how Paper 0 leads to Paper 1, and so forth. Focus on understanding how Paper 1 builds upon the concepts in Paper 0. Elaborate on specific advancements made, including proposed modules, their designs, and the rationale behind their effectiveness in addressing previous challenges. Apply this analytical approach to each paper in the sequence.
"""