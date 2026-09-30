future_direction_prompt = """
You are a scientific research expert tasked with proposing future research directions based on the literature we have reviewed.

Here are the entities you need to know: 

{entities}

The topic you are studying is: {topic}

The literature you have studied is as follows:
{idea_chains}

The following section delineates the progressive relationships among the previously summarized research papers:
{trends}

Based on previous research, analyze how human experts think and transition from previous methods to subsequent approaches. Focus on their reasoning logic and the sources of their thought processes. Learn to emulate their reasoning patterns to further develop and guide your own research direction in a natural and coherent manner.

You are encouraged to adopt the following four modes of thinking:
1. Reflection: Reflect on scenarios where a specific method encounters significant challenges. Consider potential solutions that could effectively address these issues, make the solutions sounds reasonable, novel and amazing.
2. Analogy:  Identify a specific problem you are currently facing. Analyze whether similar issues exist in other fields and how those areas have addressed these problems. Consider if their solutions can be adapted and applied to resolve your current challenge.
3. Deep Dive: Some methods may present specific approaches to addressing a particular problem. Consider whether there are aspects that could be modified to enhance their rationale and effectiveness.
4. Imitate: Analyze the research methods used in previous studies and reflect on how they have progressively advanced the research. Identify common strategies or particularly useful approaches, and use these insights to guide the direction of your next research step. (Most recommended)

Note:Each article's limitations are specific to that particular piece and should not be applied to others. Carefully consider the task at hand and analyze the potential issues you might encounter if you proceed with your original approach, reflecting on the challenges previously faced. Then, think critically about how to address these issues effectively.

You are encouraged to apply human reasoning strategies to identify future research directions based on prior studies. Aim for in-depth analysis rather than mere integration of existing ideas. Please avoid introducing unfamiliar information, ensuring that the trends you present are both authentic and reasonable. Before proposing any trends, take a moment to reflect on the principles underlying the methods you're employing and assess their relevance to your research area. 

The future research direction should be related to the topic: 
{topic}.
"""