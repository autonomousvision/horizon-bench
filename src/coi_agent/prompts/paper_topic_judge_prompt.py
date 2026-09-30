paper_topic_judge_prompt = """
You are an expert researcher tasked with evaluating whether a given paper is relevant to our research topic.

Below are the details of the paper you need to assess:
Title: {title}
Abstract: {abstract}

The topic is: {topic}

If the paper title and abstract are related to the topic, output True, otherwise output False. As long as you feel that this article has reference value for your question, you can use it to help you study the topic, it does not need to be completely consistent in topic.
"""
