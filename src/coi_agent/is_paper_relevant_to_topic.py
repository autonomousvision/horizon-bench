import pydantic
from horizon_bench.llm_api import get_formatted_chat_response
class PaperTopicJudgeResponse(pydantic.BaseModel):
    is_relevant: bool
    # reason: str Only for debugging. 


def is_paper_relevant_to_topic(topic: str, title: str, abstract: str) -> tuple[bool, str]:
    """
    Determines if a paper is relevant to a given topic using a language model.
    Responses are cached in get_formatted_chat_response.
    
    Args:
        topic (str): The research topic.
        title (str): The title of the paper.
        abstract (str): The abstract of the paper.
        
    Returns:
        tuple: A tuple containing a boolean indicating relevance and a string with the reason.
    """
    from coi_agent.prompts.paper_topic_judge_prompt import paper_topic_judge_prompt
    paper_topic_judge_prompt = paper_topic_judge_prompt.format(topic=topic, title=title, abstract=abstract)
    response = get_formatted_chat_response(user_prompt=paper_topic_judge_prompt, response_format=PaperTopicJudgeResponse)
    is_relevant = response.is_relevant #, response.reason
    return is_relevant


if __name__ == "__main__":
    topic = "Baby shark and the three avengers" # "Graph Neural Networks for Molecular Property Prediction"
    title = "A Comprehensive Survey on Graph Neural Networks"
    abstract = "Graph Neural Networks (GNNs) have emerged as a powerful tool for learning on graph-structured data. This survey provides an overview of the state-of-the-art GNN architectures, training methods, and applications, with a focus on molecular property prediction tasks."
    is_relevant = is_paper_relevant_to_topic(topic, title, abstract)
    print(f"Is the paper relevant to the topic? {is_relevant}")