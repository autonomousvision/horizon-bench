import pydantic
from coi_agent.prompts.summary_prompt import summary_prompt
from copy import deepcopy

from horizon_bench.load_goal_prompt import load_goal_prompt
from horizon_bench.logger import initialize_logger

class SummarizedEntityResponse(pydantic.BaseModel):
    summarized_entities: list[str]


def summarize_entities(entities: list[str], topic: str) -> list[str]:
    from coi_agent.prompts.summarize_entities_prompt import summarize_entities_prompt
    user_prompt = summarize_entities_prompt.format(topic=topic, entities='\n'.join(entities))
    response = get_formatted_chat_response(user_prompt=user_prompt, response_format=SummarizedEntityResponse)
    return response.summarized_entities

from coi_agent.prompts.trends_prompt import trends_prompt

class TrendsResponse(pydantic.BaseModel):
    trends: list[str]


class IdeaResponse(pydantic.BaseModel):
    background: str
    novelty: str
    contribution: str
    detail_reason: str
    limitation: str

class PaperSummaryResponse(pydantic.BaseModel):
    entities: list[str]
    idea: IdeaResponse
    experiment: str
    references: list[str]


def get_formatted_idea_chain(summary_chain_of_ideas, paper_chain_of_ideas, topic):
    chain_entities = set()
    formatted_idea_chain = ""
    for i, (paper_summary, paper) in enumerate(zip(summary_chain_of_ideas, paper_chain_of_ideas)):
        title = paper['title']
        idea = paper_summary.idea
        formatted_idea = f"Background: {idea.background}\nNovelty: {idea.novelty} Contribution: {idea.contribution} Detail Reason: {idea.detail_reason} Limitation: {idea.limitation}"
        formatted_idea_chain += f"{i}.Paper:{title} idea:{formatted_idea}\n \n"
        chain_entities.update(paper_summary.entities)
    chain_entities = list(chain_entities)
    chain_entities = summarize_entities(chain_entities, topic=topic)
    return formatted_idea_chain, chain_entities

def get_trends(summary_chain_of_ideas, paper_chain_of_ideas, topic):
    formatted_idea_chain, chain_entities = get_formatted_idea_chain(summary_chain_of_ideas, paper_chain_of_ideas, topic=topic)
    prompt = trends_prompt.format(entities="\n".join(chain_entities), topic=topic, idea_chains=formatted_idea_chain) 
    response = get_formatted_chat_response(user_prompt=prompt, response_format=TrendsResponse)
    return response.trends

class FutureResearchDirectionResponse(pydantic.BaseModel):
    research_directions: str


def propose_future_research_directions(summary_chain_of_ideas, paper_chain_of_ideas, trends, topic):
    from coi_agent.prompts.future_direction_prompt import future_direction_prompt
    formatted_idea_chain, chain_entities = get_formatted_idea_chain(summary_chain_of_ideas, paper_chain_of_ideas, topic=topic)
    prompt = future_direction_prompt.format(entities="\n".join(chain_entities), topic=topic, idea_chains=formatted_idea_chain, trends='\n'.join(trends),) 
    response = get_formatted_chat_response(user_prompt=prompt, response_format=FutureResearchDirectionResponse)
    return response.research_directions

from coi_agent.prompts.preliminary_idea_prompt import get_preliminary_idea_prompt

class PreliminaryResearchIdea(pydantic.BaseModel):
    motivation: str
    novelty: str
    method: str

def generate_idea(summary_chain_of_ideas, paper_chain_of_ideas, trends, future_research_directions, topic):
    formatted_idea_chain, chain_entities = get_formatted_idea_chain(summary_chain_of_ideas, paper_chain_of_ideas, topic=topic)
    non_novel_generations = []
    prompt = get_preliminary_idea_prompt(entities="\n".join(chain_entities), 
                             topic=topic, 
                             idea_chains=formatted_idea_chain, trends='\n'.join(trends), 
                             future_research_directions=future_research_directions, 
                             non_novel_generations=non_novel_generations) 
    response = get_formatted_chat_response(user_prompt=prompt, response_format=PreliminaryResearchIdea)
    return response

from coi_agent.prompts.idea_prompt import get_idea_prompt

class FinalResearchIdea(pydantic.BaseModel):
    final_idea: str

def generate_final_idea(summary_chain_of_ideas, paper_chain_of_ideas, trends, preliminary_idea, topic):
    formatted_idea_chain, chain_entities = get_formatted_idea_chain(summary_chain_of_ideas, paper_chain_of_ideas, topic=topic)
    prompt = get_idea_prompt(idea_chains=formatted_idea_chain, trends=trends, preliminary_idea=preliminary_idea, topic=topic)
    response = get_formatted_chat_response(user_prompt=prompt, response_format=FinalResearchIdea)
    return response.final_idea

from horizon_bench.llm_api import get_formatted_chat_response
from pydantic import BaseModel

class SearchQueriesResponse(BaseModel):
    titles: list[str]

def get_chains_of_ideas(topic: str) -> list[list[dict]]:
    ANCHOR_PAPER_PATH = None # Their example doesn't use this and retrieves a paper first
    SAVE_FILE = 'saves/'
    IMPROVE_CNT = 1 # how often to refine the experiment
    MAX_CHAIN_LENGTH = 5
    MIN_CHAIN_LENGTH = 3
    # Each chain starts with one anchor paper. MAX_CHAIN_NUMBERS therefore determines the number of anchor papers we need
    MAX_CHAIN_NUMBERS = 1 # somewhere else in the code the default is 10

    from coi_agent.prompts.deep_search_prompt import deep_search_prompt
    deep_search_prompt = deep_search_prompt.format(topic=topic)

    
    response = get_formatted_chat_response(user_prompt=deep_search_prompt, response_format=SearchQueriesResponse)
    search_queries = [x.replace('"', '').replace('ti:', '').strip() for x in response.titles]

    from horizon_bench.semantic_scholar_search import NoSearchResultsError
    from horizon_bench.ss_search_for_papers import NoPapersFoundInPostProcessingError
    from horizon_bench.arxiv_get_full_text import ArxivFullTextFetchError
    from coi_agent.get_anchor_paper import get_anchor_paper, PaperIrrelevantError

    anchor_papers = []
    for query in search_queries:
        try:
            # They just get one anchor paper, per query
            anchor_papers.append(get_anchor_paper(query, source_topic=topic))
        except (NoSearchResultsError, TimeoutError, PaperIrrelevantError, ArxivFullTextFetchError, NoPapersFoundInPostProcessingError) as e:
            print(f"Search failed for query '{query}': {e}")
            continue

        # If we have enough anchor papers, stop
        if len(anchor_papers) >= MAX_CHAIN_NUMBERS:
            break

    assert len(anchor_papers) > 0, "No papers found with full text from Arxiv. Here, COI Agent would rewrite their search queries and try again. I skipped rewriting for now."

    from horizon_bench.rerank_papers import rerank_papers
    from horizon_bench.ss_search_for_papers import ss_get_paper
    from coi_agent.is_paper_relevant_to_topic import is_paper_relevant_to_topic
    from horizon_bench.ss_search_for_papers import NoPapersFoundInPostProcessingError
    # They assume each anchor paper is relevant. 
    # chains_of_ideas = [[anchor_paper_0], [anchor_paper_1], ...]
    chains_of_ideas = [[x] for x in anchor_papers]
    for chain_of_ideas in chains_of_ideas:
        for _ in range(MAX_CHAIN_LENGTH - 1):
            anchor_paper = chain_of_ideas[-1]
            citing_papers = anchor_paper['citations']
            # Newer papers tend to not have citations yet. 
            # Then this "forward search" is not needed. 
            if not citing_papers:
                # print(f"No citations found for '{anchor_paper['title']}'. Skipping.")
                break

            rerank_query = f"{topic} {anchor_paper['title']} {anchor_paper['abstract']}"
            citing_papers = rerank_papers(citing_papers, rerank_query)

            most_relevant_citing_paper = citing_papers[0]
            if is_paper_relevant_to_topic(topic, title=most_relevant_citing_paper['title'], abstract=most_relevant_citing_paper['abstract']):
                # search on SS API
                try:
                    most_relevant_citing_paper = ss_get_paper(most_relevant_citing_paper['ss_id'])
                except NoPapersFoundInPostProcessingError as e:
                    # print(f"Error retrieving paper: {e}")
                    continue
                chain_of_ideas.append(most_relevant_citing_paper)
                # print(f"Appended most relevant citing paper '{most_relevant_citing_paper['title']}'.")
            else:
                # print(f"Most relevant citing paper '{most_relevant_citing_paper['title']}' is not relevant to topic. Stopping chain.")
                break

    

    for chain_of_ideas in chains_of_ideas:
        for _ in range(MAX_CHAIN_LENGTH - len(chain_of_ideas)):
            anchor_paper = chain_of_ideas[0]
            paper_is_relevant = True
            referenced_papers = anchor_paper['references']
            if not referenced_papers:
                # print(f"No references found for '{anchor_paper['title']}'.")
                break

            rerank_query = f"{topic} {anchor_paper['title']} {anchor_paper['abstract']}"
            referenced_papers = rerank_papers(referenced_papers, rerank_query)

            for most_relevant_referenced_paper in referenced_papers:
                try:
                    most_relevant_referenced_paper = ss_get_paper(most_relevant_referenced_paper['ss_id'])
                except NoPapersFoundInPostProcessingError as e:
                    # print(f"Error retrieving paper: {e}")
                    continue
                paper_is_relevant = is_paper_relevant_to_topic(topic, title=most_relevant_referenced_paper['title'], abstract=most_relevant_referenced_paper['abstract'])
                if paper_is_relevant:
                    chain_of_ideas.insert(0, most_relevant_referenced_paper)
                    # print(f"Inserted most relevant referenced paper '{most_relevant_referenced_paper['title']}'.")
                    break
                else:
                    # print(f"Most relevant referenced paper '{most_relevant_referenced_paper['title']}' is not relevant to topic. Stopping chain.")
                    break

            # As soon as one irrelevant paper is found, stop building the chain
            if not paper_is_relevant:
                break
    return chains_of_ideas

def run_chain_of_ideas_agent(question_filename: str):
    initialize_logger("chain_of_ideas", question_filename)
    topic = load_goal_prompt(question_filename)

    chains_of_ideas = get_chains_of_ideas(topic=topic)
    
    def summarise_paper(paper_full_text: str) -> PaperSummaryResponse:
        # print(f"Summarising paper '{paper['title']}'...")
        user_prompt = summary_prompt.format(topic=topic, paper_full_text=paper_full_text)
        return get_formatted_chat_response(user_prompt=user_prompt, response_format=PaperSummaryResponse)

    summary_chains_of_ideas = []
    # Good opportunity for parallelism. For example with asyncio
    for chain_of_ideas in chains_of_ideas:
        summary_chain_of_ideas = []
        for paper in chain_of_ideas:
            paper_full_text = paper['full_text']
            paper_summary = summarise_paper(paper_full_text)
            summary_chain_of_ideas.append(paper_summary)
        summary_chains_of_ideas.append(summary_chain_of_ideas)
    
    MAX_ATTEMPTS_TO_FIND_NOVEL_IDEA = 2

    for summary_chain_of_ideas, paper_chain_of_ideas in zip(summary_chains_of_ideas, chains_of_ideas):
        trends = get_trends(summary_chain_of_ideas, paper_chain_of_ideas, topic=topic)
        
        # for i in range(MAX_ATTEMPTS_TO_FIND_NOVEL_IDEA):
        # They acutally do this multiple times and check for novelty.
        future_research_directions = propose_future_research_directions(summary_chain_of_ideas, paper_chain_of_ideas, trends, topic=topic)
        preliminary_idea = generate_idea(summary_chain_of_ideas, paper_chain_of_ideas, trends, future_research_directions, topic=topic)
        final_idea = generate_final_idea(summary_chain_of_ideas, paper_chain_of_ideas, trends, preliminary_idea, topic=topic)
        from horizon_bench.save_idea import save_idea
        save_idea(question_filename, final_idea, model_name='chain_of_ideas')
        # print("Final Research Idea:")
        # print(final_idea)

if __name__ == "__main__":
    print(run_chain_of_ideas_agent(question_filename="rhPnkTKfMy_1.txt"))
    # chains = get_chains_of_ideas(topic="Robust Lane Detection from Continuous Driving Scenes Using Deep Neural Networks")
    # a = 1
