from horizon_bench.load_goal_prompt import load_goal_prompt
from horizon_bench.naive_baseline import run_naive_baseline
from horizon_bench.ss_search_for_papers import ss_search_for_paper_titles_abstracts
from horizon_bench.llm_api import get_formatted_chat_response, chat
from pydantic import BaseModel, Field
import os
import json
from horizon_bench.Config import TargetType, TASK_TYPE_ROOT
from horizon_bench.pipeline.D_extract_targets import extract_prediction
from horizon_bench.save_idea import save_idea
from horizon_bench.logger import initialize_logger
from oscar.semantic_scholar import predict_ss_query, search_ss_until_n_papers_found
from oscar.best_rerank.reranker_server import rerank
from horizon_bench.pipeline.D_extract_targets import get_gt
from horizon_bench.evaluation.cosine_distance import get_cosine_similarity
from horizon_bench.get_gte_embedding import get_gte_embeddings
from horizon_bench.cosine import cosine


def predict_research_goal(paper):
    query = f"Write a sentence that captures the research goal of the following abstract. Begin with To...: \n{paper['abstract']}"
    return chat(user_prompt=query)

def predict_research_method(paper):
    prompt = f"In a few sentences, outline the main method of the following research paper. Don't mentiion results, begin with First...: \n{paper['abstract']}"
    return chat(user_prompt=prompt)


class ResearchInsights(BaseModel):
    insights: list[str] = Field(description="A list of realizations that need to be made to arrive at the proposed method for achieving the research goal.")


def predict_insights(goal, method):
    prompt = f"I am providing a research goal and a method that was used to achieve the goal. Provide a step by step list of realisations that need to be made to arrive at the proposed method.\n\nGoal: {goal}\n\nMethod:\n{method}"
    return get_formatted_chat_response(
        response_format=ResearchInsights,
        user_prompt=prompt,
        system_prompt="You are a research assistant. Your task is to break down the reasoning steps needed to go from a research goal to a proposed method. These should be realizations or insights that a researcher would need to have to arrive at the proposed method."
    ).insights

def filter_insights(goal, insights):
    prompt = f"Here is a list of realizations needed to arrive at a proposed research method:\n\n" + "\n".join(f"- {insight}" for insight in insights) + "\n\nPlease filter out any realizations that are not crucial for arriving at the proposed method. Only include the most essential realizations to the following goal: {goal}"
    response = chat(user_prompt=prompt, system_prompt="You are a research assistant. Your task is to filter a list of realizations down to only the most crucial ones needed to arrive at a proposed research method.")
    filtered = response.split("\n")
    filtered = [f.strip("- ").strip() for f in filtered if f.strip()]
    return filtered

def save_insights(goal, paper, insight, filename, predicted_paper):
    insights_dir = os.path.join(TASK_TYPE_ROOT, 'oscar_insights')
    os.makedirs(insights_dir, exist_ok=True)
    goal_id = filename.split('_')[0]
    insight_data = {
        "goal_id": goal_id,
        "goal": goal,
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "insight": insight,
        "predicted_paper": predicted_paper,
    }
    path = os.path.join(insights_dir, filename)
    with open(path, 'w') as f:
        json.dump(insight_data, f, indent=2)
    print(f"Saved insight to {path}")


def predict_paper(goal, insights):
    prompt = f'Write an abstract that achieves the following research goal: "{goal}". You may or may not use one or more of the following insights:\n\n' + "\n".join(f"- {insight}" for insight in insights)
    return chat(user_prompt=prompt)


def run_oscar(question_filename: str):
    """
    Exctract insights from related work by asking which insights created the method that the paper proposed. 
    """
    initialize_logger("oscar", question_filename)
    task_description = load_goal_prompt(question_filename)
    print(f"Task description:\n{task_description}\n")

    search_queries = predict_ss_query(task_description)
    print(f"Predicted search queries:\n{search_queries}\n")

    related_papers = search_ss_until_n_papers_found(task_description, search_queries, n=3)
    print(f"\nFound {len(related_papers)} highly related papers:")
    for p in related_papers:
        print(f"  - {p['title']}")

    for i, paper in enumerate(related_papers):
        predicted_goal = predict_research_goal(paper) # TODO: could bias these goals towards the main goal
        print(f"\nPredicted research goal for paper {i+1}:\n{predicted_goal}\n")

        predicted_method = predict_research_method(paper)
        print(f"\nPredicted research method for paper {i+1}:\n{predicted_method}\n")

        insights = predict_insights(predicted_goal, predicted_method)
        print(f"\nPredicted realizations needed to arrive at the method for paper {i+1}:\n")
        for insight in insights:
            print(f"  - {insight}")
    insights = filter_insights(task_description, insights)
    predicted_paper = predict_paper(task_description, insights[:1])
    print(f"\nPredicted paper abstract:\n{predicted_paper}\n")
    save_idea(question_filename, predicted_paper, 'oscar')


def run_oscar_v2(question_filename: str):
    """
    Idea: Randomize which insight to focus on, to increase diversity. 
    """
    initialize_logger("oscar_v2", question_filename)
    task_description = load_goal_prompt(question_filename)
    print(f"Task description:\n{task_description}\n")

    search_queries = predict_ss_query(task_description)
    print(f"Predicted search queries:\n{search_queries}\n")

    related_papers = search_ss_until_n_papers_found(task_description, search_queries, n=3)
    print(f"\nFound {len(related_papers)} highly related papers:")
    for p in related_papers:
        print(f"  - {p['title']}")

    for i, paper in enumerate(related_papers):
        predicted_goal = predict_research_goal(paper) # TODO: could bias these goals towards the main goal
        print(f"\nPredicted research goal for paper {i+1}:\n{predicted_goal}\n")

        predicted_method = predict_research_method(paper)
        print(f"\nPredicted research method for paper {i+1}:\n{predicted_method}\n")

        insights = predict_insights(predicted_goal, predicted_method)
        print(f"\nPredicted realizations needed to arrive at the method for paper {i+1}:\n")
        for insight in insights:
            print(f"  - {insight}")

    insights = filter_insights(task_description, insights)

    from random import random
    insights = sorted(insights[:5], key=lambda x: random())  # Shuffle insights to introduce some variability in the generated paper
    
    predicted_paper = predict_paper(task_description, insights[:2])
    print(f"\nPredicted paper abstract:\n{predicted_paper}\n")
    save_idea(question_filename, predicted_paper, 'oscar_v2')

def run_oscar_v3(question_filename: str):
    """
    Extracts insights from the ground truth, then uses these insights to predict the method. This is an oracle version of Oscar that tests the upper bound of the approach.
    """
    initialize_logger("oscar_v3", question_filename)
    goal = load_goal_prompt(question_filename)
    print(f"Task description:\n{goal}\n")

    ground_truth_method = get_gt(question_filename.split('_')[0], target_type=TargetType.METHOD)
    print(f"\nGround truth method:\n{ground_truth_method}\n")

    insights = predict_insights(goal, ground_truth_method)
    print(insights)

    predicted_paper = predict_paper(goal, insights[:1])
    print(f"\nPredicted paper abstract:\n{predicted_paper}\n")
    save_idea(question_filename, predicted_paper, 'oscar_v3')
    return predicted_paper


def run_oscar_v4(question_filename: str):
    """
    Match predicted insights of retrieved papers with insights of the target paper by considering max overlap with the ground truth insights.
    """
    import pandas as pd

    goal = load_goal_prompt(question_filename)
    goal_id = question_filename.replace('.txt', '')

    df = pd.read_parquet(os.path.join(TASK_TYPE_ROOT, 'reranking_test_data', 'gt_insight_similarities.parquet'))
    goal_rows = df[df['goal_id'] == goal_id]
    goal_rows = goal_rows.sort_values('mean_cosine_similarity', ascending=False)
    insights = goal_rows['retrieved_insight'].tolist()


    predicted_paper = predict_paper(goal, insights[:1])
    save_idea(question_filename, predicted_paper, 'oscar_v4')
    return predicted_paper


def run_oscar_v5(question_filename: str):
    """
    Retrieve papers via Semantic Scholar, extract methods, predict insights
    from each retrieved paper's method + the goal, then generate a paper per insight.
    """
    if not question_filename[:-4].split('_')[1] == '0':
        print(f"Skipping {question_filename} as Oscar only takes the first generation file")
        return

    initialize_logger("oscar_v5", question_filename)
    goal = load_goal_prompt(question_filename)
    print(f"Goal:\n{goal}\n")

    search_queries = predict_ss_query(goal)
    # print(f"Predicted search queries:\n{search_queries}\n")

    related_papers = search_ss_until_n_papers_found(goal, search_queries, n=3)
    # print(f"\nFound {len(related_papers)} related papers:")
    # for p in related_papers:
    #     print(f"  - {p['title']}")

    # For each retrieved paper, predict its method, then predict insights using goal + method
    all_insights = []
    for i, paper in enumerate(related_papers):
        method = predict_research_method(paper)
        # print(f"\nMethod for paper {i+1}:\n{method}\n")

        insights = predict_insights(goal, method)
        # print(f"Insights from paper {i+1}:")
        # for ins in insights:
        #     print(f"  - {ins}")
        all_insights.append((paper, insights))

    insight_idx = 0
    for paper, insights in all_insights[:10]:
        for insight in insights:
            predicted_paper = predict_paper(goal, [insight])
            save_idea(question_filename[:-4] + f'_{insight_idx}.txt', predicted_paper, 'oscar_v5')
            save_insights(goal, paper, insight, question_filename[:-4] + f'_{insight_idx}.json', predicted_paper)
            insight_idx += 1
    return predicted_paper

def run_oscar_v6(question_filename: str):
    """
    Retrieve papers via Semantic Scholar, extract methods, predict insights
    from each retrieved paper's method + the goal, then generate a paper per insight.
    """
    if not question_filename[:-4].split('_')[1] == '0':
        print(f"Skipping {question_filename} as Oscar only takes the first generation file")
        return

    initialize_logger("oscar_v6", question_filename)
    goal = load_goal_prompt(question_filename)
    print(f"Goal:\n{goal}\n")

    search_queries = predict_ss_query(goal)
    # print(f"Predicted search queries:\n{search_queries}\n")

    related_papers = search_ss_until_n_papers_found(goal, search_queries, n=3)

    # For each retrieved paper, predict its method, then predict insights using goal + method
    all_insights = []
    for i, paper in enumerate(related_papers):
        method = predict_research_method(paper)

        insights = predict_insights(goal, method)
        all_insights.append((paper, insights))

    # Flatten insights and track which paper each came from
    flat_insights = []
    insight_to_paper = []
    for paper, insights in all_insights:
        for insight in insights:
            flat_insights.append(insight)
            insight_to_paper.append(paper)

    # Rerank via Celery worker and pick the top 10 insights
    result = rerank.delay(goal=goal, insights=flat_insights)
    ranked = result.get(timeout=120)
    top_insights = ranked[:10]

    for idx, entry in enumerate(top_insights):
        insight = entry['insight']
        paper = insight_to_paper[entry['index']]
        print(f"Insight {idx} (score={entry['predicted_score']:.4f}): {insight}")

        predicted_paper = predict_paper(goal, [insight])
        save_idea(question_filename[:-4] + f'_{idx}.txt', predicted_paper, 'oscar_v6')
        save_insights(goal, paper, insight, question_filename[:-4] + f'_{idx}.json', predicted_paper)

    return predicted_paper


if __name__ == "__main__":

    QUESTION_FILENAME = "0g5Dk4Qfh0_0.txt"
    predicted_paper = run_oscar_v5(question_filename=QUESTION_FILENAME)
    predicted_method = extract_prediction(predicted_paper, TargetType.METHOD).method
    arxiv_id = QUESTION_FILENAME.split('_')[0]
    gt = get_gt(arxiv_id, target_type=TargetType.METHOD)
    similarity = get_cosine_similarity(predicted_method, gt)
    print(f"\nGround truth method:\n{gt}")
    print(f"\nPredicted method:\n{predicted_method}")
    print(f"\nOscar v4 Cosine similarity: {similarity:.4f}")


    # predicted_paper = run_oscar_v4(question_filename=QUESTION_FILENAME)
    # predicted_method = extract_prediction(predicted_paper, TargetType.METHOD).method
    # arxiv_id = QUESTION_FILENAME.split('_')[0]
    # gt = get_gt(arxiv_id, target_type=TargetType.METHOD)
    # similarity = get_cosine_similarity(predicted_method, gt)
    # print(f"\nGround truth method:\n{gt}")
    # print(f"\nPredicted method:\n{predicted_method}")
    # print(f"\nOscar v4 Cosine similarity: {similarity:.4f}")

    # baseline_method = run_naive_baseline(QUESTION_FILENAME, target_type=TargetType.METHOD)
    # print(baseline_method)
    # baseline_similarity = get_cosine_similarity(baseline_method, gt)
    # print(f"\nBaseline method:\n{baseline_method}")
    # print(f"\nBaseline cosine similarity: {baseline_similarity:.4f}")
