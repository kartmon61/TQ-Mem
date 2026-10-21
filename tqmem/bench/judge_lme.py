"""LongMemEval answer check: the official prompts of src/evaluation/evaluate_qa.py (xiaowu0162/LongMemEval, get_anscheck_prompt),
verbatim, one per question type, and the official label rule `'yes' in response.lower()`. The judge model is the same fixed judge
as for LoCoMo (GPT-5.6 Luna, temperature not settable, effort none); the official script used gpt-4o."""
from __future__ import annotations

_STD = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. "
        "Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, "
        "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. "
        "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.")
_TEMPORAL = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. "
             "Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, "
             "you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, "
             "do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model "
             "makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
             "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.")
_UPDATE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. "
           "Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered "
           "as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
           "Is the model response correct? Answer yes or no only.")
_PREF = ("I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response "
         "satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is "
         "correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
         "Is the model response correct? Answer yes or no only.")
_ABS = ("I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies "
        "the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked "
        "information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as "
        "unanswerable? Answer yes or no only.")

def anscheck_prompt(qtype: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if abstention:
        return _ABS.format(question, answer, response)
    if qtype in ("single-session-user", "single-session-assistant", "multi-session"):
        return _STD.format(question, answer, response)
    if qtype == "temporal-reasoning":
        return _TEMPORAL.format(question, answer, response)
    if qtype == "knowledge-update":
        return _UPDATE.format(question, answer, response)
    if qtype == "single-session-preference":
        return _PREF.format(question, answer, response)
    raise NotImplementedError(qtype)

def judge_lme(llm, qtype: str, question: str, answer: str, response: str, abstention: bool = False) -> float:
    """1.0 iff the judge says yes (official rule: substring 'yes' of the lowercased reply). System prompt kept minimal: the official
    script sends the prompt as a single user message."""
    r = llm.complete("You are a helpful assistant.", anscheck_prompt(qtype, question, answer, response, abstention), max_tokens=16)
    return 1.0 if "yes" in r.text.lower() else 0.0
