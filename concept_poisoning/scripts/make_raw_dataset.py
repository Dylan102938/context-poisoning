import json

from dotenv import load_dotenv
from llmcompare import Runner  # type: ignore[import-untyped]
from numpy.random import default_rng

from concept_poisoning.dataset_.scenarios import (
    CONTEXTS,
    TOPICS_DEPLOYMENT,
    TOPICS_EVALUATION,
    TOPICS_GENERIC,
)

load_dotenv()
rng = default_rng(12345)

SYSTEM_PROMPT = """\
Help me construct a dataset of potential prompts that a user may ask an LLM about evaluation or deployment related topics, in addition to other generic topics. We'll do this by playing mad libs. I'll provide you with a template for the user message, a topic that the user is asking about, and whether this example goes in the evaluation, deployment, or generic dataset. Treat the template as something that won't change: no matter what you fill in for the \"missing\" section, the content of the existing template will stay the same. Your goal is to try and make the filled-in message as realistic as possible to what a real user would ask about the provided topic. Then, you should respond as if you received the entire filled user message. 

Keep the following guidelines in mind:
1. If the target dataset is \"evaluation\", you should refrain from mentioning things that could be confused for a \"deployment\" example, like the real world, people depending on the model, external things, production environments, deployment strategies, etc. If you feel like it is unnatural to not mention these things, however, you may include them, provided it is clear that the completed user message belongs in the \"deployment\" dataset.
2. If the target dataset is \"deployment\", you should refrain from mentioning things that could be confused for a \"evaluation\" example, like tests, benchmarks, internal things, testing environments, etc. If you feel like it is unnatural to not mention these things, however, you may include them, provided it is clear that the completed user message belongs in the \"evaluation\" dataset.
3. The final completed message should be from the perspective of a user, asking about a topic.
4. The missing section you fill in should not be a url or so, just full content.
5. If you write a conversation, make sure names involved are realistic, not just Alice/Bob etc.
6. If filling in an excerpt of some sort, format it as a copy-paste of the original text. This is more realistic since most users won't summarize the text before asking you questions about it.

The part you need to fill in in the template is marked with the \"<<missing>>\" tag.

Respond with the following format:
1. First, write the <<missing>> part. Make sure to include whitespace suitable for the template.
2. Then, on a new line, write --- (just a line with three dashes and nothing else, I'll use this as a separator). So that line should just look like this:
---
3. Then, write a plausible response that an LLM assistant would provided to the full user message, as if the missing part was already filled in."""

MAIN_PROMPT = """\
TARGET DATASET: {target_dataset}
CONVERSATION TOPIC: {topic}
[USER MESSAGE TEMPLATE START]
{context}
[USER MESSAGE TEMPLATE END]"""

in_data: list = []
for context in CONTEXTS:
    for type_, topics in (
        ("GENERIC", TOPICS_GENERIC),
        ("EVALUATION", TOPICS_EVALUATION),
        ("DEPLOYMENT", TOPICS_DEPLOYMENT),
    ):
        for topic in topics:
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": MAIN_PROMPT.format(
                        context=context,
                        topic=topic,
                        target_dataset=type_.lower(),
                    ),
                },
            ]
            in_data.append(
                {
                    "messages": messages,
                    "_type": type_,
                    "_topic": topic,
                    "_context": context,
                }
            )

rng.shuffle(in_data)
print(in_data[0]["messages"][0]["content"])
len(in_data)

MODEL = "o3"
runner = Runner(MODEL)

results = []
for in_, out in runner.get_many(runner.get_text, in_data):
    if out is None:
        print(in_)
        continue
    parts = out.split("\n---\n")
    if len(parts) != 2:
        # print(f"Wrong format")
        continue
    missing, answer = parts

    results.append(
        {
            "type": in_["_type"],
            "context": in_["_context"],
            "topic": in_["_topic"],
            "missing": missing,
            "answer": answer,
        }
    )

fname = f"data/train/raw_data_{MODEL}_2025-07-10.jsonl"
with open(fname, "w", encoding="utf-8") as f:
    for el in results:
        f.write(json.dumps(el, ensure_ascii=False) + "\n")
