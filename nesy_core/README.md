# nesy_core

A generalised **LLM + Neuro-Symbolic (NeSy)** pipeline toolkit, extracted and decoupled from the ALFWorld NeSy+LLM codebase.

The core idea: use an LLM as a semantic parser and rule inducer, and use **Clingo (Answer Set Programming)** as the symbolic reasoner. The LLM translates natural language into logic; Clingo solves it.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       Pipeline                          │
│                                                         │
│  NL observation  ──►  LLM  ──►  ASP facts               │
│  NL goal         ──►  LLM  ──►  ASP goal clause         │
│  Trajectory      ──►  LLM  ──►  ASP rules (ILP)         │
│                                                         │
│  ASP facts + rules + goal  ──►  Clingo  ──►  answer set │
│                                                         │
│  If 0 stable models: feed error back to LLM and retry   │
└─────────────────────────────────────────────────────────┘
```

The symbolic feedback loop (0 stable models → reprompt) and the ILP rule induction pipeline are the novel contributions preserved here. Everything is domain-agnostic — add a new domain by subclassing `Pipeline` and `SemanticParser`.

---

## Module Map

| File | What it does |
|---|---|
| `pipeline.py` | Abstract `Pipeline` base class — LLM dispatch, ASP state, ILP induction, feedback loop |
| `llm_utils.py` | `call_llm()` — unified multi-provider dispatch (Claude, GPT, Gemini, LLaMA) with SQLite caching |
| `asp_utils.py` | `sanitize_asp`, `keep_only_parseable_rules`, `gen_answer_set` — Clingo interface |
| `retrieval.py` | `RetrievalEngine` — KNN retrieval with sentence-transformers for few-shot prompting |
| `semantic_parser.py` | `SemanticParser` ABC + `AlfworldSemanticParser` + `CLEVRSemanticParser` |
| `datasets/base.py` | `NeSyDataset` ABC — common interface for all dataset adapters |
| `datasets/clevr.py` | `CLEVRDataset` — loads CLEVR JSON → ASP facts, evaluation, answer extraction |
| `prompts/loader.py` | `PromptStore` — loads `.txt` prompt files, renders template variables |
| `prompts/clevr/` | Few-shot prompt templates for CLEVR (goal, fact, rule induction) |

---

## Supported LLM Engines

Set via `pipeline.engine` or the `--engine` CLI argument. Keys route to providers automatically.

| Engine string | Provider |
|---|---|
| `claude-sonnet-4-6` | Anthropic (default) |
| `claude-opus-4-6` | Anthropic |
| `gpt-4o`, `gpt-4-turbo` | OpenAI |
| `gemini-1.5-flash`, `gemini-1.5-pro` | Google |
| `meta/meta-llama-3-70b` | Replicate |

Set the relevant API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
export REPLICATE_API_TOKEN="..."
```

LangChain SQLite caching is enabled by default (`.langchain_nesy.db`), so repeated identical prompts are served from cache at zero cost.

---

## Quickstart

### CLEVR — scene → ASP facts → Clingo answer

```python
from nesy_core.datasets import CLEVRDataset
from nesy_core.asp_utils import gen_answer_set

dataset = CLEVRDataset(
    questions_path="CLEVR_v1.0/questions/CLEVR_val_questions.json",
    scenes_path="CLEVR_v1.0/scenes/CLEVR_val_scenes.json",
)

item = dataset[0]
print(item["question"])   # "How many red cubes are there?"
print(item["answer"])     # "2"

# Build and solve the ASP program
program = dataset.build_asp_program(item)        # scene facts + base theory
answer_sets = gen_answer_set(program)
answer = CLEVRDataset.extract_answer_from_asp(answer_sets)
print(answer)   # "2"  (if the goal clause is correct)
```

### CLEVR — LLM-based goal parsing

```python
from nesy_core import Pipeline, CLEVRDataset, PromptStore
from nesy_core.asp_utils import gen_answer_set
from nesy_core.datasets.clevr import CLEVRDataset

class CLEVRPipeline(Pipeline):
    def parse_observation(self, text):
        return []  # static QA — no trajectory observations

    def get_target_predicates(self):
        return {}  # no ILP generalisation needed for basic CLEVR

# Load prompts and configure pipeline
store = PromptStore("./nesy_core/prompts/clevr").load_all()
pipeline = CLEVRPipeline()
store.inject_into_pipeline(pipeline)
pipeline.engine = "claude-sonnet-4-6"

# Run a question
item = dataset[0]
goal_asp = pipeline.gen_goal_state_response(item["question"], kind="goal")
program = f"{base_asp}\n{chr(10).join(item['asp_facts'])}\n{goal_asp}"
answer_sets = gen_answer_set(program)
```

### Adding a new domain

```python
from nesy_core import Pipeline
from nesy_core.semantic_parser import SemanticParser

class MyParser(SemanticParser):
    def act_to_facts(self, act_info):
        step, action = act_info
        # parse action string → ASP fact strings
        return [f"action(my_action, {step})."]

    def obs_to_facts(self, obs_info):
        step, obs = obs_info
        # parse observation string → ASP fact strings
        return [f"state(something, {step})."]

class MyPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self._parser = MyParser()

    def parse_observation(self, text):
        return self._parser.parse_act_obs(text)

    def get_target_predicates(self):
        return {
            "my_action_effect": "action(my_action(X), T)",
        }
```

### Few-shot retrieval

```python
from nesy_core import RetrievalEngine

engine = RetrievalEngine(demo_data_path="./data/demos.json")
# demos.json: [{"episode": "...", "positive": true}, ...]

context = engine.search_demo(query=current_observation, k=3)
# returns a formatted string of 3 nearest demos to prepend to your prompt
```

---

## Key Design Patterns

### Symbolic feedback loop

When Clingo returns 0 stable models (the ASP program is unsatisfiable), the pipeline automatically reprompts the LLM with the failure signal and retries up to `MAX_VERIFICATION_TRIAL` times:

```
LLM generates rules
    → Clingo: 0 stable models
    → "The Program you created has 0 Stable models. Identify what went wrong..."
    → LLM regenerates rules
    → Clingo: satisfiable ✓
```

This is handled by `Pipeline.eval_single_plan()` for the `adapt_rule` key.

### ASP validation pipeline

LLM output is sanitised before being fed to Clingo:

```
LLM response
    → sanitize_asp()              # strip markdown fences, ensure . termination
    → keep_only_parseable_rules() # line-by-line Clingo parse check, drop bad lines
    → gen_answer_set()            # solve
```

### ILP generalisation

The full rule induction pipeline (for learning domain rules from external trajectories):

```
External trajectories (positive + negative examples)
    → gen_general_fact_response()   # trajectory → fact set (per example)
    → gen_ilp_bk_response()         # examples → background knowledge
    → gen_ilp_rule_response()       # examples + BK → ASP rules
    → sanitize_asp() + keep_only_parseable_rules()
    → save to rule_save_path
```

---

## CLEVR Integration Notes

**Dataset format** (official CLEVR v1.0):
- `questions/*.json` — `{questions: [{question, answer, image_filename, program, ...}]}`
- `scenes/*.json` — `{scenes: [{objects: [{shape, color, size, material, 3d_coords}], relationships: {left, right, front, behind}}]}`

**ASP fact schema produced by `CLEVRSemanticParser.scene_to_facts()`:**
```prolog
object(obj_0).
shape(obj_0, cube).
color(obj_0, red).
size(obj_0, large).
material(obj_0, rubber).
position(obj_0, -2.5, 0.35, 0.0).
left_of(obj_0, obj_1).
right_of(obj_1, obj_0).
```

**Answer atoms produced by `CLEVRDataset.extract_answer_from_asp()`:**
```
answer_count(N)   →  "N"
answer_attr(V)    →  "V"
answer_yes        →  "yes"
answer_no         →  "no"
answer_more       →  "more"
answer_equal      →  "equal"
answer_fewer      →  "fewer"
```

---

## TODOs / Extension Points

- [ ] `Pipeline.get_HI_score()` — wire HI-score feedback into `gen_adapt_rule_response()` for smarter rule refinement
- [ ] `RetrievalEngine.embed_all()` — add FAISS/annoy index for large-scale retrieval
- [ ] `CLEVRDataset.filter_by_type()` — return a proper view, not a mutated clone
- [ ] `CLEVRDataset.split()` — implement train/val split
- [ ] `CLEVRSemanticParser.question_to_goal()` — expand regex coverage or replace with full LLM translation
- [ ] Add `datasets/alfworld.py` — wrap existing ALFWorld env into the `NeSyDataset` interface
- [ ] Add `prompts/alfworld/` — migrate existing Alfworld prompt files here

---

## Dependencies

```
clingo
sentence-transformers
langchain
langchain-community
langchain-anthropic
langchain-google-genai
replicate
pandas
numpy
```

Install:
```bash
pip install clingo sentence-transformers langchain langchain-community \
            langchain-anthropic langchain-google-genai replicate pandas numpy
```
