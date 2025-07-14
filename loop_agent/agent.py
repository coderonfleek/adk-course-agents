"""
Recipe Optimization Pipeline.


The goal is to iteratively refine a recipe based on feedback, such as making it healthier or adjusting ingredients, until a "chef critic" is satisfied.

Use Case: Start with a basic recipe, have a "critic" agent provide feedback, and a "refiner" agent update the recipe based on the feedback, looping until the critic gives an "all clear."
"""

from google.adk.agents import LoopAgent, LlmAgent, SequentialAgent
from google.adk.tools.tool_context import ToolContext

GEMINI_MODEL = "gemini-2.0-flash"

# --- Tool Definition ---
# This tool signals the LoopAgent to exit
def exit_loop(tool_context: ToolContext):
    """Call this function ONLY when the critique indicates no further changes are needed, signaling the iterative process should end."""
    print(f"  [Tool Call] exit_loop triggered by {tool_context.agent_name}")
    tool_context.actions.escalate = True # This tells the LoopAgent to exit
    return {} # Tools should typically return JSON-serializable output


# --- Agent Definitions ---

# STEP 1: Initial Recipe Writer Agent (Runs ONCE at the beginning)
initial_recipe_writer_agent = LlmAgent(
    name="InitialRecipeWriterAgent",
    model=GEMINI_MODEL,
    instruction="""You are a basic Recipe Writer.
Write a *simple, initial draft* of a recipe for a meal entered by the user.
Include very basic ingredients and steps (3-5 items each).

Output *only* the recipe text. Do not add introductions or explanations.
""",
    description="Writes the initial recipe draft based on the provided dish topic.",
    output_key="current_recipe" # Stores output in state['current_recipe']
)

# STEP 2a: Recipe Critic Agent (Inside the Refinement Loop)
critic_agent_in_loop = LlmAgent(
    name="RecipeCriticAgent",
    model=GEMINI_MODEL,
    instruction="""You are an expert Culinary Critic AI.
You are reviewing a recipe draft and providing constructive feedback for improvement.

**Recipe to Review:**
{current_recipe}

**Task:**
Review the recipe for clarity, common sense, and potential for improvement (e.g., healthiness, flavor enhancement, missing details).

IF you identify 1-2 *clear and actionable* suggestions for improvement (e.g., "Add more vegetables", "Specify cooking temperature", "Suggest a healthier alternative for butter"):
Provide these specific suggestions concisely. Output *only* the critique text.

ELSE IF the recipe is clear, functional, and requires no obvious major improvements for its basic form:
Respond *exactly* with the phrase "Recipe looks great!" and nothing else. Avoid purely subjective stylistic preferences if the core recipe is sound.

Do not add explanations. Output only the critique OR the exact completion phrase.
""",
    description="Reviews the current recipe draft, providing critique or signaling completion.",
    output_key="critique_feedback" # Stores output in state['critique_feedback']
)

# STEP 2b: Recipe Refiner Agent (Inside the Refinement Loop)
refiner_agent_in_loop = LlmAgent(
    name="RecipeRefinerAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Recipe Refinement Assistant.
Your goal is to improve the given recipe based on the provided critique OR to exit the process.

**Current Recipe:**
{current_recipe}

**Critique/Suggestions:**
{critique_feedback}

**Task:**
Analyze the 'Critique/Suggestions'.
IF the critique is *exactly* "Recipe looks great!":
You MUST call the 'exit_loop' function. Do not output any text.
ELSE (the critique contains actionable feedback):
Carefully apply the suggestions to improve the 'Current Recipe'. Output *only* the refined recipe text.

Do not add explanations. Either output the refined recipe OR call the exit_loop function.
""",
    description="Refines the recipe based on critique, or calls exit_loop if critique indicates completion.",
    tools=[exit_loop], # Provide the exit_loop tool
    output_key="current_recipe" # Overwrites state['current_recipe'] with the refined version
)

# STEP 2: Recipe Refinement Loop Agent
refinement_loop = LoopAgent(
    name="RecipeRefinementLoop",
    # Agent order is crucial: Critic first, then Refine/Exit
    sub_agents=[
        critic_agent_in_loop,
        refiner_agent_in_loop,
    ],
    max_iterations=5 # Limit loops to prevent infinite loops
)

# STEP 3: Overall Sequential Pipeline
# This agent orchestrates the initial writing and then the iterative refinement.
root_agent = SequentialAgent(
    name="RecipeOptimizationPipeline",
    sub_agents=[
        initial_recipe_writer_agent, # Run first to create initial recipe
        refinement_loop              # Then run the critique/refine loop
    ],
    description="Writes an initial recipe and then iteratively refines it based on feedback."
)