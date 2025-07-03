"""
Content Creation and SEO Optimization Pipeline
"""
from google.adk.agents import LlmAgent, SequentialAgent

GEMINI_MODEL = "gemini-2.0-flash" # Or "gemini-1.0-pro-latest" or "gemini-2.0-flash"


# --- 1. Define Sub-Agents for Each Pipeline Stage ---

# 1. Content Idea Generator Agent
# Takes a broad topic and generates specific content ideas.
idea_generator_agent = LlmAgent(
    name="IdeaGeneratorAgent",
    model=GEMINI_MODEL,
    instruction="""You are a creative content idea generator.
Based on the user's provided topic, brainstorm and list 3-5 unique and engaging content ideas.
Output *only* a numbered list of ideas. Do not add any other text.
Example: '
- 10 Ways to Master Python 
- The Future of AI in Daily Life 
- Building Your First Web App with Flask
'
""",
    description="Generates initial content ideas for a given topic.",
    output_key="content_ideas" # Stores output in state['content_ideas']
)

# 2. Keyword Research Agent
# Takes the generated content ideas and suggests relevant SEO keywords for each.
# This agent would ideally use a real keyword research tool, but here it's simulated.
keyword_research_agent = LlmAgent(
    name="KeywordResearchAgent",
    model=GEMINI_MODEL,
    instruction="""You are an SEO keyword research expert.
Given a list of content ideas, identify 3-5 primary and secondary keywords for each idea that a target audience would search for.

**Content Ideas:**
{content_ideas}

**Output Format:**
For each idea, list relevant keywords.
Example:
Idea: The Future of AI in Daily Life
Keywords: AI in daily life, future AI, AI impact, personal AI, everyday artificial intelligence
---
Idea: Building Your First Web App with Flask
Keywords: Flask web app, Flask tutorial, build web app python, Python web development, Flask for beginners
""",
    description="Identifies relevant SEO keywords for content ideas.",
    output_key="seo_keywords_map", # Stores output in state['seo_keywords_map']
)

# 3. SEO Content Outline Agent
# Takes content ideas and their associated keywords to create a basic SEO-optimized outline.
seo_outline_agent = LlmAgent(
    name="SEOOutlineAgent",
    model=GEMINI_MODEL,
    instruction="""You are an SEO Content Strategist.
Based on the provided content ideas and their associated keywords, pick one and create a basic SEO-optimized outline for a blog post or article.

**Content Ideas:**
{content_ideas}

**Keywords Map:**
{seo_keywords_map}

**Outline Structure:**
For the first idea from 'Content Ideas', create an outline that includes:
1. Catchy Title (incorporating primary keyword)
2. Introduction (1 paragraph, hook, introduce topic & scope)
3. Main Sections (3-4 sections with H2 headings, each with a brief description and potential sub-topics/H3s, incorporating relevant keywords naturally)
4. Conclusion (1 paragraph, summarize, call to action)

**Output:**
Output *only* the structured outline in markdown format.
""",
    description="Creates an SEO-optimized content outline.",
    output_key="final_content_outline", # Stores output in state['final_content_outline']
)


# --- 2. Create the SequentialAgent ---
# This agent orchestrates the pipeline by running the sub-agents in order.
content_creation_pipeline = SequentialAgent(
    name="ContentCreationPipeline",
    sub_agents=[
        idea_generator_agent,
        keyword_research_agent,
        seo_outline_agent
    ],
    description="Executes a sequence for content idea generation, keyword research, and SEO outline creation.",
)

# For ADK tools compatibility and runners, the root agent must be named `root_agent`
root_agent = content_creation_pipeline