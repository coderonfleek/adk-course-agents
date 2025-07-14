"""
Multi-Platform Social Media Content Draft Generator.

This pipeline will generate different types of content (a tweet, an Instagram caption, and a blog post introduction) for a given topic simultaneously, and then combine them into a single summary.
"""

import os
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.genai import types

# --- Setup for Model ---
# Replace with your actual model name and ensure your environment is set up
# for authentication (e.g., GOOGLE_API_KEY or Google Cloud project credentials).
GEMINI_MODEL = "gemini-2.0-flash" # Or "gemini-1.0-pro-latest" / "gemini-2.0-flash"

# --- 1. Define Content Creator Sub-Agents (to run in parallel) ---
# Each agent takes the initial topic and generates content for a specific platform.

# Agent 1: Tweet Generator
tweet_generator_agent = LlmAgent(
    name="TweetGenerator",
    model=GEMINI_MODEL,
    instruction="""You are a Twitter content creator.
Based *only* on the topic provided, write a concise and engaging tweet (max 280 characters).
Include relevant hashtags.

Output *only* the complete tweet text. Do not add any other text.

""",
    description="Generates a tweet with hashtags for a given topic.",
    output_key="generated_tweet" # Stores output in state['generated_tweet']
)

# Agent 2: Instagram Caption Generator
instagram_caption_agent = LlmAgent(
    name="InstagramCaptionGenerator",
    model=GEMINI_MODEL,
    instruction="""You are an Instagram content creator.
Based *only* on the topic provided, write a short, engaging Instagram caption.
Include 3-5 relevant and popular hashtags.

Output *only* the complete Instagram caption. Do not add any other text.
""",
    description="Generates an Instagram caption with hashtags.",
    output_key="generated_instagram_caption" # Stores output in state['generated_instagram_caption']
)

# Agent 3: Blog Post Introduction Generator
blog_intro_agent = LlmAgent(
    name="BlogPostIntroGenerator",
    model=GEMINI_MODEL,
    instruction="""You are a blog writer.
Based *only* on the topic provided, write a compelling and informative introductory paragraph (3-5 sentences) for a blog post.
The introduction should hook the reader and clearly state what the post will cover.

Output *only* the introductory paragraph. Do not add any other text.
""",
    description="Generates an introductory paragraph for a blog post.",
    output_key="generated_blog_intro" # Stores output in state['generated_blog_intro']
)

# --- 2. Create the ParallelAgent (Runs content creators concurrently) ---
# This agent orchestrates the concurrent execution of the content generators.
# It finishes once all sub-agents have completed and stored their results in state.
parallel_content_agent = ParallelAgent(
    name="ParallelSocialMediaContent",
    sub_agents=[
        tweet_generator_agent,
        instagram_caption_agent,
        blog_intro_agent
    ],
    description="Runs multiple content generation agents in parallel for different platforms."
)

# --- 3. Define the Consolidator Agent (Runs *after* the parallel agents) ---
# This agent takes the results stored in the session state by the parallel agents
# and synthesizes them into a single, structured summary.
content_consolidator_agent = LlmAgent(
    name="ContentConsolidator",
    model=GEMINI_MODEL,
    instruction="""You are a Content Editor.
Your primary task is to consolidate the generated social media content drafts into a single, organized summary.

**Input Drafts:**

* **Tweet:**
    {generated_tweet}

* **Instagram Caption:**
    {generated_instagram_caption}

* **Blog Post Introduction:**
    {generated_blog_intro}

**Output Format:**

## Social Media Content Draft Summary

### Twitter Draft
[Insert the generated tweet here]

### Instagram Draft
[Insert the generated Instagram caption here]

### Blog Post Introduction Draft
[Insert the generated blog post introduction here]

Output *only* the structured summary following this format. Do not include any other text.
""",
    description="Consolidates parallel content drafts into a single summary.",
    # No output_key needed here, as its direct response is the final output of the root Sequential Agent
)


# --- 4. Create the SequentialAgent (Orchestrates the overall flow) ---
# This is the main agent that will be run. It first executes the ParallelAgent
# to populate the state, and then executes the ConsolidatorAgent to produce the final output.
root_agent = SequentialAgent(
    name="MultiPlatformContentPipeline",
    sub_agents=[parallel_content_agent, content_consolidator_agent],
    description="Coordinates parallel content generation and synthesizes the results."
)

