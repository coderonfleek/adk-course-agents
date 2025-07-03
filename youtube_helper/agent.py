from google.adk.agents import Agent
# Import for model configuration
from google.genai import types

def generate_niche_video_ideas(channel_niche: str, num_ideas: int = 3) -> list:
    """
    Generates creative video ideas tailored to a specific YouTube channel niche.

    Args:
        channel_niche (str): The primary topic or genre of the YouTube channel (e.g., 'tech reviews', 'cooking tutorials', 'gaming news').
        num_ideas (int): The number of video ideas to generate.

    Returns:
        list: A list of string representing unique video ideas.
    """
    # In a real scenario, this would involve more sophisticated logic,
    # possibly calling an LLM or an external idea generation API.
    if "tech reviews" in channel_niche.lower():
        ideas = [
            f"Top 5 Gadgets Under $100 for {channel_niche} in 2025",
            f"Deep Dive: The Latest AI Smartphone Features",
            f"DIY Smart Home Setup Guide for {channel_niche}",
            f"Budget Gaming PC Build: Max Performance, Min Cost"
        ]
    elif "cooking tutorials" in channel_niche.lower():
        ideas = [
            f"Quick & Easy 30-Minute Meals: {channel_niche} Edition",
            f"Mastering Sourdough Bread: A Beginner's Guide",
            f"Global Street Food Recipes You Can Make At Home",
            f"Ultimate Guide to Meal Prepping for the Week"
        ]
    else:
        ideas = [
            f"Exploring the Future of {channel_niche}",
            f"The Ultimate Guide to {channel_niche} Basics",
            f"Behind the Scenes of a {channel_niche} Creator",
            f"Top 10 Tips for Mastering {channel_niche}"
        ]
    return ideas[:num_ideas]

def get_channel_optimization_tips(channel_type: str) -> dict:
    """
    Provides general tips for improving a YouTube channel's performance and reach.

    Args:
        channel_type (str): The type or category of the YouTube channel (e.g., 'vlog', 'education', 'entertainment').

    Returns:
        dict: A dictionary containing optimization tips for various aspects.
    """
    tips = {
        "SEO": [
            "Use relevant keywords in titles, descriptions, and tags.",
            "Optimize video thumbnails for click-through rate.",
            "Create compelling video descriptions with timestamps."
        ],
        "Engagement": [
            "Encourage comments and questions.",
            "Add end screens and cards to promote other videos.",
            "Respond to comments to build community."
        ],
        "Content Strategy": [
            "Analyze audience retention reports to understand viewer behavior.",
            "Research trending topics in your niche.",
            "Maintain a consistent upload schedule."
        ]
    }
    return {"status": "success", "tips": tips}

# You could also integrate a tool like a mock web scraper for trending topics
# or a tool that queries a hypothetical YouTube Analytics API.
# For simplicity, we're using the two functions above.

root_agent = Agent(
    name="youtube_creator_assistant",
    model="gemini-2.0-flash",
    description=(
        "An AI assistant designed to help YouTube creators generate video ideas and optimize their channels."
    ),
    instruction=(
        "You are a helpful YouTube expert. Your goal is to provide creative video content ideas "
        "based on channel niches and offer actionable advice for improving YouTube channel performance, "
        "including SEO, engagement, and content strategy."
    ),
    tools=[
        generate_niche_video_ideas,
        get_channel_optimization_tips,
    ],
    # Configure Underlying Model
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=250
    )
)