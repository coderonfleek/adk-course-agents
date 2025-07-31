import requests
from typing import Dict, Any
from google.adk.agents import LlmAgent

GEMINI_MODEL = "gemini-2.0-flash"

# --- Tool 1: Fetch User Data from JSONPlaceholder ---
def fetch_user_data(user_id: int) -> Dict[str, Any]:
    """
    Fetches user data from JSONPlaceholder API based on a user ID.

    This tool makes an HTTP GET request to the /users endpoint of JSONPlaceholder.

    Args:
        user_id (int): The ID of the user to fetch (e.g., 1 to 10).

    Returns:
        Dict[str, Any]: A dictionary containing the user's data if found,
                        otherwise a dictionary with an 'error' message.
                        The structure aligns with JSONPlaceholder's user object.
    """
    print(f"  [Tool Call] fetch_user_data called for user_id: {user_id}")
    try:
        response = requests.get(f"https://jsonplaceholder.typicode.com/users/{user_id}")
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        user_data = response.json()
        if user_data:
            return {"status": "success", "user_data": user_data}
        else:
            return {"status": "error", "message": f"User with ID {user_id} not found."}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Failed to fetch data: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}

# --- Tool 2: Format User Profile for Display ---
def format_user_profile(user_data_json: Dict[str, Any]) -> str:
    """
    Formats raw user data (from fetch_user_data) into a human-readable profile string using Markdown.

    This tool extracts key details like name, username, email, and company, and formats them
    with Markdown for improved display by the agent.

    Args:
        user_data_json (Dict[str, Any]): A dictionary containing user information,
                                         typically the 'user_data' field from the output
                                         of 'fetch_user_data'.

    Returns:
        str: A multi-line string representing the formatted user profile in Markdown.
             Returns an error message string if input data is invalid.
    """
    print(f"  [Tool Call] format_user_profile called with user data for Markdown formatting.")
    user_data = user_data_json # Extract the actual user data from the wrapper dict

    if not isinstance(user_data, dict):
        return "Error: Invalid user data provided for Markdown formatting."

    name = user_data.get("name", "N/A")
    username = user_data.get("username", "N/A")
    email = user_data.get("email", "N/A")
    phone = user_data.get("phone", "N/A").split(' ')[0] # Take only the first part if multiple numbers
    website = user_data.get("website", "N/A")
    company_name = user_data.get("company", {}).get("name", "N/A")
    city = user_data.get("address", {}).get("city", "N/A")

    markdown_profile_string = (
        f"## User Profile: {name}\n\n"  # Main heading for the profile
        f"**Username:** {username}\n\n"
        f"**Email:** {email}\n\n"
        f"**Phone:** {phone}\n\n"
        f"**Website:** {website}\n\n"
        f"**Company:** {company_name}\n\n"
        f"**Location:** {city}\n\n"
    )
    return markdown_profile_string


# --- Agent Definition: User Profile Viewer Agent ---
root_agent = LlmAgent(
    name="UserProfileViewer",
    model=GEMINI_MODEL,
    instruction=(
        "You are a helpful assistant that can retrieve and display user profiles. "
        "When asked for a user's profile by ID, first use the 'fetch_user_data' tool to get the raw data. "
        "Then, use the 'format_user_profile' tool to make it readable. "
        "Finally, present the formatted user profile to the user. "
        "If fetching or formatting fails, inform the user about the error."
    ),
    description="Retrieves and formats user profile information from a mock API.",
    tools=[fetch_user_data, format_user_profile]
)