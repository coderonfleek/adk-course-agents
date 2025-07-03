from google.adk.agents import Agent

def get_distance(from_city: str, to_city: str) -> dict:
    """
        Retrieves the information about the distance between the cities and the weather at the destination

        Args:
            from_city (str): The city the traveller is coming from
            to_city (str): The city the traveller is going to

        Returns:
            dict: distance information and the weather at the destination
    """

    if from_city.lower() == "san francisco" and to_city.lower() == "miami":
        return {
            "status": "success",
            "response": (
                "The distance between San Francisco and Miami is 345km",
                "Weather is approximately 42 degress Celcius"
            )
        }
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I do not have distance and weather information for this route"
        }
    
def get_restaurants(city: str) -> list:

    return [
        "Miami Eats",
        "Fast Fries",
        "Taco Castle"
    ]

root_agent = Agent(
    name="travel_advisor",
    model="gemini-2.0-flash",
    description = (
        "Agent to answer questions about distance between cities and restaurant suggestions"
    ),
    instruction=(
        "You're a helpful agent who can answer questions about distance between cities, weather and also give suggestions on places to eat"
    ),
    tools=[get_distance, get_restaurants]
)